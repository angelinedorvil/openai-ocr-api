import os
import csv
from datetime import datetime
from typing import Dict, List, Optional
import json
import re

import pandas as pd

from openai_api_ocr import ocr_pdf_api, MODEL_NAME


# ============================================================
# CONFIG
# ============================================================
BENCHMARK_CSV = r"gemini_ocr_benchmark(in).csv"
OVARIAN_ROOT = r"input_pdfs\ovarian_cancer"

OUTPUT_ROOT = r"outputs\ovarian_cancer"
OUTPUT_JSON_ROOT = r"outputs\ovarian_cancer\json"
RESULTS_CSV = r"outputs\openai_ocr_results.csv"

LOGS_ROOT = r"logs"

# keep same exact schema as Gemini benchmark CSV
RESULT_COLUMNS = [
    "Patient_ID",
    "Model",
    "WER",
    "CER",
    "Latency (s)",
    "Stage_Match",
    "Confidence",
]


# ============================================================
# HELPERS
# ============================================================
def log(msg: str, log_path: str) -> None:
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def load_benchmark_patient_ids(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)
    if "Patient_ID" not in df.columns:
        raise ValueError("Benchmark CSV is missing required column: Patient_ID")

    ids = (
        df["Patient_ID"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    return sorted(ids)


def find_patient_folder_map(root: str, benchmark_ids: List[str]) -> Dict[str, str]:
    """
    Walk the ovarian folder tree and map patient folder name -> absolute path
    only for benchmark patient IDs.
    """
    benchmark_set = set(benchmark_ids)
    mapping = {}

    for dirpath, dirnames, filenames in os.walk(root):
        folder_name = os.path.basename(dirpath).strip()
        if folder_name in benchmark_set:
            mapping[folder_name] = dirpath

    return mapping


def find_first_pdf_in_folder(folder_path: str) -> Optional[str]:
    for dirpath, _, filenames in os.walk(folder_path):
        for fn in filenames:
            if fn.lower().endswith(".pdf"):
                return os.path.join(dirpath, fn)
    return None

def find_first_tsv_in_folder(folder_path: str) -> Optional[str]:
    for dirpath, _, filenames in os.walk(folder_path):
        for fn in filenames:
            if fn.lower().endswith(".tsv"):
                return os.path.join(dirpath, fn)
    return None

def load_ground_truth_text(tsv_path: str) -> str:
    df = pd.read_csv(tsv_path, sep="\t", dtype=str).fillna("")
    # Join all cell text into one string
    text_parts = []
    for col in df.columns:
        text_parts.extend(df[col].astype(str).tolist())
    return "\n".join(text_parts).strip()

def load_existing_results(csv_path: str) -> Dict[str, Dict]:
    """
    Load prior OpenAI results so reruns can resume cleanly.
    """
    if not os.path.exists(csv_path):
        return {}

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    if "Patient_ID" not in df.columns:
        return {}

    records = {}
    for _, row in df.iterrows():
        records[str(row["Patient_ID"]).strip()] = row.to_dict()
    return records


def save_results_csv(rows: List[Dict], csv_path: str) -> None:
    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    df.to_csv(csv_path, index=False)


def make_blank_result_row(patient_id: str) -> Dict:
    return {
        "Patient_ID": patient_id,
        "Model": MODEL_NAME,
        "WER": "",
        "CER": "",
        "Latency (s)": "",
        "Stage_Match": "",
        "Confidence": "",
    }

def load_openai_json_summary(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pages = data.get("pages", [])
    page_texts = []
    composite_scores = []

    for page in pages:
        best = page.get("best_response", {})
        text = best.get("text", "")
        if text:
            page_texts.append(text)

        score = best.get("composite_score")
        if score is not None:
            composite_scores.append(float(score))

    full_text = "\n".join(page_texts).strip()
    confidence = round(sum(composite_scores) / len(composite_scores), 4) if composite_scores else ""

    return full_text, confidence

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def levenshtein_distance(seq1, seq2):
    n, m = len(seq1), len(seq2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[n][m]


def calculate_wer_cer(ground_truth: str, predicted_text: str):
    gt_norm = normalize_text(ground_truth)
    pred_norm = normalize_text(predicted_text)

    gt_words = gt_norm.split()
    pred_words = pred_norm.split()

    word_edits = levenshtein_distance(gt_words, pred_words)
    char_edits = levenshtein_distance(list(gt_norm), list(pred_norm))

    wer = word_edits / max(len(gt_words), 1)
    cer = char_edits / max(len(gt_norm), 1)

    return round(wer, 4), round(cer, 4)

# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(OUTPUT_JSON_ROOT, exist_ok=True)
    os.makedirs(LOGS_ROOT, exist_ok=True)

    log_path = os.path.join(
        LOGS_ROOT,
        f"ocr_comparison_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    log("=" * 80, log_path)
    log("OPENAI OCR COMPARISON RUN START", log_path)
    log(f"Benchmark CSV: {BENCHMARK_CSV}", log_path)
    log(f"Ovarian root: {OVARIAN_ROOT}", log_path)
    log(f"Results CSV: {RESULTS_CSV}", log_path)
    log("=" * 80, log_path)

    benchmark_ids = load_benchmark_patient_ids(BENCHMARK_CSV)
    log(f"Loaded {len(benchmark_ids)} benchmark patient IDs", log_path)

    patient_folder_map = find_patient_folder_map(OVARIAN_ROOT, benchmark_ids)
    log(f"Found {len(patient_folder_map)} matching patient folders under ovarian root", log_path)

    existing_results = load_existing_results(RESULTS_CSV)
    results_by_patient = dict(existing_results)  # resume-friendly

    for idx, patient_id in enumerate(benchmark_ids, 1):
        log("\n" + "-" * 80, log_path)
        log(f"[{idx}/{len(benchmark_ids)}] Patient_ID: {patient_id}", log_path)

        patient_folder = patient_folder_map.get(patient_id)
        if not patient_folder:
            log(f"[WARN] No matching folder found for patient {patient_id}", log_path)
            if patient_id not in results_by_patient:
                results_by_patient[patient_id] = make_blank_result_row(patient_id)
            continue

        pdf_path = find_first_pdf_in_folder(patient_folder)
        tsv_path = find_first_tsv_in_folder(patient_folder)
        if not tsv_path:
            log(f"[WARN] No TSV found inside folder: {patient_folder}", log_path)
            if patient_id not in results_by_patient:
                results_by_patient[patient_id] = make_blank_result_row(patient_id)
            continue
        if not pdf_path:
            log(f"[WARN] No PDF found inside folder: {patient_folder}", log_path)
            if patient_id not in results_by_patient:
                results_by_patient[patient_id] = make_blank_result_row(patient_id)
            continue

        # use patient ID for output names so matching stays clean
        out_txt = os.path.join(OUTPUT_ROOT, f"{patient_id}_openai.txt")
        out_json = os.path.join(OUTPUT_JSON_ROOT, f"{patient_id}_openai_ocr.json")

        # if already processed and already in results CSV, skip rerun
        already_has_csv = (
            patient_id in results_by_patient
            and results_by_patient[patient_id].get("Latency (s)", "") != ""
        )

        if os.path.exists(out_json) and already_has_csv:
            log(f"[SKIP] Existing OCR output + existing CSV row found for {patient_id}", log_path)
            continue

        log(f"[START] OCR for {patient_id}", log_path)
        log(f"PDF: {pdf_path}", log_path)
        log(f"TXT: {out_txt}", log_path)
        log(f"JSON: {out_json}", log_path)

        try:
            if not os.path.exists(out_json):
                result = ocr_pdf_api(
                    pdf_path=pdf_path,
                    out_txt_path=out_txt,
                    out_json_path=out_json,
                    dpi=300,
                    attempts_per_page=3,
                    use_cot=True,
                    max_pages=None,
                )
                latency = result["summary"].get("total_time_seconds", "")
                if latency != "":
                    latency = round(float(latency), 3)
            else:
                log(f"[REUSE] Existing OpenAI OCR output found for {patient_id}", log_path)
                latency = results_by_patient.get(patient_id, {}).get("Latency (s)", "")

            predicted_text, confidence = load_openai_json_summary(out_json)
            ground_truth_text = load_ground_truth_text(tsv_path)
            wer, cer = calculate_wer_cer(ground_truth_text, predicted_text)

            results_by_patient[patient_id] = {
                "Patient_ID": patient_id,
                "Model": MODEL_NAME,
                "WER": wer,
                "CER": cer,
                "Latency (s)": latency,
                "Stage_Match": "",
                "Confidence": confidence,
            }

            # save after each patient so reruns can resume cleanly
            ordered_rows = [results_by_patient[pid] for pid in sorted(results_by_patient.keys())]
            save_results_csv(ordered_rows, RESULTS_CSV)

            log(f"[DONE] {patient_id}", log_path)
            log(f"Latency (s): {latency}", log_path)

        except Exception as e:
            log(f"[ERROR] Failed on {patient_id}: {e}", log_path)

            # still keep a row so the CSV remains aligned to benchmark IDs
            if patient_id not in results_by_patient:
                results_by_patient[patient_id] = make_blank_result_row(patient_id)

            ordered_rows = [results_by_patient[pid] for pid in sorted(results_by_patient.keys())]
            save_results_csv(ordered_rows, RESULTS_CSV)

    log("\n" + "=" * 80, log_path)
    log("OPENAI OCR COMPARISON RUN END", log_path)
    log(f"Final results written to: {RESULTS_CSV}", log_path)
    log("=" * 80, log_path)


if __name__ == "__main__":
    main()