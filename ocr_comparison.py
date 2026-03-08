import os
import re
import json
from typing import Dict, List

import pandas as pd

# ============================================================
# CONFIG
# ============================================================
OUTPUT_ROOT = r"outputs\ovarian_cancer"
OUTPUT_JSON = r"outputs\ovarian_cancer_json"
RESULTS_CSV = r"outputs\openai_vs_qwen_comparison.csv"

RESULT_COLUMNS = [
    "Patient_ID",
    "Model",
    "WER",
    "CER",
    "Latency (s)",
    "Confidence",
]

# ============================================================
# HELPERS
# ============================================================
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def levenshtein_distance(seq1, seq2):
    n, m = len(seq1), len(seq2)

    if n < m:
        seq1, seq2 = seq2, seq1
        n, m = m, n

    previous = list(range(m + 1))

    for i in range(1, n + 1):
        current = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
        previous = current

    return previous[m]


def calculate_wer_cer(qwen_text: str, openai_text: str):
    qwen_norm = normalize_text(qwen_text)
    openai_norm = normalize_text(openai_text)

    qwen_words = qwen_norm.split()
    openai_words = openai_norm.split()

    word_edits = levenshtein_distance(qwen_words, openai_words)
    char_edits = levenshtein_distance(list(qwen_norm), list(openai_norm))

    return word_edits, char_edits


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_file_map(folder: str) -> Dict[str, Dict[str, str]]:
    """
    Maps:
      TCGA-XXXX -> {"openai": path, "qwen": path}
    """
    mapping: Dict[str, Dict[str, str]] = {}

    for filename in os.listdir(folder):
        lower = filename.lower()

        if not lower.endswith(".txt"):
            continue

        full_path = os.path.join(folder, filename)

        if lower.endswith("_openai.txt"):
            patient_id = filename[:-11]   # removes "_openai.txt"
            mapping.setdefault(patient_id, {})["openai"] = full_path

        elif lower.endswith("_qwen.txt"):
            patient_id = filename[:-9]    # removes "_qwen.txt"
            mapping.setdefault(patient_id, {})["qwen"] = full_path

    return mapping


# ============================================================
# MAIN
# ============================================================
def main():
    file_map = build_file_map(OUTPUT_ROOT)

    rows: List[Dict] = []

    for patient_id in sorted(file_map.keys()):
        pair = file_map[patient_id]

        if "openai" not in pair or "qwen" not in pair:
            continue

        openai_text = load_text(pair["openai"])
        qwen_text = load_text(pair["qwen"])

        wer, cer = calculate_wer_cer(qwen_text, openai_text)

        latency = ""
        confidence = ""

        json_path = os.path.join(OUTPUT_JSON, f"{patient_id}_openai_ocr.json")

        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            latency = data.get("metadata", {}).get("total_time_seconds", "")

            scores = []
            for page in data.get("pages", []):
                score = page.get("best_response", {}).get("composite_score")
                if score is not None:
                    scores.append(float(score))

            if scores:
                confidence = round(sum(scores) / len(scores) * 100, 2)

        rows.append({
            "Patient_ID": patient_id,
            "Model": "gpt-4.1",
            "WER": wer,
            "CER": cer,
            "Latency (s)": latency,
            "Confidence": confidence,
        })

    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    df.to_csv(RESULTS_CSV, index=False)

    print(f"Saved: {RESULTS_CSV}")
    print(f"Compared {len(rows)} file pairs")


if __name__ == "__main__":
    main()