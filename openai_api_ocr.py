import os
import io
import time
import base64
import sys
import json
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import fitz  # PyMuPDF
from PIL import Image
from openai import OpenAI

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "gpt-4.1-mini"  # vision-capable model
TEMPERATURE_SCHEDULE = [0.1, 0.2, 0.3]

QUALITY_SCORE_THRESHOLD = 0.5
PERPLEXITY_THRESHOLD = 20.0
MIN_PROB_THRESHOLD = 0.05

DEFAULT_DPI = 300
DEFAULT_MAX_OUTPUT_TOKENS = 2048
DEBUG = False

LIMITS = {
        "high_quality": 15,
        "low_quality": 15,
        "medium_quality": 20,
    }

client = OpenAI()


# -----------------------------
# PDF -> file paths
# -----------------------------
def iter_pdfs_recursive(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".pdf"):
                yield os.path.join(dirpath, fn)

class FileOnlyLogger:
    """Redirect stdout/stderr to a file only."""
    def __init__(self, f):
        self.f = f

    def write(self, data):
        self.f.write(data)
        self.f.flush()

    def flush(self):
        self.f.flush()

# -----------------------------
# PDF -> Images
# -----------------------------
def pdf_to_images(pdf_path: str, dpi: int = DEFAULT_DPI) -> List[Image.Image]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    images: List[Image.Image] = []

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    try:
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    finally:
        doc.close()

    return images


def preprocess_image_for_ocr(image: Image.Image, max_size: int = 2048) -> Image.Image:
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return image


def pil_to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

# -----------------------------
# Confidence from API logprobs
# -----------------------------
def confidence_from_logprobs(token_logprobs: List[float]) -> Dict:
    if not token_logprobs:
        return {
            "mean_probability": None,
            "mean_log_probability": None,
            "perplexity": None,
            "min_probability": None
        }

    lps = np.array(token_logprobs, dtype=np.float64)
    probs = np.exp(lps)
    mean_lp = float(np.mean(lps))

    return {
        "mean_probability": float(np.mean(probs)),
        "mean_log_probability": mean_lp,
        "perplexity": float(np.exp(-mean_lp)),
        "min_probability": float(np.min(probs))
    }

# -----------------------------
# Scoring / selection
# -----------------------------
def calculate_composite_score(chars: int, confidence: Dict, temperature: float, expected_chars: int) -> float:
    if confidence.get("perplexity") is None:
        return 0.0

    perplexity = confidence["perplexity"]
    min_prob = confidence.get("min_probability", 0) or 0.0

    normalized_length = min(chars / 2000.0, 1.0)
    quality_score = 1.0 / (1.0 + np.log(max(perplexity, 1.0)))

    min_prob_penalty = 1.0 if min_prob > MIN_PROB_THRESHOLD else 0.7
    temperature_penalty = 1.0 if temperature <= 0.1 else 0.95

    # length outlier penalty (prevents runaway long attempts)
    
    length_penalty = 1.0
    if expected_chars > 0 and chars > int(1.8 * expected_chars):
        length_penalty = 0.6

    return ((0.80 * quality_score) + (0.20 * normalized_length)) * min_prob_penalty * temperature_penalty * length_penalty


def select_best_ocr(all_responses: List[Dict]) -> Dict:
    scored = []
    char_counts = [r["chars"] for r in all_responses]
    expected_chars = int(np.median(char_counts))

    for r in all_responses:
        score = calculate_composite_score(r["chars"], r["confidence"], r["temperature"], expected_chars)
        r["composite_score"] = score
        scored.append((score, r))

    best_score, best = max(scored, key=lambda x: x[0])

    
    quality_warnings = []
    perplexity = best["confidence"].get("perplexity", float("inf")) if best.get("confidence") else float("inf")
    min_prob = best["confidence"].get("min_probability", 0) if best.get("confidence") else 0

    if best_score < QUALITY_SCORE_THRESHOLD:
        quality_warnings.append(f"Low composite quality score: {best_score:.3f} (threshold: {QUALITY_SCORE_THRESHOLD})")
    if perplexity is not None and perplexity > PERPLEXITY_THRESHOLD:
        quality_warnings.append(f"High perplexity (uncertainty): {perplexity:.2f} (threshold: {PERPLEXITY_THRESHOLD})")
    if min_prob is not None and min_prob < MIN_PROB_THRESHOLD:
        quality_warnings.append(f"Very uncertain tokens: min_prob={min_prob:.4f} (threshold: {MIN_PROB_THRESHOLD})")

    
    if len(all_responses) > 1:
        char_counts = [r["chars"] for r in all_responses]
        cv = np.std(char_counts) / np.mean(char_counts) if np.mean(char_counts) > 0 else 0
        if cv > 0.3:
            quality_warnings.append(f"High variance between attempts: {cv:.1%} - results inconsistent")

    best["quality_warning"] = len(quality_warnings) > 0
    best["quality_warnings"] = quality_warnings
    if quality_warnings:
        best["warning_message"] = "QUALITY ALERT - Manual review recommended:\n" + "\n".join(f"  - {w}" for w in quality_warnings)

    return best

# -----------------------------
# OpenAI API call (vision OCR)
# -----------------------------
def ocr_via_api_once(
    image: Image.Image,
    prompt: str,
    temperature: float,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    model: str = MODEL_NAME,
) -> Dict:
    data_url = pil_to_data_url(image)
    start = time.time()

    resp = client.responses.create(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        include=["message.output_text.logprobs"],
        input=[{
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": data_url},
                {"type": "input_text", "text": prompt},
            ],
        }],
    )

    if DEBUG:
        print("DEBUG: resp.output types:", [item.type for item in resp.output])
        for item in resp.output:
            if item.type == "message":
                for c in item.content:
                    print("DEBUG: content type:", getattr(c, "type", None), "has logprobs:", hasattr(c, "logprobs"))
                    if hasattr(c, "logprobs") and c.logprobs:
                        lp0 = c.logprobs[0]
                        print("DEBUG: first logprob entry type:", type(lp0))
                        print("DEBUG: first logprob entry attrs:", [a for a in dir(lp0) if not a.startswith("_")][:25])
                        print("DEBUG: first logprob entry logprob:", getattr(lp0, "logprob", None))
                        print("DEBUG: first logprob entry token:", getattr(lp0, "token", None))

    elapsed = time.time() - start
    text = getattr(resp, "output_text", "") or ""

    token_logprobs: List[float] = []
    try:
        for item in resp.output:
            if item.type == "message":
                for c in item.content:
                    lp_list = getattr(c, "logprobs", None)
                    if lp_list:
                        for lp in lp_list:
                            lpv = getattr(lp, "logprob", None)
                            if lpv is not None:
                                token_logprobs.append(float(lpv))
    except Exception:
        token_logprobs = []
        pass

    usage = getattr(resp, "usage", None)
    return {"text": text, "time": elapsed, "usage": usage, "token_logprobs": token_logprobs}

# -----------------------------
# OCR with retry (page)
# -----------------------------
def ocr_with_retry_api(
    image: Image.Image,
    page_num: int,
    attempts: int = 3,
    use_cot: bool = True
) -> Dict:
    if use_cot:
        prompt = (
            "Extract ALL text from this medical pathology report image exactly as shown.\n"
            "Preserve formatting, line breaks, tables, numbers, and special characters.\n"
            "Do not summarize. Do not interpret. Only transcribe."
        )
    else:
        prompt = "Transcribe all text exactly as shown, preserving formatting."

    all_responses: List[Dict] = []

    print(f"\nPage {page_num} - {attempts} attempts")
    for i in range(attempts):
        temp = TEMPERATURE_SCHEDULE[i % len(TEMPERATURE_SCHEDULE)]
        print(f"  Attempt {i+1}/{attempts} (temp={temp}) ...", end=" ")

        try:
            r = ocr_via_api_once(image=image, prompt=prompt, temperature=temp)
            out_text = r["text"]
            conf = confidence_from_logprobs(r["token_logprobs"])
            chars = len(out_text)
            words = len(out_text.split())

            entry = {
                "attempt": i + 1,
                "temperature": temp,
                "time": r["time"],
                "chars": chars,
                "words": words,
                "text": out_text,
                "confidence": conf,
                "usage": None
            }

            
            if r["usage"] is not None:
                try:
                    entry["usage"] = {
                        "input_tokens": r["usage"].input_tokens,
                        "output_tokens": r["usage"].output_tokens,
                        "total_tokens": r["usage"].total_tokens,
                    }
                except Exception:
                    entry["usage"] = None

            perp = conf["perplexity"]
            print(f"ok | chars={chars} | perplexity={perp:.2f}" if perp is not None else f"ok | chars={chars} | perplexity=N/A")
            all_responses.append(entry)

        except Exception as e:
            print(f"failed: {e}")
            continue

    if not all_responses:
        return {"error": f"All OCR attempts failed on page {page_num}"}

    best = select_best_ocr(all_responses)
    return {"page_num": page_num, "best_response": best, "all_responses": all_responses, "total_attempts": len(all_responses)}

# -----------------------------
# OCR full PDF (with crash-safe saving per page)
# -----------------------------
def ocr_pdf_api(
    pdf_path: str,
    out_txt_path: str,
    out_json_path: Optional[str] = None,
    dpi: int = DEFAULT_DPI,
    attempts_per_page: int = 3,
    use_cot: bool = True,
    max_pages: Optional[int] = None
) -> Dict:
    images = pdf_to_images(pdf_path, dpi=dpi)
    if max_pages:
        images = images[:max_pages]

    results_pages: List[Dict] = []
    total_start = time.time()

    os.makedirs(os.path.dirname(out_txt_path) or ".", exist_ok=True)
    if out_json_path:
        os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)

    
    with open(out_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("OPENAI API OCR OUTPUT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Source PDF: {pdf_path}\n")
        f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

    for idx, img in enumerate(images, 1):
        processed = preprocess_image_for_ocr(img)
        page_result = ocr_with_retry_api(processed, page_num=idx, attempts=attempts_per_page, use_cot=use_cot)
        results_pages.append(page_result)

        if "best_response" in page_result:
            text = page_result["best_response"]["text"]
            with open(out_txt_path, "a", encoding="utf-8") as f:
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"PAGE {idx}\n")
                f.write("=" * 80 + "\n")
                f.write(text + "\n")
        else:
            with open(out_txt_path, "a", encoding="utf-8") as f:
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"PAGE {idx} - ERROR\n")
                f.write("=" * 80 + "\n")
                f.write(page_result.get("error", "Unknown error") + "\n")

        if out_json_path:
            try:
                with open(out_json_path, "w", encoding="utf-8") as jf:
                    json.dump({"pdf": pdf_path, "pages": results_pages}, jf, indent=2, ensure_ascii=False)
            except Exception:
                pass

    total_time = time.time() - total_start

    total_in = 0
    total_out = 0
    for p in results_pages:
        if "best_response" in p and p["best_response"].get("usage"):
            u = p["best_response"]["usage"]
            total_in += u.get("input_tokens", 0)
            total_out += u.get("output_tokens", 0)

    summary = {
        "pdf_path": pdf_path,
        "total_pages": len(images),
        "attempts_per_page": attempts_per_page,
        "total_time_seconds": total_time,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "output_txt": out_txt_path,
        "output_json": out_json_path,
    }

    return {"summary": summary, "pages": results_pages}

# if __name__ == "__main__":
#     input_folder = r"input_pdfs"
#     output_folder = r"outputs"   
#     logs_folder = r"logs"

#     os.makedirs(logs_folder, exist_ok=True)
#     os.makedirs(os.path.join(output_folder, "json"), exist_ok=True)

#     log_path = os.path.join(
#         logs_folder,
#         f"ocr_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
#     )

#     with open(log_path, "a", encoding="utf-8") as logf:
        
#         old_stdout, old_stderr = sys.stdout, sys.stderr
#         sys.stdout = FileOnlyLogger(logf)
#         sys.stderr = FileOnlyLogger(logf)

#         try:
#             print("=" * 80)
#             print("OPENAI OCR RUN START")
#             print(f"Timestamp: {datetime.now().isoformat()}")
#             print(f"Input folder: {input_folder}")
#             print(f"Output folder: {output_folder}")
#             print(f"Limits: {LIMITS}")
#             print("=" * 80)

#             for quality, limit in LIMITS.items():
#                 pdf_root = os.path.join(input_folder, quality)
#                 if not os.path.exists(pdf_root):
#                     print(f"[WARN] Missing folder: {pdf_root}")
#                     continue

#                 print("\n" + "-" * 80)
#                 print(f"QUALITY: {quality} | limit={limit}")
#                 print("-" * 80)

#                 processed = 0

#                 for pdf_path in iter_pdfs_recursive(pdf_root):
#                     if processed >= limit:
#                         break

#                     base_name = os.path.splitext(os.path.basename(pdf_path))[0]
#                     out_txt = os.path.join(output_folder, f"{base_name}.txt")
#                     out_json = os.path.join(output_folder, "json", f"{base_name}_ocr.json")

#                     if os.path.exists(out_txt):
#                         print(f"\n[SKIP] {pdf_path} already processed (TXT exists)")
#                         continue

#                     print(f"\n[START] {pdf_path}")
#                     print(f"  -> TXT:  {out_txt}")
#                     print(f"  -> JSON: {out_json}")

#                     try:
#                         result = ocr_pdf_api(
#                             pdf_path=pdf_path,
#                             out_txt_path=out_txt,
#                             out_json_path=out_json,
#                             dpi=300,
#                             attempts_per_page=3,
#                             use_cot=True,
#                             max_pages=10,   # keep your cost-control cap; set None later
#                         )
#                         print("[DONE]")
#                         print(result["summary"])
#                         processed += 1

#                     except Exception as e:
#                         print("[ERROR]")
#                         print(f"Exception: {e}")
#                         print(traceback.format_exc())

#                 print(f"\n[SUMMARY] {quality}: processed {processed}/{limit}")

#             print("\n" + "=" * 80)
#             print("OPENAI OCR RUN END")
#             print(f"Finished: {datetime.now().isoformat()}")
#             print("=" * 80)

#         finally:
#             # Restore normal stdout/stderr so your terminal isn't “silent” 
#             sys.stdout, sys.stderr = old_stdout, old_stderr

    
#     print(f"Log written to: {log_path}")

if __name__ == "__main__":
    input_folder = r"reports"
    output_folder = r"outputs\ovarian_cancer"
    logs_folder = r"logs"

    os.makedirs(logs_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(os.path.join(output_folder, "json"), exist_ok=True)

    log_path = os.path.join(
        logs_folder,
        f"ocr_run_ovarian_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    with open(log_path, "a", encoding="utf-8") as logf:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = FileOnlyLogger(logf)
        sys.stderr = FileOnlyLogger(logf)

        try:
            print("=" * 80)
            print("OPENAI OCR RUN START - OVARIAN REPORTS")
            print(f"Timestamp: {datetime.now().isoformat()}")
            print(f"Input folder: {input_folder}")
            print(f"Output folder: {output_folder}")
            print("=" * 80)

            if not os.path.exists(input_folder):
                print(f"[ERROR] Missing input folder: {input_folder}")
            else:
                patient_folders = [
                    os.path.join(input_folder, name)
                    for name in os.listdir(input_folder)
                    if os.path.isdir(os.path.join(input_folder, name))
                ]

                print(f"Found {len(patient_folders)} patient folders")

                processed = 0
                skipped = 0
                failed = 0

                for patient_folder in sorted(patient_folders):
                    patient_id = os.path.basename(patient_folder)

                    pdf_files = [
                        f for f in os.listdir(patient_folder)
                        if f.lower().endswith(".pdf")
                    ]

                    if not pdf_files:
                        print(f"\n[WARN] No PDF found for {patient_id}")
                        failed += 1
                        continue

                    pdf_path = os.path.join(patient_folder, pdf_files[0])

                    out_txt = os.path.join(output_folder, f"{patient_id}_openai.txt")
                    out_json = os.path.join(output_folder, "json", f"{patient_id}_openai_ocr.json")

                    if os.path.exists(out_txt):
                        print(f"\n[SKIP] {patient_id} already processed (TXT exists)")
                        skipped += 1
                        continue

                    print(f"\n[START] {patient_id}")
                    print(f"  PDF:  {pdf_path}")
                    print(f"  TXT:  {out_txt}")
                    print(f"  JSON: {out_json}")

                    try:
                        result = ocr_pdf_api(
                            pdf_path=pdf_path,
                            out_txt_path=out_txt,
                            out_json_path=out_json,
                            dpi=300,
                            attempts_per_page=3,
                            use_cot=True,
                            max_pages=None,
                        )
                        print("[DONE]")
                        print(result["summary"])
                        processed += 1

                    except Exception as e:
                        print("[ERROR]")
                        print(f"Patient: {patient_id}")
                        print(f"Exception: {e}")
                        print(traceback.format_exc())
                        failed += 1

                print("\n" + "=" * 80)
                print("OVARIAN REPORTS SUMMARY")
                print(f"Processed: {processed}")
                print(f"Skipped:   {skipped}")
                print(f"Failed:    {failed}")
                print("=" * 80)

            print("\n" + "=" * 80)
            print("OPENAI OCR RUN END - OVARIAN REPORTS")
            print(f"Finished: {datetime.now().isoformat()}")
            print("=" * 80)

        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    print(f"Log written to: {log_path}")