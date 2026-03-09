from difflib import SequenceMatcher
import re
import pandas as pd
import csv
import glob
import os


def tokenize_words(text):
    # words only; change regex if you want punctuation included
    return re.findall(r"\S+", text)


def char_diff_count(s1, s2):
    """
    Count character-level differences between two strings.
    Insertions, deletions, and replacements are all counted.
    """
    matcher = SequenceMatcher(None, s1, s2)
    diff_count_1 = 0
    diff_count_2 = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        part1 = s1[i1:i2]
        part2 = s2[j1:j2]

        if tag == "equal":
            continue

        if tag == "replace":
            diff_count_1 += len(part1)
            diff_count_2 += len(part2)
        elif tag == "delete":
            diff_count_1 += len(part1)
        elif tag == "insert":
            diff_count_2 += len(part2)

    return {
        "different_chars_in_text1": diff_count_1,
        "different_chars_in_text2": diff_count_2,
        "total_different_chars": diff_count_1 + diff_count_2,
    }


def strip_openai_header(text):
    """
    Remove the metadata header block at the top of OpenAI output files.
    Strips everything up to and including the blank line after the last
    header separator (================...=) before PAGE 1 content begins.
    """
    # Split on the PAGE 1 marker and keep everything from there onward
    marker = "PAGE 1"
    idx = text.find(marker)
    if idx != -1:
        # Walk back to include the separator line before PAGE 1
        start = text.rfind("=" * 10, 0, idx)
        if start != -1:
            return text[start:].strip()
    return text.strip()


def compare_texts(text1, text2):
    words1 = tokenize_words(text1)
    words2 = tokenize_words(text2)

    matcher = SequenceMatcher(None, words1, words2)

    diff_count_words_1 = 0
    diff_count_words_2 = 0
    total_char_diff_1 = 0
    total_char_diff_2 = 0
    diff_blocks = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        block1 = words1[i1:i2]
        block2 = words2[j1:j2]

        if tag == "equal":
            continue

        joined1 = " ".join(block1)
        joined2 = " ".join(block2)

        if tag == "replace":
            diff_count_words_1 += len(block1)
            diff_count_words_2 += len(block2)
        elif tag == "delete":
            diff_count_words_1 += len(block1)
        elif tag == "insert":
            diff_count_words_2 += len(block2)

        char_result = char_diff_count(joined1, joined2)
        total_char_diff_1 += char_result["different_chars_in_text1"]
        total_char_diff_2 += char_result["different_chars_in_text2"]

    return {
        "total_words_text1": len(words1),
        "total_words_text2": len(words2),
        "different_words_in_text1": diff_count_words_1,
        "different_words_in_text2": diff_count_words_2,
        "total_different_words": diff_count_words_1 + diff_count_words_2,
        "different_chars_in_text1": total_char_diff_1,
        "different_chars_in_text2": total_char_diff_2,
        "total_different_chars": total_char_diff_1 + total_char_diff_2,
        "diff_blocks": diff_blocks
    }


# ── Paths ──────────────────────────────────────────────────────────────────────
QWEN_DIR   = r'outputs\ovarian_cancer\ocr'     # qwen files: <ID>.<GUID>_ocr.txt
OPENAI_DIR = r'outputs\ovarian_cancer\openai'  # openai files: <ID>_openai.txt
OUTPUT_FILE = 'results-openai.csv'

FIELDNAMES = [
    'file_name',
    'total_words_qwen', 'total_words_openai',
    'different_words_in_qwen', 'different_words_in_openai',
    'different_chars_in_qwen', 'different_chars_in_openai',
    'total_different_chars', 'total_different_words',
]

# ── Build a lookup: short ID -> full qwen filepath ────────────────────────────
# Qwen filenames look like:  TCGA-A5-A0G1.A2EB7E85-..._ocr.txt
# The short ID is everything before the first dot: TCGA-A5-A0G1
qwen_lookup = {}
for fp in glob.glob(os.path.join(QWEN_DIR, '*.txt')):
    basename = os.path.basename(fp)               # TCGA-A5-A0G1.GUID_ocr.txt
    short_id = basename.split('.')[0]             # TCGA-A5-A0G1
    qwen_lookup[short_id] = fp

# ── Open output CSV ───────────────────────────────────────────────────────────
file_exists = os.path.exists(OUTPUT_FILE)
results = open(OUTPUT_FILE, 'a' if file_exists else 'w', newline='', encoding='utf-8')
writer = csv.DictWriter(results, fieldnames=FIELDNAMES)
if not file_exists:
    writer.writeheader()

# ── Main loop ─────────────────────────────────────────────────────────────────
skipped = []

for fp_openai in glob.glob(os.path.join(OPENAI_DIR, '*_openai.txt')):
    basename = os.path.basename(fp_openai)        # TCGA-A5-A0G1_openai.txt
    short_id = basename.replace('_openai.txt', '') # TCGA-A5-A0G1

    fp_qwen = qwen_lookup.get(short_id)
    if fp_qwen is None:
        print(f"[SKIP] No matching qwen file for: {short_id}")
        skipped.append(short_id)
        continue

    with open(fp_qwen, 'r', encoding='utf-8') as f:
        text_qwen = f.read()

    with open(fp_openai, 'r', encoding='utf-8') as f:
        raw_openai = f.read()

    text_openai = strip_openai_header(raw_openai)

    result = compare_texts(text_qwen, text_openai)

    writer.writerow({
        'file_name':                short_id,
        'total_words_qwen':         result['total_words_text1'],
        'total_words_openai':       result['total_words_text2'],
        'different_words_in_qwen':  result['different_words_in_text1'],
        'different_words_in_openai':result['different_words_in_text2'],
        'different_chars_in_qwen':  result['different_chars_in_text1'],
        'different_chars_in_openai':result['different_chars_in_text2'],
        'total_different_chars':    result['total_different_chars'],
        'total_different_words':    result['total_different_words'],
    })
    print(f"[OK] {short_id}")

results.close()

if skipped:
    print(f"\nSkipped {len(skipped)} files with no qwen match: {skipped}")
else:
    print("\nAll files matched successfully.")