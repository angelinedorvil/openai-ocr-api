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
    char_blocks = []

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

        # char_blocks.append({
        #     "operation": tag,
        #     "text1_chars": part1,
        #     "text2_chars": part2,
        #     "count_text1": len(part1),
        #     "count_text2": len(part2),
        # })

    return {
        "different_chars_in_text1": diff_count_1,
        "different_chars_in_text2": diff_count_2,
        "total_different_chars": diff_count_1 + diff_count_2,
        # "char_diff_blocks": char_blocks
    }


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

        # diff_blocks.append({
        #     "operation": tag,
        #     "text1_words": block1,
        #     "text2_words": block2,
        #     "count_words_text1": len(block1),
        #     "count_words_text2": len(block2),
        #     "char_comparison": char_result
        # })

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



path = 'ocr/stats.csv' # list of file names
stats = pd.read_csv(path, header='infer')
output_file = 'results.csv' # output file for comparison results
if os.path.exists(output_file):
    results = open(output_file, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(results, fieldnames=['file_name', 'total_words_qwen', 'total_words_claude', 'different_words_in_qwen', 'different_words_in_claude', 'different_chars_in_qwen', 'different_chars_in_claude', 'total_different_chars', 'total_different_words'])
    writer.writeheader()
else:
    results = open(output_file, 'w', newline='', encoding='utf-8')
    writer = csv.DictWriter(results, fieldnames=['file_name', 'total_words_qwen', 'total_words_claude', 'different_words_in_qwen', 'different_words_in_claude', 'different_chars_in_qwen', 'different_chars_in_claude', 'total_different_chars', 'total_different_words'])

for idx, row in stats.iterrows():
    file_qwen = glob.glob('ocr/' + row['file'] + '*') #root folder of outputs from qwen-ocr
    file_claude = glob.glob('claude/' + row['file'] + '*') #root folder of outputs from claude <==== update with your model

    with open(file_qwen[0], 'r', encoding='utf-8') as f:
        text1 = f.read()
    with open(file_claude[0], 'r', encoding='utf-8') as f:
        text2 = f.read()

    result = compare_texts(text1, text2)
    writer.writerow({
        'file_name': row['file'],
        'total_words_qwen': result['total_words_text1'],
        'total_words_claude': result['total_words_text2'],
        'different_words_in_qwen': result['different_words_in_text1'],
        'different_words_in_claude': result['different_words_in_text2'],
        'different_chars_in_qwen': result['different_chars_in_text1'],
        'different_chars_in_claude': result['different_chars_in_text2'],
        'total_different_chars': result['total_different_chars'],
        'total_different_words': result['total_different_words']
    })
