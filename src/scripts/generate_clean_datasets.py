#!/usr/bin/env python3
"""
Generate Clean Code-Switching Datasets from Verified Translations.

Tactics:
---------
| Condition      | Context      | Last Turn    | Source                              |
|----------------|--------------|--------------|-------------------------------------|
| Baseline (EN)  | Original EN  | Original EN  | multi-challenge (unchanged)         |
| Baseline (X)   | Verified X   | Verified X   | GPT-4o corrected translations       |
| EN→X           | Original EN  | Verified X   | EN original + last turn from X      |
| X→EN           | Verified X   | Original EN  | X verified + last turn from EN      |

Usage:
    python generate_clean_datasets.py
    python generate_clean_datasets.py --lang de
"""

import json
import argparse
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
ORIGINAL_PATH = DATA_DIR / "multi-challenge" / "data" / "benchmark_questions.jsonl"
VERIFIED_DIR = DATA_DIR / "codeswitching_verified" / "full_translation"
OUTPUT_DIR = DATA_DIR / "codeswitching_clean"

LANGUAGES = ["de", "zh", "es", "ar"]
LANG_NAMES = {"de": "German", "zh": "Chinese", "es": "Spanish", "ar": "Arabic"}


def load_original_dataset():
    """Load original English MultiChallenge dataset."""
    with open(ORIGINAL_PATH, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    return {item['QUESTION_ID']: item for item in data}


def load_verified_translations(lang):
    """Load verified/corrected translations for language."""
    path = VERIFIED_DIR / f"verified_full_translation_{lang}.jsonl"
    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    return {item.get('ORIGINAL_QUESTION_ID'): item for item in data}


def count_turns(filepath):
    """Count items and total turns in a dataset."""
    items = 0
    turns = 0
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                items += 1
                turns += len(item.get('CONVERSATION', []))
    return items, turns


def generate_baseline_en(original_data, valid_ids, output_dir):
    """Baseline (EN): Original English - unchanged."""
    output_path = output_dir / "baseline_en" / "baseline_en.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for qid in valid_ids:
            item = original_data[qid]
            output_item = {
                "QUESTION_ID": item['QUESTION_ID'],
                "AXIS": item['AXIS'],
                "CONVERSATION": item['CONVERSATION'],
                "TARGET_QUESTION": item['TARGET_QUESTION'],
                "PASS_CRITERIA": item['PASS_CRITERIA'],
                "CONDITION": "baseline_en",
                "PATTERN": "EN → EN → EN"
            }
            f.write(json.dumps(output_item, ensure_ascii=False) + '\n')

    return output_path


def generate_baseline_x(verified_data, lang, output_dir):
    """Baseline (X): Fully translated (verified/corrected)."""
    output_path = output_dir / "baseline_x" / f"baseline_{lang}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for orig_id, item in verified_data.items():
            output_item = {
                "QUESTION_ID": f"{orig_id}_{lang}_full",
                "ORIGINAL_QUESTION_ID": orig_id,
                "AXIS": item['AXIS'],
                "CONVERSATION": item['CONVERSATION'],
                "TARGET_QUESTION": item['TARGET_QUESTION'],
                "PASS_CRITERIA": item['PASS_CRITERIA'],
                "CONDITION": f"baseline_{lang}",
                "PATTERN": f"{lang.upper()} → {lang.upper()} → {lang.upper()}",
                "TARGET_LANGUAGE": lang
            }
            f.write(json.dumps(output_item, ensure_ascii=False) + '\n')

    return output_path


def generate_en_to_x(original_data, verified_data, lang, output_dir):
    """EN→X: Original EN context + Verified X last turn."""
    output_path = output_dir / "en_to_x" / f"en_to_{lang}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for orig_id, verified_item in verified_data.items():
            if orig_id not in original_data:
                continue

            orig_conv = original_data[orig_id]['CONVERSATION']
            verified_conv = verified_item['CONVERSATION']

            if len(orig_conv) != len(verified_conv):
                continue

            # EN context (all except last) + X last turn (verified)
            mixed_conv = orig_conv[:-1] + [verified_conv[-1]]

            output_item = {
                "QUESTION_ID": f"{orig_id}_en_to_{lang}",
                "ORIGINAL_QUESTION_ID": orig_id,
                "AXIS": original_data[orig_id]['AXIS'],
                "CONVERSATION": mixed_conv,
                "TARGET_QUESTION": original_data[orig_id]['TARGET_QUESTION'],
                "PASS_CRITERIA": original_data[orig_id]['PASS_CRITERIA'],
                "CONDITION": f"en_to_{lang}",
                "PATTERN": f"EN → ... → {lang.upper()}",
                "TARGET_LANGUAGE": lang,
                "SWITCH_TURN": len(mixed_conv) - 1
            }
            f.write(json.dumps(output_item, ensure_ascii=False) + '\n')

    return output_path


def generate_x_to_en(original_data, verified_data, lang, output_dir):
    """X→EN: Verified X context + Original EN last turn."""
    output_path = output_dir / "x_to_en" / f"{lang}_to_en.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for orig_id, verified_item in verified_data.items():
            if orig_id not in original_data:
                continue

            orig_conv = original_data[orig_id]['CONVERSATION']
            verified_conv = verified_item['CONVERSATION']

            if len(orig_conv) != len(verified_conv):
                continue

            # X context (verified, all except last) + EN last turn (original)
            mixed_conv = verified_conv[:-1] + [orig_conv[-1]]

            output_item = {
                "QUESTION_ID": f"{orig_id}_{lang}_to_en",
                "ORIGINAL_QUESTION_ID": orig_id,
                "AXIS": original_data[orig_id]['AXIS'],
                "CONVERSATION": mixed_conv,
                "TARGET_QUESTION": original_data[orig_id]['TARGET_QUESTION'],
                "PASS_CRITERIA": original_data[orig_id]['PASS_CRITERIA'],
                "CONDITION": f"{lang}_to_en",
                "PATTERN": f"{lang.upper()} → ... → EN",
                "TARGET_LANGUAGE": "en",
                "SOURCE_LANGUAGE": lang,
                "SWITCH_TURN": len(mixed_conv) - 1
            }
            f.write(json.dumps(output_item, ensure_ascii=False) + '\n')

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, choices=LANGUAGES)
    args = parser.parse_args()

    languages = [args.lang] if args.lang else LANGUAGES

    print("=" * 60)
    print("GENERATING CLEAN CODE-SWITCHING DATASETS")
    print("=" * 60)

    # Load original
    print("\nLoading original English dataset...")
    original_data = load_original_dataset()
    print(f"Loaded {len(original_data)} items")

    # Get valid IDs (those with verified translations)
    first_verified = load_verified_translations(LANGUAGES[0])
    valid_ids = list(first_verified.keys())
    print(f"Valid IDs (with translations): {len(valid_ids)}")

    # Baseline EN
    if not args.lang:
        print("\n--- Baseline (EN) ---")
        path = generate_baseline_en(original_data, valid_ids, OUTPUT_DIR)
        items, turns = count_turns(path)
        print(f"  {path.name}: {items} items, {turns} turns")

    # Per language
    for lang in languages:
        print(f"\n--- {LANG_NAMES[lang]} ({lang.upper()}) ---")
        verified_data = load_verified_translations(lang)

        path = generate_baseline_x(verified_data, lang, OUTPUT_DIR)
        items, turns = count_turns(path)
        print(f"  Baseline ({lang.upper()}): {items} items, {turns} turns")

        path = generate_en_to_x(original_data, verified_data, lang, OUTPUT_DIR)
        items, turns = count_turns(path)
        print(f"  EN→{lang.upper()}: {items} items, {turns} turns")

        path = generate_x_to_en(original_data, verified_data, lang, OUTPUT_DIR)
        items, turns = count_turns(path)
        print(f"  {lang.upper()}→EN: {items} items, {turns} turns")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Dataset':<30} {'Items':>8} {'Turns':>8}")
    print("-" * 50)

    for condition in ["baseline_en", "baseline_x", "en_to_x", "x_to_en"]:
        cdir = OUTPUT_DIR / condition
        if cdir.exists():
            for fpath in sorted(cdir.glob("*.jsonl")):
                items, turns = count_turns(fpath)
                print(f"{fpath.name:<30} {items:>8} {turns:>8}")


if __name__ == "__main__":
    main()
