#!/usr/bin/env python3
"""
Generate Full Translation Dataset

Translates ALL turns (user and assistant) to the target language,
creating a fully monolingual non-English dataset.

This serves as a baseline to test pure multilingual capability
without any code-switching.

Usage:
    python generate_full_translation.py --lang de
    python generate_full_translation.py --lang zh
    python generate_full_translation.py --lang es
    python generate_full_translation.py --lang ar
"""

import os
import json
import argparse
from deep_translator import GoogleTranslator
from tqdm import tqdm

# Language codes for Google Translate
LANGUAGES = {
    "de": "german",
    "zh": "chinese (simplified)",
    "es": "spanish",
    "ar": "arabic",
}


def load_dataset(data_path, axes=None):
    """Load MultiChallenge dataset with optional filtering."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]

    if axes:
        data = [d for d in data if d['AXIS'] in axes]

    for item in data:
        item['turn_count'] = len(item['CONVERSATION'])

    return data


def translate_text(text, target_lang):
    """Translate text to target language using Google Translate."""
    try:
        translator = GoogleTranslator(source='en', target=LANGUAGES[target_lang])
        return translator.translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return None


def create_full_translation_item(item, target_lang):
    """
    Create a fully translated version of a conversation item.
    Translates ALL turns to the target language.
    """
    conversation = item['CONVERSATION']
    new_conversation = []
    translated_turns = []

    for i, msg in enumerate(conversation):
        original_text = msg['content']
        translated_text = translate_text(original_text, target_lang)

        if translated_text is None:
            return None

        new_conversation.append({
            'role': msg['role'],
            'content': translated_text
        })
        translated_turns.append({
            'turn_idx': i,
            'role': msg['role'],
            'original': original_text,
            'translated': translated_text,
        })

    new_item = {
        'QUESTION_ID': f"{item['QUESTION_ID']}_{target_lang}_full",
        'ORIGINAL_QUESTION_ID': item['QUESTION_ID'],
        'AXIS': item['AXIS'],
        'CONVERSATION': new_conversation,
        'TARGET_QUESTION': item['TARGET_QUESTION'],
        'PASS_CRITERIA': item['PASS_CRITERIA'],
        'TRANSLATION': {
            'type': 'full',
            'target_language': target_lang,
            'translated_turns': translated_turns,
        },
        'turn_count': len(new_conversation),
    }

    return new_item


def main():
    parser = argparse.ArgumentParser(description="Generate full translation dataset")

    parser.add_argument("--lang", type=str, required=True,
                        choices=list(LANGUAGES.keys()),
                        help="Target language for translation")

    parser.add_argument("--axis", type=str, default=None, nargs="+",
                        choices=["INFERENCE_MEMORY", "INSTRUCTION_RETENTION",
                                "SELF_COHERENCE", "RELIABLE_VERSION_EDITING"],
                        help="Filter by axis (default: INFERENCE_MEMORY + INSTRUCTION_RETENTION)")

    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: data/codeswitching/)")

    parser.add_argument("--samples", type=int, default=None,
                        help="Limit number of samples")

    args = parser.parse_args()

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data/multi-challenge/data/benchmark_questions.jsonl")
    output_dir = args.output or os.path.join(base_dir, "data/codeswitching")

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    axes = args.axis or ["INFERENCE_MEMORY", "INSTRUCTION_RETENTION"]
    data = load_dataset(data_path, axes=axes)

    print("=" * 60)
    print("GENERATE FULL TRANSLATION DATASET")
    print("=" * 60)
    print(f"Source: {data_path}")
    print(f"Target language: {args.lang} ({LANGUAGES[args.lang]})")
    print(f"Axes: {axes}")
    print(f"Questions found: {len(data)}")

    if args.samples and len(data) > args.samples:
        data = data[:args.samples]
        print(f"Using first {args.samples} samples")

    output_file = os.path.join(output_dir, f"full_translation_{args.lang}.jsonl")

    print(f"\nOutput: {output_file}")
    print("=" * 60)

    success = 0
    errors = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc=f"Translating all turns to {args.lang}"):
            new_item = create_full_translation_item(item, args.lang)

            if new_item:
                f.write(json.dumps(new_item, ensure_ascii=False) + "\n")
                success += 1
            else:
                errors += 1

    print("\n" + "=" * 60)
    print(f"COMPLETE: {success} questions generated")
    print(f"Errors: {errors}")
    print(f"Output: {output_file}")
    print("=" * 60)

    if success > 0:
        print("\nExample (first item):")
        with open(output_file, 'r', encoding='utf-8') as f:
            example = json.loads(f.readline())
            t = example['TRANSLATION']['translated_turns'][0]
            print(f"  First turn original: {t['original'][:80]}...")
            print(f"  First turn translated: {t['translated'][:80]}...")


if __name__ == "__main__":
    main()
