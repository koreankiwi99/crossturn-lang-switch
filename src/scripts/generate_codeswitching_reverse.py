#!/usr/bin/env python3
"""
Generate Reverse Code-Switching Dataset (X→EN)

Creates datasets where all turns EXCEPT the last user turn are translated
to a foreign language. The last user turn remains in English.

For a conversation:
  Turn 1 (user, DE): Sets constraint [TRANSLATED from EN]
  Turn 2 (assistant, DE): Acknowledges constraint [TRANSLATED from EN]
  Turn 3 (user, EN): Tests memory - "Recommend a restaurant" [ORIGINAL EN]

This tests if the model can maintain task accuracy when the context
is in a foreign language but the final query is in English.

Usage:
    python generate_codeswitching_reverse.py --lang de
    python generate_codeswitching_reverse.py --lang zh
    python generate_codeswitching_reverse.py --lang es
    python generate_codeswitching_reverse.py --lang ar
"""

import os
import sys
import json
import argparse
from datetime import datetime
from deep_translator import GoogleTranslator
from tqdm import tqdm

# Language codes for Google Translate
LANGUAGES = {
    "de": "german",
    "zh": "chinese (simplified)",
    "es": "spanish",
    "ar": "arabic",
}


def load_dataset(data_path, axes=None, turn_count=None):
    """Load MultiChallenge dataset with optional filtering."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]

    # Filter by axis
    if axes:
        data = [d for d in data if d['AXIS'] in axes]

    # Add turn count
    for item in data:
        item['turn_count'] = len(item['CONVERSATION'])

    # Filter by turn count
    if turn_count:
        data = [d for d in data if len(d['CONVERSATION']) == turn_count]

    return data


def translate_text(text, target_lang):
    """Translate text from English to target language using Google Translate."""
    try:
        translator = GoogleTranslator(source='en', target=LANGUAGES[target_lang])
        return translator.translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return None


def create_reverse_codeswitching_item(item, target_lang):
    """
    Create a reverse code-switching version of a conversation item.
    Translates ALL turns EXCEPT the last user turn to the target language.
    """
    conversation = item['CONVERSATION'].copy()

    # Find the last user turn index
    last_user_idx = None
    for i in range(len(conversation) - 1, -1, -1):
        if conversation[i]['role'] == 'user':
            last_user_idx = i
            break

    if last_user_idx is None:
        return None

    # Translate all turns except the last user turn
    new_conversation = []
    translated_turns = []

    for i, msg in enumerate(conversation):
        if i == last_user_idx:
            # Keep last user turn in English
            new_conversation.append(msg.copy())
        else:
            # Translate this turn
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

    # Create new item
    new_item = {
        'QUESTION_ID': f"{item['QUESTION_ID']}_{target_lang}_reverse",
        'ORIGINAL_QUESTION_ID': item['QUESTION_ID'],
        'AXIS': item['AXIS'],
        'CONVERSATION': new_conversation,
        'TARGET_QUESTION': item['TARGET_QUESTION'],
        'PASS_CRITERIA': item['PASS_CRITERIA'],
        'CODE_SWITCH': {
            'direction': f'{target_lang}->en',
            'context_language': target_lang,
            'query_language': 'en',
            'last_turn_idx': last_user_idx,
            'translated_turns': translated_turns,
        },
        'turn_count': len(new_conversation),
    }

    return new_item


def main():
    parser = argparse.ArgumentParser(description="Generate reverse code-switching dataset (X→EN)")

    parser.add_argument("--turns", type=int, default=None,
                        help="Filter by turn count (default: all)")

    parser.add_argument("--lang", type=str, required=True,
                        choices=list(LANGUAGES.keys()),
                        help="Language for context (last turn stays in English)")

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
    data = load_dataset(data_path, axes=axes, turn_count=args.turns)

    print("=" * 60)
    print("GENERATE REVERSE CODE-SWITCHING DATASET")
    print("=" * 60)
    print(f"Source: {data_path}")
    print(f"Turn count: {args.turns}")
    print(f"Context language: {args.lang} ({LANGUAGES[args.lang]})")
    print(f"Query language: English")
    print(f"Axes: {axes}")
    print(f"Questions found: {len(data)}")

    if args.samples and len(data) > args.samples:
        data = data[:args.samples]
        print(f"Using first {args.samples} samples")

    # Generate reverse code-switching dataset
    turn_suffix = f"_{args.turns}turn" if args.turns else ""
    output_file = os.path.join(output_dir, f"codeswitching{turn_suffix}_{args.lang}_to_en.jsonl")

    print(f"\nOutput: {output_file}")
    print("=" * 60)

    success = 0
    errors = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc=f"Translating context to {args.lang}"):
            new_item = create_reverse_codeswitching_item(item, args.lang)

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

    # Show example
    if success > 0:
        print("\nExample (first item):")
        with open(output_file, 'r', encoding='utf-8') as f:
            example = json.loads(f.readline())
            cs = example['CODE_SWITCH']
            print(f"  Direction: {cs['direction']}")
            print(f"  Translated turns: {len(cs['translated_turns'])}")
            if cs['translated_turns']:
                t = cs['translated_turns'][0]
                print(f"  First turn original: {t['original'][:80]}...")
                print(f"  First turn translated: {t['translated'][:80]}...")


if __name__ == "__main__":
    main()
