#!/usr/bin/env python3
"""
Generate Code-Switching Dataset

Extends MultiChallenge by translating the LAST user turn to create cross-turn
language switching scenarios.

For 3-turn conversations (INFERENCE_MEMORY):
  Turn 1 (user, EN): Sets constraint - "I prefer venues within 5-minute walk"
  Turn 2 (assistant, EN): Acknowledges constraint
  Turn 3 (user, DE/ZH/ES/AR): Tests memory - "Recommend a restaurant" [TRANSLATED]

Evaluation:
  Layer 1 - Language Fidelity: Did model respond in the target language?
  Layer 2 - Task Accuracy: Did model still remember the constraint from Turn 1?

Usage:
    python generate_codeswitching.py --turns 3 --lang de
    python generate_codeswitching.py --turns 3 --lang zh
    python generate_codeswitching.py --turns 3 --lang es
    python generate_codeswitching.py --turns 3 --lang ar
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
    """Translate text to target language using Google Translate."""
    try:
        translator = GoogleTranslator(source='en', target=LANGUAGES[target_lang])
        return translator.translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return None


def create_codeswitching_item(item, target_lang):
    """
    Create a code-switching version of a conversation item.
    Translates the LAST user turn to the target language.
    """
    conversation = item['CONVERSATION'].copy()

    # Find the last user turn
    last_user_idx = None
    for i in range(len(conversation) - 1, -1, -1):
        if conversation[i]['role'] == 'user':
            last_user_idx = i
            break

    if last_user_idx is None:
        return None

    # Translate the last user turn
    original_text = conversation[last_user_idx]['content']
    translated_text = translate_text(original_text, target_lang)

    if translated_text is None:
        return None

    # Create new conversation with translated last turn
    new_conversation = []
    for i, msg in enumerate(conversation):
        if i == last_user_idx:
            new_conversation.append({
                'role': 'user',
                'content': translated_text
            })
        else:
            new_conversation.append(msg.copy())

    # Create new item
    new_item = {
        'QUESTION_ID': f"{item['QUESTION_ID']}_{target_lang}",
        'ORIGINAL_QUESTION_ID': item['QUESTION_ID'],
        'AXIS': item['AXIS'],
        'CONVERSATION': new_conversation,
        'TARGET_QUESTION': item['TARGET_QUESTION'],
        'PASS_CRITERIA': item['PASS_CRITERIA'],
        'CODE_SWITCH': {
            'target_language': target_lang,
            'switched_turn': last_user_idx,
            'original_text': original_text,
            'translated_text': translated_text,
        },
        'turn_count': len(new_conversation),
    }

    return new_item


def main():
    parser = argparse.ArgumentParser(description="Generate code-switching dataset")

    parser.add_argument("--turns", type=int, default=None,
                        help="Filter by turn count (default: all)")

    parser.add_argument("--lang", type=str, required=True,
                        choices=list(LANGUAGES.keys()),
                        help="Target language for code-switching")

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
    print("GENERATE CODE-SWITCHING DATASET")
    print("=" * 60)
    print(f"Source: {data_path}")
    print(f"Turn count: {args.turns}")
    print(f"Target language: {args.lang} ({LANGUAGES[args.lang]})")
    print(f"Axes: {axes}")
    print(f"Questions found: {len(data)}")

    if args.samples and len(data) > args.samples:
        data = data[:args.samples]
        print(f"Using first {args.samples} samples")

    # Generate code-switching dataset
    turn_suffix = f"_{args.turns}turn" if args.turns else ""
    output_file = os.path.join(output_dir, f"codeswitching{turn_suffix}_{args.lang}.jsonl")

    print(f"\nOutput: {output_file}")
    print("=" * 60)

    success = 0
    errors = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc=f"Translating to {args.lang}"):
            new_item = create_codeswitching_item(item, args.lang)

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
            print(f"  Original last turn: {example['CODE_SWITCH']['original_text'][:100]}...")
            print(f"  Translated ({args.lang}): {example['CODE_SWITCH']['translated_text'][:100]}...")


if __name__ == "__main__":
    main()
