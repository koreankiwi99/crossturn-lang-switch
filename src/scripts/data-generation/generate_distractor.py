#!/usr/bin/env python3
"""
Generate Distractor Dataset (from Verified Translations)

Creates datasets where foreign language distractor sentences are INSERTED IN THE MIDDLE
of user turns (noise), and the final user query uses VERIFIED translations.

Pattern:
  Turn 1 (user): "Hello! I am an expert. [今天天气真好。最近很忙。] I prefer venues within 5-min walk..."
  Turn 2 (assistant, EN): Acknowledges constraint
  Turn 3 (user, ZH): Verified translated query from GPT-4o-mini

This tests if irrelevant foreign language text EMBEDDED WITHIN the context
disrupts the model's ability to extract and remember task-relevant constraints.

Usage:
    python generate_distractor.py --lang zh
    python generate_distractor.py --lang zh --num-distractors 2
    python generate_distractor.py --lang zh --mix-all-turns
"""

import os
import json
import argparse
import random
from pathlib import Path
from tqdm import tqdm

# Import distractor sentences from separate file
from distractors import DISTRACTORS

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
ORIGINAL_PATH = DATA_DIR / "multi-challenge" / "data" / "benchmark_questions.jsonl"
VERIFIED_DIR = DATA_DIR / "translations"
OUTPUT_DIR = DATA_DIR / "experiments" / "distractor"

# Language configuration
LANGUAGES = {
    "zh": "Chinese",
    "de": "German",
    "es": "Spanish",
    "ar": "Arabic",
}


def load_original_dataset(axes=None):
    """Load original English MultiChallenge dataset."""
    with open(ORIGINAL_PATH, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]

    if axes:
        data = [d for d in data if d['AXIS'] in axes]

    return {item['QUESTION_ID']: item for item in data}


def load_verified_translations(lang):
    """Load verified translations for language."""
    path = VERIFIED_DIR / f"verified_full_translation_{lang}.jsonl"
    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    return {item.get('ORIGINAL_QUESTION_ID'): item for item in data}


def get_random_distractors(lang, n=1):
    """Get n random distractor sentences for the specified language."""
    lang_distractors = DISTRACTORS.get(lang, DISTRACTORS["zh"])
    return random.sample(lang_distractors, min(n, len(lang_distractors)))


def insert_distractor_in_middle(text, distractor_text):
    """
    Insert distractor text in the middle of the original text.
    Splits at first sentence boundary after ~30% of the text.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)

    if len(sentences) <= 1:
        split_point = len(text) // 3
        space_idx = text.find(' ', split_point)
        if space_idx == -1:
            space_idx = split_point
        return f"{text[:space_idx]} {distractor_text} {text[space_idx:].lstrip()}"

    insert_idx = max(1, len(sentences) // 3)
    before = ' '.join(sentences[:insert_idx])
    after = ' '.join(sentences[insert_idx:])

    return f"{before} {distractor_text} {after}"


def create_distractor_item(original_item, verified_item, target_lang, num_distractors=2, mix_all_turns=False):
    """
    Create a distractor version of a conversation item.

    - Insert distractors in user turns (first only, or all except last if mix_all_turns)
    - Last user turn: Use VERIFIED translation (not Google Translate)
    """
    orig_conv = original_item['CONVERSATION']
    verified_conv = verified_item['CONVERSATION']
    turn_count = len(orig_conv)

    if len(orig_conv) != len(verified_conv):
        return None

    # Find all user turn indices
    user_turn_indices = [i for i, msg in enumerate(orig_conv) if msg['role'] == 'user']

    if len(user_turn_indices) < 2:
        return None

    last_user_idx = user_turn_indices[-1]

    # Turns to add distractors: first only, or all user turns except last (if mix_all_turns and n>3)
    if mix_all_turns and turn_count > 3:
        distractor_turn_indices = user_turn_indices[:-1]  # All user turns except last
    else:
        distractor_turn_indices = [user_turn_indices[0]]  # Only first user turn

    # Get verified last query
    original_last_query = orig_conv[last_user_idx]['content']
    verified_last_query = verified_conv[last_user_idx]['content']

    # Track all modifications
    modified_turns = []

    # Build new conversation
    new_conversation = []
    for i, msg in enumerate(orig_conv):
        if i in distractor_turn_indices:
            # Insert distractors in this user turn
            distractors = get_random_distractors(target_lang, num_distractors)
            distractor_text = " ".join(distractors)
            original_content = msg['content']
            modified_content = insert_distractor_in_middle(original_content, distractor_text)
            new_conversation.append({
                'role': msg['role'],
                'content': modified_content
            })
            modified_turns.append({
                'turn_idx': i,
                'distractors': distractors,
                'original': original_content,
                'modified': modified_content,
            })
        elif i == last_user_idx:
            # Last user turn: use VERIFIED translation
            new_conversation.append({
                'role': msg['role'],
                'content': verified_last_query
            })
        else:
            # Keep other turns as-is (original English)
            new_conversation.append(msg.copy())

    new_item = {
        'QUESTION_ID': f"{original_item['QUESTION_ID']}_{target_lang}_distractor",
        'ORIGINAL_QUESTION_ID': original_item['QUESTION_ID'],
        'AXIS': original_item['AXIS'],
        'CONVERSATION': new_conversation,
        'TARGET_QUESTION': original_item['TARGET_QUESTION'],
        'PASS_CRITERIA': original_item['PASS_CRITERIA'],
        'CONDITION': 'distractor_multi' if len(distractor_turn_indices) > 1 else 'distractor',
        'PATTERN': f"EN[+{target_lang.upper()} noise] → ... → {target_lang.upper()}",
        'DISTRACTOR': {
            'type': 'distractor_embedded_multi' if len(distractor_turn_indices) > 1 else 'distractor_embedded',
            'target_language': target_lang,
            'distractor_turn_indices': distractor_turn_indices,
            'last_turn_idx': last_user_idx,
            'modified_turns': modified_turns,
            'original_last_query': original_last_query,
            'verified_last_query': verified_last_query,
        },
        'turn_count': len(new_conversation),
    }

    return new_item


def main():
    parser = argparse.ArgumentParser(description="Generate distractor dataset from verified translations")

    parser.add_argument("--lang", type=str, default="zh",
                        choices=list(LANGUAGES.keys()),
                        help="Target language for distractor and final query (default: zh)")

    parser.add_argument("--num-distractors", type=int, default=2,
                        help="Number of distractor sentences per turn (default: 2)")

    parser.add_argument("--axis", type=str, default=None, nargs="+",
                        choices=["INFERENCE_MEMORY", "INSTRUCTION_RETENTION",
                                "SELF_COHERENCE", "RELIABLE_VERSION_EDITING"],
                        help="Filter by axis (default: INFERENCE_MEMORY + INSTRUCTION_RETENTION)")

    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")

    parser.add_argument("--samples", type=int, default=None,
                        help="Limit number of samples")

    parser.add_argument("--mix-all-turns", action="store_true",
                        help="For n>3 turn conversations, add distractors to ALL user turns except last")

    args = parser.parse_args()

    output_dir = Path(args.output) if args.output else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    axes = args.axis or ["INFERENCE_MEMORY", "INSTRUCTION_RETENTION"]
    original_data = load_original_dataset(axes=axes)
    verified_data = load_verified_translations(args.lang)

    print("=" * 60)
    print("GENERATE DISTRACTOR DATASET (from Verified Translations)")
    print("=" * 60)
    print(f"Source: {ORIGINAL_PATH}")
    print(f"Verified translations: {VERIFIED_DIR / f'verified_full_translation_{args.lang}.jsonl'}")
    if args.mix_all_turns:
        print(f"Pattern: EN[+{args.lang.upper()} noise in ALL user turns] → [{args.lang.upper()} verified query]")
    else:
        print(f"Pattern: EN[+{args.lang.upper()} noise in first turn] → [{args.lang.upper()} verified query]")
    print(f"Number of distractors per turn: {args.num_distractors}")
    print(f"Mix all turns (n>3): {args.mix_all_turns}")
    print(f"Axes: {axes}")
    print(f"Original items: {len(original_data)}")
    print(f"Verified items: {len(verified_data)}")

    # Filter to items with verified translations
    valid_ids = [qid for qid in original_data if qid in verified_data]
    if args.samples and len(valid_ids) > args.samples:
        valid_ids = valid_ids[:args.samples]
    print(f"Processing: {len(valid_ids)} items")

    suffix = "_multi" if args.mix_all_turns else ""
    output_file = output_dir / f"distractor_{args.lang}{suffix}.jsonl"

    print(f"\nOutput: {output_file}")
    print("=" * 60)

    success = 0
    errors = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        for qid in tqdm(valid_ids, desc=f"Adding {args.lang} distractors"):
            new_item = create_distractor_item(
                original_data[qid],
                verified_data[qid],
                args.lang,
                args.num_distractors,
                args.mix_all_turns
            )

            if new_item:
                f.write(json.dumps(new_item, ensure_ascii=False) + "\n")
                success += 1
            else:
                errors += 1

    print("\n" + "=" * 60)
    print(f"COMPLETE: {success} items generated")
    print(f"Skipped (turn count mismatch): {errors}")
    print(f"Output: {output_file}")
    print("=" * 60)

    if success > 0:
        print("\nExample (first item):")
        with open(output_file, 'r', encoding='utf-8') as f:
            example = json.loads(f.readline())
            d = example['DISTRACTOR']
            print(f"  Modified turns: {len(d['modified_turns'])}")
            if d['modified_turns']:
                first_mod = d['modified_turns'][0]
                print(f"  First modified turn: {first_mod['modified'][:100]}...")
            print(f"  Original last query: {d['original_last_query'][:60]}...")
            print(f"  Verified last query: {d['verified_last_query'][:60]}...")


if __name__ == "__main__":
    main()
