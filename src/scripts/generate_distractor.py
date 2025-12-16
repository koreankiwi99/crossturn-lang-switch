#!/usr/bin/env python3
"""
Generate Distractor Dataset

Creates datasets where Chinese distractor sentences are INSERTED IN THE MIDDLE
of the first user turn (constraint-setting turn), and the final user query is
translated to Chinese.

Pattern:
  Turn 1 (user): "Hello! I am an expert. [今天天气真好。最近很忙。] I prefer venues within 5-min walk..."
  Turn 2 (assistant, EN): Acknowledges constraint
  Turn 3 (user, ZH): Translated query - "推荐一家餐厅"

This tests if irrelevant foreign language text EMBEDDED WITHIN the context
disrupts the model's ability to extract and remember task-relevant constraints.

Usage:
    python generate_distractor.py --lang zh
    python generate_distractor.py --lang zh --num-distractors 2
"""

import os
import json
import argparse
import random
from deep_translator import GoogleTranslator
from tqdm import tqdm

# Import distractor sentences from separate file
from distractors import DISTRACTORS

# Language configuration
LANGUAGES = {
    "zh": "chinese (simplified)",
    "de": "german",
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


def get_random_distractors(lang, n=1):
    """Get n random distractor sentences for the specified language."""
    lang_distractors = DISTRACTORS.get(lang, DISTRACTORS["zh"])
    return random.sample(lang_distractors, min(n, len(lang_distractors)))


def insert_distractor_in_middle(text, distractor_text):
    """
    Insert distractor text in the middle of the original text.
    Splits at first sentence boundary after ~30% of the text.
    """
    # Find sentence boundaries (., !, ?)
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)

    if len(sentences) <= 1:
        # If only one sentence, insert after first 30% of characters
        split_point = len(text) // 3
        # Find next space after split point
        space_idx = text.find(' ', split_point)
        if space_idx == -1:
            space_idx = split_point
        return f"{text[:space_idx]} {distractor_text} {text[space_idx:].lstrip()}"

    # Insert after first sentence (or first ~30% of sentences)
    insert_idx = max(1, len(sentences) // 3)
    before = ' '.join(sentences[:insert_idx])
    after = ' '.join(sentences[insert_idx:])

    return f"{before} {distractor_text} {after}"


def create_distractor_item(item, target_lang, num_distractors=2, mix_all_turns=False):
    """
    Create a distractor version of a conversation item.

    - For 3-turn conversations: Insert distractors in first user turn only
    - For n>3 turns (if mix_all_turns=True): Insert distractors in ALL user turns except last
    - Last user turn: Translate to target language
    """
    conversation = item['CONVERSATION'].copy()
    turn_count = len(conversation)

    # Find all user turn indices
    user_turn_indices = [i for i, msg in enumerate(conversation) if msg['role'] == 'user']

    if len(user_turn_indices) < 2:
        return None

    last_user_idx = user_turn_indices[-1]
    # Turns to add distractors: all user turns except last (if mix_all_turns and n>3)
    if mix_all_turns and turn_count > 3:
        distractor_turn_indices = user_turn_indices[:-1]  # All user turns except last
    else:
        distractor_turn_indices = [user_turn_indices[0]]  # Only first user turn

    # Get original last query
    original_last_query = conversation[last_user_idx]['content']

    # Translate the last query to target language
    translated_query = translate_text(original_last_query, target_lang)
    if translated_query is None:
        return None

    # Track all modifications
    modified_turns = []

    # Build new conversation
    new_conversation = []
    for i, msg in enumerate(conversation):
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
            # Last user turn: translate to target language
            new_conversation.append({
                'role': msg['role'],
                'content': translated_query
            })
        else:
            # Keep other turns as-is
            new_conversation.append(msg.copy())

    new_item = {
        'QUESTION_ID': f"{item['QUESTION_ID']}_{target_lang}_distractor",
        'ORIGINAL_QUESTION_ID': item['QUESTION_ID'],
        'AXIS': item['AXIS'],
        'CONVERSATION': new_conversation,
        'TARGET_QUESTION': item['TARGET_QUESTION'],
        'PASS_CRITERIA': item['PASS_CRITERIA'],
        'DISTRACTOR': {
            'type': 'distractor_embedded_multi' if len(distractor_turn_indices) > 1 else 'distractor_embedded',
            'target_language': target_lang,
            'distractor_turn_indices': distractor_turn_indices,
            'last_turn_idx': last_user_idx,
            'modified_turns': modified_turns,
            'original_last_query': original_last_query,
            'translated_last_query': translated_query,
        },
        'turn_count': len(new_conversation),
    }

    return new_item


def main():
    parser = argparse.ArgumentParser(description="Generate distractor dataset")

    parser.add_argument("--lang", type=str, default="zh",
                        choices=list(LANGUAGES.keys()),
                        help="Target language for distractor and final query (default: zh)")

    parser.add_argument("--num-distractors", type=int, default=2,
                        help="Number of distractor sentences to prepend (default: 2)")

    parser.add_argument("--axis", type=str, default=None, nargs="+",
                        choices=["INFERENCE_MEMORY", "INSTRUCTION_RETENTION",
                                "SELF_COHERENCE", "RELIABLE_VERSION_EDITING"],
                        help="Filter by axis (default: INFERENCE_MEMORY + INSTRUCTION_RETENTION)")

    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: data/codeswitching/distractor/)")

    parser.add_argument("--samples", type=int, default=None,
                        help="Limit number of samples")

    parser.add_argument("--mix-all-turns", action="store_true",
                        help="For n>3 turn conversations, add distractors to ALL user turns except last")

    args = parser.parse_args()

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data/multi-challenge/data/benchmark_questions.jsonl")
    output_dir = args.output or os.path.join(base_dir, "data/codeswitching/distractor")

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    axes = args.axis or ["INFERENCE_MEMORY", "INSTRUCTION_RETENTION"]
    data = load_dataset(data_path, axes=axes)

    print("=" * 60)
    print("GENERATE DISTRACTOR DATASET")
    print("=" * 60)
    print(f"Source: {data_path}")
    if args.mix_all_turns:
        print(f"Pattern: EN[+{args.lang.upper()} noise in ALL user turns] → [{args.lang.upper()} query]")
    else:
        print(f"Pattern: EN[+{args.lang.upper()} noise in first turn] → [{args.lang.upper()} query]")
    print(f"Number of distractors per turn: {args.num_distractors}")
    print(f"Mix all turns (n>3): {args.mix_all_turns}")
    print(f"Axes: {axes}")
    print(f"Questions found: {len(data)}")

    if args.samples and len(data) > args.samples:
        data = data[:args.samples]
        print(f"Using first {args.samples} samples")

    suffix = "_multi" if args.mix_all_turns else ""
    output_file = os.path.join(output_dir, f"distractor_{args.lang}{suffix}.jsonl")

    print(f"\nOutput: {output_file}")
    print("=" * 60)

    success = 0
    errors = 0

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc=f"Adding {args.lang} distractors"):
            new_item = create_distractor_item(item, args.lang, args.num_distractors, args.mix_all_turns)

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
            d = example['DISTRACTOR']
            print(f"  Modified turns: {len(d['modified_turns'])}")
            if d['modified_turns']:
                first_mod = d['modified_turns'][0]
                print(f"  First modified turn: {first_mod['modified'][:100]}...")
            print(f"  Original last query: {d['original_last_query'][:60]}...")
            print(f"  Translated last query: {d['translated_last_query'][:60]}...")


if __name__ == "__main__":
    main()
