#!/usr/bin/env python3
"""
Verify and Correct Translations using GPT-4o.

Compares original English MultiChallenge dataset with full translations,
identifies errors, and outputs verified/corrected datasets.

Usage:
    python verify_and_correct_translations.py --lang de
    python verify_and_correct_translations.py --lang zh --samples 20
    python verify_and_correct_translations.py --lang es --workers 8
"""

import json
import os
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

# Load API key from .env
env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith('#') and '=' in line:
                key, value = line.strip().split('=', 1)
                os.environ[key] = value.strip('"').strip("'")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
PROMPTS_DIR = BASE_DIR / "prompts"
DATA_DIR = BASE_DIR / "data"

# Language names
LANGUAGE_NAMES = {
    "de": "German",
    "zh": "Chinese (Simplified)",
    "es": "Spanish",
    "ar": "Arabic",
}


def load_prompt(filename):
    """Load prompt from file."""
    with open(PROMPTS_DIR / filename, 'r', encoding='utf-8') as f:
        return f.read().strip()


def format_conversation(conversation):
    """Format conversation list as readable text."""
    lines = []
    for i, turn in enumerate(conversation):
        role = turn['role'].upper()
        content = turn['content']
        lines.append(f"Turn {i+1} ({role}): {content}")
    return "\n\n".join(lines)


def load_original_dataset():
    """Load original English MultiChallenge dataset."""
    original_path = DATA_DIR / "multi-challenge" / "data" / "benchmark_questions.jsonl"
    with open(original_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    # Index by question ID
    return {item['QUESTION_ID']: item for item in data}


def verify_translation(original_item, translated_item, target_lang, system_prompt, user_prompt_template):
    """Verify a single translation using GPT-4o."""

    original_conv = format_conversation(original_item['CONVERSATION'])
    translated_conv = format_conversation(translated_item['CONVERSATION'])

    turn_count = len(original_item['CONVERSATION'])
    user_prompt = user_prompt_template.format(
        original_conversation=original_conv,
        target_language=LANGUAGE_NAMES.get(target_lang, target_lang),
        translated_conversation=translated_conv,
        turn_count=turn_count
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=16000,
            response_format={"type": "json_object"}
        )

        result_text = response.choices[0].message.content
        result = json.loads(result_text)

        return {
            'question_id': translated_item.get('QUESTION_ID'),
            'success': True,
            'verification': result,
            'usage': {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens
            }
        }

    except json.JSONDecodeError as e:
        return {
            'question_id': translated_item.get('QUESTION_ID'),
            'success': False,
            'error': f'JSON parse error: {str(e)}',
            'raw_response': result_text if 'result_text' in locals() else None
        }
    except Exception as e:
        return {
            'question_id': translated_item.get('QUESTION_ID'),
            'success': False,
            'error': str(e)
        }


def apply_corrections(item, verification_result, original_turn_count):
    """Apply corrections to create corrected item. Validates turn count."""
    corrected_item = item.copy()
    verification = verification_result.get('verification', {})

    # Validate turn count
    corrected_conv = verification.get('corrected_conversation', [])
    if corrected_conv and len(corrected_conv) != original_turn_count:
        # Turn count mismatch - reject correction, keep original
        corrected_item['VERIFICATION'] = {
            'verified': True,
            'had_issues': False,
            'num_issues': 0,
            'issues': [],
            'rejected_reason': f'Turn count mismatch: expected {original_turn_count}, got {len(corrected_conv)}'
        }
        return corrected_item

    # New format: accurate=false means issues found
    has_issues = not verification.get('accurate', True)
    issues = verification.get('issues', [])

    if has_issues and corrected_conv:
        if isinstance(corrected_conv, list):
            corrected_item['CONVERSATION'] = corrected_conv

        corrected_item['VERIFICATION'] = {
            'verified': True,
            'had_issues': True,
            'num_issues': len(issues),
            'issues': issues,
            'verified_at': datetime.now().isoformat()
        }
    else:
        # No issues or use corrected_conversation as-is
        if corrected_conv and isinstance(corrected_conv, list):
            corrected_item['CONVERSATION'] = corrected_conv

        corrected_item['VERIFICATION'] = {
            'verified': True,
            'had_issues': False,
            'num_issues': 0,
            'issues': [],
            'verified_at': datetime.now().isoformat()
        }

    return corrected_item


def main():
    parser = argparse.ArgumentParser(description="Verify and correct translations using GPT-4o")
    parser.add_argument("--lang", type=str, required=True, choices=list(LANGUAGE_NAMES.keys()),
                       help="Target language to verify")
    parser.add_argument("--samples", type=int, default=None,
                       help="Limit to N samples (for testing)")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from previous run, only re-verify failed items")

    args = parser.parse_args()

    # Load prompts
    system_prompt = load_prompt("verification_system_prompt.txt")
    user_prompt_template = load_prompt("verification_user_prompt.txt")

    # Load original English dataset
    print("Loading original English dataset...")
    original_data = load_original_dataset()
    print(f"Loaded {len(original_data)} original items")

    # Load translated dataset
    translated_path = DATA_DIR / "codeswitching" / "full_translation" / f"full_translation_{args.lang}.jsonl"
    if not translated_path.exists():
        print(f"Error: Translated file not found: {translated_path}")
        return

    with open(translated_path, 'r', encoding='utf-8') as f:
        translated_data = [json.loads(line) for line in f if line.strip()]

    if args.samples:
        translated_data = translated_data[:args.samples]

    print(f"Loaded {len(translated_data)} translated items")

    # Output paths
    output_dir = DATA_DIR / "codeswitching_verified" / "full_translation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"verified_full_translation_{args.lang}.jsonl"

    # Handle resume mode
    existing_verified = {}
    items_to_process = translated_data
    if args.resume and output_file.exists():
        print("Loading existing verified data for resume...")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    verification = item.get('VERIFICATION', {})
                    # Keep successfully verified items
                    if verification.get('verified', False):
                        existing_verified[item.get('QUESTION_ID')] = item

        # Filter to only failed/unverified items
        items_to_process = [
            item for item in translated_data
            if item.get('QUESTION_ID') not in existing_verified
        ]
        print(f"Found {len(existing_verified)} already verified, {len(items_to_process)} to re-verify")

    print("\n" + "=" * 60)
    print("TRANSLATION VERIFICATION WITH GPT-4o")
    print("=" * 60)
    print(f"Language: {LANGUAGE_NAMES[args.lang]}")
    print(f"Items:    {len(items_to_process)} (of {len(translated_data)} total)")
    print(f"Workers:  {args.workers}")
    print(f"Output:   {output_file}")
    print("=" * 60 + "\n")

    results = []
    corrected_items = list(existing_verified.values())  # Start with existing verified items

    # Process with thread pool
    def process_item(translated_item):
        # Get original question ID
        orig_id = translated_item.get('ORIGINAL_QUESTION_ID')
        if not orig_id:
            # Fallback: try to extract from QUESTION_ID
            qid = translated_item.get('QUESTION_ID', '')
            # Remove suffixes like _de_full, _de, etc.
            for suffix in [f'_{args.lang}_full', f'_{args.lang}']:
                if qid.endswith(suffix):
                    orig_id = qid[:-len(suffix)]
                    break
            else:
                orig_id = qid

        original_item = original_data.get(orig_id)
        if not original_item:
            return None, translated_item, 0

        result = verify_translation(
            original_item, translated_item, args.lang,
            system_prompt, user_prompt_template
        )
        original_turn_count = len(original_item['CONVERSATION'])
        return result, translated_item, original_turn_count

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_item, item) for item in items_to_process]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Verifying"):
            try:
                result, item, original_turn_count = future.result()

                if result is None:
                    item['VERIFICATION'] = {'verified': False, 'error': 'Original not found'}
                    corrected_items.append(item)
                elif result.get('success'):
                    results.append(result)
                    corrected_item = apply_corrections(item, result, original_turn_count)
                    corrected_items.append(corrected_item)
                else:
                    results.append(result)
                    item['VERIFICATION'] = {'verified': False, 'error': result.get('error')}
                    corrected_items.append(item)

            except Exception as e:
                print(f"Error: {e}")

    # Save corrected dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in corrected_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # Statistics - count from the corrected items (includes existing + new)
    successful = sum(
        1 for item in corrected_items
        if item.get('VERIFICATION', {}).get('verified', False)
    )
    with_issues = sum(
        1 for item in corrected_items
        if item.get('VERIFICATION', {}).get('verified', False) and item.get('VERIFICATION', {}).get('had_issues', False)
    )
    total_issues = sum(
        item.get('VERIFICATION', {}).get('num_issues', 0)
        for item in corrected_items
    )

    # Save report
    report_file = output_file.with_suffix('.report.json')
    report = {
        'language': args.lang,
        'total_items': len(translated_data),
        'successful_verifications': successful,
        'items_with_issues': with_issues,
        'total_issues': total_issues,
        'timestamp': datetime.now().isoformat()
    }

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    print(f"Total items:        {len(translated_data)}")
    print(f"Verified:           {successful}")
    print(f"Items with issues:  {with_issues}")
    print(f"Total issues:       {total_issues}")
    print(f"Output:             {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
