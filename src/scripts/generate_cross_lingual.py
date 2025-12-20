#!/usr/bin/env python3
"""
Generate Cross-Lingual Transfer Datasets (X→Y)

Creates datasets where context is in language X and final question is in language Y,
where neither X nor Y is English.

Pairs: ZH→DE, DE→ZH, ES→AR, AR→ES
"""

import json
import os
import argparse
from pathlib import Path


def load_jsonl(filepath):
    """Load JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data, filepath):
    """Save data to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def create_cross_lingual_dataset(source_data, target_data, source_lang, target_lang):
    """
    Create X→Y dataset.
    - Context (all turns except last): from source_lang translation
    - Final question: from target_lang translation
    """
    # Create lookup by question ID base
    def get_base_id(qid):
        # Remove language suffixes like _es_full, _zh_full, etc.
        for suffix in ['_es_full', '_zh_full', '_de_full', '_ar_full', '_en_full']:
            if qid.endswith(suffix):
                return qid.replace(suffix, '')
        return qid

    target_lookup = {get_base_id(item['QUESTION_ID']): item for item in target_data}

    results = []
    for source_item in source_data:
        base_id = get_base_id(source_item['QUESTION_ID'])

        if base_id not in target_lookup:
            continue

        target_item = target_lookup[base_id]

        # Get context from source (all turns except last user turn)
        source_conv = source_item['CONVERSATION']
        target_conv = target_item['CONVERSATION']

        if len(source_conv) != len(target_conv):
            continue

        # Build new conversation: source context + target final turn
        new_conv = source_conv[:-1] + [target_conv[-1]]

        # Create new item
        new_item = {
            'QUESTION_ID': f"{base_id}_{source_lang}_to_{target_lang}",
            'AXIS': source_item['AXIS'],
            'CONVERSATION': new_conv,
            'TARGET_QUESTION': source_item['TARGET_QUESTION'],
            'PASS_CRITERIA': source_item['PASS_CRITERIA'],
            'CODE_SWITCH': {
                'type': 'cross_lingual',
                'source_lang': source_lang,
                'target_lang': target_lang,
                'context_lang': source_lang,
                'query_lang': target_lang
            }
        }

        results.append(new_item)

    return results


def main():
    parser = argparse.ArgumentParser(description='Generate cross-lingual X→Y datasets')
    parser.add_argument('--pairs', type=str, nargs='+',
                        default=['zh_de', 'de_zh', 'es_ar', 'ar_es'],
                        help='Language pairs to generate (e.g., zh_de for ZH→DE)')
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent.parent
    translations_dir = base_dir / 'data' / 'translations'
    output_dir = base_dir / 'data' / 'experiments'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all translations
    print("Loading translations...")
    translations = {}
    for lang in ['de', 'zh', 'es', 'ar']:
        filepath = translations_dir / f'verified_full_translation_{lang}.jsonl'
        if filepath.exists():
            translations[lang] = load_jsonl(filepath)
            print(f"  {lang}: {len(translations[lang])} items")
        else:
            print(f"  WARNING: {filepath} not found")

    # Generate cross-lingual datasets
    for pair in args.pairs:
        source_lang, target_lang = pair.split('_')

        if source_lang not in translations or target_lang not in translations:
            print(f"Skipping {pair}: missing translation data")
            continue

        print(f"\nGenerating {source_lang.upper()}→{target_lang.upper()} dataset...")

        dataset = create_cross_lingual_dataset(
            translations[source_lang],
            translations[target_lang],
            source_lang,
            target_lang
        )

        output_path = output_dir / f'{source_lang}_to_{target_lang}.jsonl'
        save_jsonl(dataset, output_path)
        print(f"  Saved {len(dataset)} items to {output_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
