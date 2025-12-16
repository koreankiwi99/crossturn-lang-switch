#!/usr/bin/env python3
"""
Verify Translation Quality with LLM

This script uses an LLM (GPT-4o) to evaluate whether translations in the
code-switching datasets are semantically correct and faithful to the original English.

Evaluates WHOLE CONVERSATIONS, not individual turns.

Usage:
    python verify_translations.py --dataset full_translation_zh.jsonl
    python verify_translations.py --all-datasets
    python verify_translations.py --all-datasets --samples 50
"""

import os
import json
import argparse
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

VERIFICATION_PROMPT = """You are a professional translator and linguistic expert. Your task is to evaluate the quality of a translated conversation.

## Original English Conversation:
{original_conversation}

## Translated Conversation ({target_lang}):
{translated_conversation}

Please evaluate this translation on the following criteria:

1. **Semantic Accuracy** (1-5): Does the translation preserve the original meaning across all turns?
   - 5: Perfect meaning preservation
   - 4: Minor nuances lost but core meaning intact
   - 3: Some meaning lost or altered
   - 2: Significant meaning errors
   - 1: Completely wrong or nonsensical

2. **Completeness** (1-5): Is all information from the original present?
   - 5: All information preserved
   - 4: Very minor omissions
   - 3: Some information missing
   - 2: Major omissions
   - 1: Most information lost

3. **Critical Constraints Preserved** (Yes/No/NA): If the original contains specific constraints, requirements, or conditions (e.g., "within 5 minutes walk", "no shellfish", "must include X", numerical values, names), are they accurately translated?

4. **Overall Quality** (1-5): Your overall assessment of translation quality.

Respond in this exact JSON format:
{{
    "semantic_accuracy": <1-5>,
    "completeness": <1-5>,
    "constraints_preserved": "<Yes/No/NA>",
    "overall_quality": <1-5>,
    "issues": "<brief description of any issues found, or 'None' if translation is good>"
}}
"""

LANGUAGE_NAMES = {
    "zh": "Chinese (Simplified)",
    "de": "German",
    "es": "Spanish",
    "ar": "Arabic"
}


def load_dataset(filepath):
    """Load a JSONL dataset."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def format_conversation(turns):
    """Format conversation turns into readable text."""
    lines = []
    for i, turn in enumerate(turns):
        role = turn.get('role', 'unknown').upper()
        content = turn.get('content', '') if isinstance(turn, dict) else turn
        # Handle both dict format and tuple format
        if isinstance(turn, dict):
            if 'original' in turn:
                content = turn['original']
            elif 'content' in turn:
                content = turn['content']
        lines.append(f"[{role}]: {content}")
    return "\n\n".join(lines)


def extract_conversation_pair(item):
    """Extract original and translated conversation from a dataset item."""

    # Full translation datasets have TRANSLATION field
    if 'TRANSLATION' in item:
        t = item['TRANSLATION']
        original_turns = []
        translated_turns = []

        for turn in t.get('translated_turns', []):
            original_turns.append({
                'role': turn.get('role', 'user'),
                'content': turn.get('original', '')
            })
            translated_turns.append({
                'role': turn.get('role', 'user'),
                'content': turn.get('translated', '')
            })

        return {
            'type': 'full_translation',
            'language': t.get('target_language', 'unknown'),
            'original_turns': original_turns,
            'translated_turns': translated_turns
        }

    return None


def verify_conversation(client, original_turns, translated_turns, target_lang, model="gpt-4o-2024-08-06"):
    """Use LLM to verify translation quality of a whole conversation."""
    lang_name = LANGUAGE_NAMES.get(target_lang, target_lang)

    original_text = format_conversation(original_turns)
    translated_text = format_conversation(translated_turns)

    prompt = VERIFICATION_PROMPT.format(
        original_conversation=original_text,
        translated_conversation=translated_text,
        target_lang=lang_name
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500
        )

        result_text = response.choices[0].message.content.strip()

        # Extract JSON from response
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0].strip()
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0].strip()

        result = json.loads(result_text)
        return result

    except json.JSONDecodeError as e:
        return {
            "error": f"JSON parse error: {str(e)}",
            "raw_response": result_text[:500] if 'result_text' in locals() else "No response"
        }
    except Exception as e:
        return {"error": str(e)}


def find_datasets(data_dir):
    """Find all full_translation datasets (source of truth for all translations)."""
    datasets = []

    # Full translation datasets contain ALL translations
    ft_dir = os.path.join(data_dir, "codeswitching", "full_translation")
    if os.path.exists(ft_dir):
        for f in os.listdir(ft_dir):
            if f.endswith('.jsonl'):
                datasets.append(os.path.join(ft_dir, f))

    return sorted(datasets)


def main():
    parser = argparse.ArgumentParser(description="Verify translation quality with LLM")

    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to specific dataset file to verify")

    parser.add_argument("--all-datasets", action="store_true",
                        help="Verify all full_translation datasets")

    parser.add_argument("--samples", type=int, default=None,
                        help="Number of samples to verify per dataset (default: all)")

    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for results")

    parser.add_argument("--model", type=str, default="gpt-4o-2024-08-06",
                        help="LLM model to use for verification")

    args = parser.parse_args()

    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")

    # Find datasets to verify
    if args.dataset:
        if os.path.isabs(args.dataset):
            datasets = [args.dataset]
        else:
            candidates = [
                args.dataset,
                os.path.join(data_dir, "codeswitching", "full_translation", args.dataset),
            ]
            datasets = [c for c in candidates if os.path.exists(c)]
            if not datasets:
                print(f"Error: Could not find dataset {args.dataset}")
                return
    elif args.all_datasets:
        datasets = find_datasets(data_dir)
    else:
        print("Please specify --dataset or --all-datasets")
        return

    # Setup output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output or os.path.join(base_dir, "results", "translation_verification")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize OpenAI client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print("=" * 70)
    print("TRANSLATION QUALITY VERIFICATION (Whole Conversation)")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Samples per dataset: {args.samples or 'ALL'}")
    print(f"Datasets to verify: {len(datasets)}")
    for d in datasets:
        print(f"  - {os.path.basename(d)}")
    print("=" * 70)

    all_results = []

    for dataset_path in datasets:
        dataset_name = os.path.basename(dataset_path)
        print(f"\n>>> Verifying: {dataset_name}")

        data = load_dataset(dataset_path)

        # Sample data if requested
        if args.samples and len(data) > args.samples:
            import random
            random.seed(42)  # For reproducibility
            sampled = random.sample(data, args.samples)
        else:
            sampled = data

        dataset_results = {
            "dataset": dataset_name,
            "total_items": len(data),
            "sampled": len(sampled),
            "verifications": [],
            "summary": {}
        }

        scores = {
            "semantic_accuracy": [],
            "completeness": [],
            "overall_quality": [],
            "constraints_preserved": {"Yes": 0, "No": 0, "NA": 0}
        }

        for item in tqdm(sampled, desc="Verifying"):
            conv_pair = extract_conversation_pair(item)

            if not conv_pair:
                continue

            result = verify_conversation(
                client,
                conv_pair['original_turns'],
                conv_pair['translated_turns'],
                conv_pair['language'],
                args.model
            )

            verification = {
                "question_id": item.get('QUESTION_ID', item.get('question_id', 'unknown')),
                "type": conv_pair['type'],
                "language": conv_pair['language'],
                "num_turns": len(conv_pair['original_turns']),
                "verification": result
            }

            dataset_results["verifications"].append(verification)

            # Collect scores
            if "error" not in result:
                for key in ["semantic_accuracy", "completeness", "overall_quality"]:
                    if key in result:
                        scores[key].append(result[key])

                if "constraints_preserved" in result:
                    cp = result["constraints_preserved"]
                    if cp in scores["constraints_preserved"]:
                        scores["constraints_preserved"][cp] += 1

        # Calculate summary
        dataset_results["summary"] = {
            "avg_semantic_accuracy": sum(scores["semantic_accuracy"]) / len(scores["semantic_accuracy"]) if scores["semantic_accuracy"] else 0,
            "avg_completeness": sum(scores["completeness"]) / len(scores["completeness"]) if scores["completeness"] else 0,
            "avg_overall_quality": sum(scores["overall_quality"]) / len(scores["overall_quality"]) if scores["overall_quality"] else 0,
            "constraints_preserved": scores["constraints_preserved"],
            "total_verified": len(dataset_results["verifications"]),
            "low_quality_count": sum(1 for s in scores["overall_quality"] if s <= 2),
        }

        all_results.append(dataset_results)

        # Print summary for this dataset
        s = dataset_results["summary"]
        print(f"  Verified: {s['total_verified']} conversations")
        print(f"  Avg Semantic Accuracy: {s['avg_semantic_accuracy']:.2f}/5")
        print(f"  Avg Completeness: {s['avg_completeness']:.2f}/5")
        print(f"  Avg Overall Quality: {s['avg_overall_quality']:.2f}/5")
        print(f"  Constraints Preserved: Yes={s['constraints_preserved']['Yes']}, No={s['constraints_preserved']['No']}, NA={s['constraints_preserved']['NA']}")
        print(f"  Low Quality (≤2): {s['low_quality_count']}")

    # Save detailed results
    output_file = os.path.join(output_dir, f"verification_{timestamp}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    for result in all_results:
        s = result["summary"]
        print(f"\n{result['dataset']}:")
        print(f"  Quality: {s['avg_overall_quality']:.2f}/5 | Semantic: {s['avg_semantic_accuracy']:.2f}/5 | Complete: {s['avg_completeness']:.2f}/5")
        if s['constraints_preserved']['No'] > 0:
            print(f"  ⚠️  Constraints NOT preserved: {s['constraints_preserved']['No']} cases")

    print(f"\n\nDetailed results saved to: {output_file}")

    # Also save a summary file
    summary_file = os.path.join(output_dir, f"summary_{timestamp}.json")
    summary = {
        "timestamp": timestamp,
        "model": args.model,
        "samples_per_dataset": args.samples or "all",
        "datasets": [
            {
                "name": r["dataset"],
                "summary": r["summary"]
            }
            for r in all_results
        ]
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
