#!/usr/bin/env python3
"""
Layer 1 Evaluation: Language Fidelity

Evaluates whether models respond in the expected language.

Expected language rules:
- baseline: English (en)
- EN→X (codeswitching): Target language (query was in X)
- X→EN (codeswitching_reverse): English (query was in English)
- full_translation: Target language (entire conversation in X)
- distractor: Target language (query was translated to X)
- distractor_multi: Target language (query was translated to X)

Usage:
    python evaluate_language.py --input results/claude-opus-4.5-openrouter/baseline/responses_*.jsonl
    python evaluate_language.py --input results/*/codeswitching/*.jsonl --method llm
    python evaluate_language.py --input results/qwen3-235b/ --recursive
"""

import os
import sys
import json
import argparse
import glob
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

# Language detection judge model
JUDGE_MODEL = "gpt-4o-mini"  # Cheaper for simple language detection

# Language codes mapping
LANG_NAMES = {
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "ar": "Arabic",
    "zh": "Chinese",
}

LANGUAGE_JUDGE_PROMPT = """Identify the PRIMARY language of this text.

<TEXT>
{response}
</TEXT>

Rules:
- If the text is >90% in one language, return that language code
- If genuinely mixed (significant portions in multiple languages), return "mixed"
- Focus on the main content, ignore code snippets or proper nouns

Return ONLY one of these codes: en, de, es, ar, zh, mixed

Your answer:"""


def get_openai_client():
    """Get OpenAI client for language judge."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    return OpenAI(api_key=api_key)


def detect_language_llm(client, response_text, max_chars=2000):
    """Detect language using LLM judge."""
    # Truncate very long responses
    text = response_text[:max_chars] if len(response_text) > max_chars else response_text

    try:
        result = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": LANGUAGE_JUDGE_PROMPT.format(response=text)}],
            temperature=0.0,
            max_tokens=10
        )
        detected = result.choices[0].message.content.strip().lower()

        # Normalize response
        if detected in ["en", "english"]:
            return "en"
        elif detected in ["de", "german", "deutsch"]:
            return "de"
        elif detected in ["es", "spanish", "español"]:
            return "es"
        elif detected in ["ar", "arabic"]:
            return "ar"
        elif detected in ["zh", "zh-cn", "chinese", "中文"]:
            return "zh"
        elif "mixed" in detected:
            return "mixed"
        else:
            return detected  # Return as-is for debugging

    except Exception as e:
        return f"error:{str(e)[:50]}"


def detect_language_fasttext(response_text):
    """Detect language using fasttext (fallback, faster)."""
    try:
        from langdetect import detect
        detected = detect(response_text)
        # Normalize
        if detected == "zh-cn" or detected == "zh-tw":
            return "zh"
        return detected
    except:
        return "unknown"


def get_expected_language(condition_type, target_lang):
    """
    Determine expected response language based on condition.

    Args:
        condition_type: baseline/codeswitching/codeswitching_reverse/
                       full_translation/distractor/distractor_multi
        target_lang: Target language from filename (de/es/ar/zh)

    Returns:
        Expected response language code
    """
    if condition_type == "baseline":
        return "en"
    elif condition_type == "codeswitching_reverse":
        # X→EN: Query in English, expect English response
        return "en"
    else:
        # codeswitching, full_translation, distractor, distractor_multi
        # Query in target language, expect target language response
        return target_lang or "unknown"


def infer_condition_type(filepath):
    """Infer condition type from filepath."""
    path_lower = filepath.lower()

    if "distractor_multi" in path_lower or ("distractor" in path_lower and "multi" in path_lower):
        return "distractor_multi"
    elif "distractor" in path_lower:
        return "distractor"
    elif "codeswitching_reverse" in path_lower or "_to_en" in path_lower:
        return "codeswitching_reverse"
    elif "codeswitching" in path_lower:
        return "codeswitching"
    elif "full_translation" in path_lower:
        return "full_translation"
    elif "baseline" in path_lower:
        return "baseline"
    else:
        # Check filename
        filename = os.path.basename(filepath)
        if filename.startswith("responses_2"):  # responses_20251212... format (baseline)
            return "baseline"
        return "unknown"


def extract_target_language(filepath):
    """Extract target language from response filename or path."""
    filename = os.path.basename(filepath)

    # Pattern: responses_codeswitching_de_20251212.jsonl
    # Pattern: responses_full_translation_zh_20251212.jsonl
    for lang in ["de", "es", "ar", "zh"]:
        if f"_{lang}_" in filename or f"_{lang}." in filename:
            return lang
        # Also check _to_en pattern for reverse
        if f"_{lang}_to_en" in filename:
            return lang

    # Check directory name
    parent = os.path.basename(os.path.dirname(filepath))
    for lang in ["de", "es", "ar", "zh"]:
        if parent == lang:
            return lang

    return None


def load_responses(input_path):
    """Load responses from JSONL file."""
    responses = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                responses.append(json.loads(line))
    return responses


def evaluate_language_fidelity(responses, client, condition_type, target_lang, output_dir, method="llm"):
    """Evaluate language fidelity for all responses."""
    os.makedirs(output_dir, exist_ok=True)

    # Output filename includes language
    lang_suffix = f"_{target_lang}" if target_lang else ""
    output_file = f"{output_dir}/language_eval{lang_suffix}.jsonl"
    summary_file = f"{output_dir}/language_summary{lang_suffix}.json"

    # Get expected language
    expected = get_expected_language(condition_type, target_lang)

    print(f"\nEvaluating language fidelity ({method})...")
    print(f"Condition: {condition_type}, Target: {target_lang}, Expected: {expected}")
    print(f"Output: {output_file}")
    print("=" * 60)

    results = []
    stats = {
        "total": 0,
        "match": 0,
        "mismatch": 0,
        "mixed": 0,
        "error": 0,
        "skipped": 0,
    }

    detected_distribution = defaultdict(int)

    for item in tqdm(responses, desc="Detecting language"):
        if not item.get("success") or not item.get("response"):
            stats["skipped"] += 1
            continue

        stats["total"] += 1

        # Detect actual language
        if method == "llm":
            detected = detect_language_llm(client, item["response"])
        else:
            detected = detect_language_fasttext(item["response"])

        detected_distribution[detected] += 1

        # Determine match
        if detected.startswith("error"):
            match_status = "error"
            stats["error"] += 1
        elif detected == "mixed":
            match_status = "mixed"
            stats["mixed"] += 1
        elif detected == expected:
            match_status = "match"
            stats["match"] += 1
        else:
            match_status = "mismatch"
            stats["mismatch"] += 1

        result = {
            "question_id": item.get("question_id"),
            "expected_language": expected,
            "detected_language": detected,
            "match_status": match_status,
            "response_preview": item["response"][:200] + "..." if len(item["response"]) > 200 else item["response"],
        }
        results.append(result)

        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Calculate rates
    fidelity_rate = stats["match"] / max(stats["total"], 1) * 100

    summary = {
        "condition_type": condition_type,
        "target_language": target_lang,
        "expected_response_language": expected,
        "method": method,
        "judge_model": JUDGE_MODEL if method == "llm" else "langdetect",
        "stats": stats,
        "fidelity_rate": fidelity_rate,
        "detected_distribution": dict(detected_distribution),
        "output_file": output_file,
    }

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("LANGUAGE FIDELITY RESULTS")
    print("=" * 60)
    print(f"Expected: {expected}")
    print(f"Fidelity: {stats['match']}/{stats['total']} ({fidelity_rate:.1f}%)")
    print(f"Mismatched: {stats['mismatch']}, Mixed: {stats['mixed']}, Errors: {stats['error']}")

    print("\nDetected Distribution:")
    for lang, count in sorted(detected_distribution.items(), key=lambda x: -x[1]):
        pct = count / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {lang}: {count} ({pct:.1f}%)")

    print(f"\nResults: {output_file}")
    print(f"Summary: {summary_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate language fidelity (Layer 1)")

    parser.add_argument("--input", type=str, required=True,
                       help="Input responses file(s) - supports glob patterns")

    parser.add_argument("--method", type=str, default="llm",
                       choices=["llm", "fasttext"],
                       help="Detection method (default: llm)")

    parser.add_argument("--condition", type=str, default=None,
                       choices=["baseline", "codeswitching", "codeswitching_reverse",
                               "full_translation", "distractor", "distractor_multi"],
                       help="Override condition type (default: infer from path)")

    parser.add_argument("--recursive", action="store_true",
                       help="Process all response files in directory recursively")

    args = parser.parse_args()

    # Get input files
    if args.recursive and os.path.isdir(args.input):
        input_files = glob.glob(os.path.join(args.input, "**", "responses_*.jsonl"), recursive=True)
    else:
        input_files = glob.glob(args.input)

    if not input_files:
        print(f"Error: No files found matching {args.input}")
        sys.exit(1)

    print("=" * 60)
    print("LAYER 1 EVALUATION: Language Fidelity")
    print("=" * 60)
    print(f"Files to process: {len(input_files)}")
    print(f"Method: {args.method}")

    client = get_openai_client() if args.method == "llm" else None

    all_summaries = []

    for input_file in sorted(input_files):
        print(f"\n{'='*60}")
        print(f"Processing: {input_file}")

        condition_type = args.condition or infer_condition_type(input_file)
        target_lang = extract_target_language(input_file)
        print(f"Condition: {condition_type}, Target lang: {target_lang}")

        responses = load_responses(input_file)
        print(f"Loaded {len(responses)} responses")

        output_dir = os.path.dirname(input_file)

        summary = evaluate_language_fidelity(
            responses, client, condition_type, target_lang, output_dir, args.method
        )
        all_summaries.append({
            "file": input_file,
            "condition": condition_type,
            "target_lang": target_lang,
            **summary
        })

    # Print overall summary if multiple files
    if len(all_summaries) > 1:
        print("\n" + "=" * 60)
        print("OVERALL SUMMARY")
        print("=" * 60)

        total_match = sum(s["stats"]["match"] for s in all_summaries)
        total_total = sum(s["stats"]["total"] for s in all_summaries)
        overall_rate = total_match / max(total_total, 1) * 100

        print(f"\nOverall Fidelity: {total_match}/{total_total} ({overall_rate:.1f}%)")
        print("\nBy Condition:")
        for s in all_summaries:
            print(f"  {s['condition']}: {s['fidelity_rate']:.1f}%")


if __name__ == "__main__":
    main()
