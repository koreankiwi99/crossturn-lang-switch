#!/usr/bin/env python3
"""
Layer 1 Evaluation: Language Fidelity (Parallel Version)

Evaluates whether models respond in the expected language.
Uses GPTBatcher for parallel processing.

Expected language rules:
- baseline: English (en)
- EN→X (codeswitching): Target language (query was in X)
- X→EN (codeswitching_reverse): English (query was in English)
- full_translation: Target language (entire conversation in X)
- distractor: Target language (query was translated to X)
- distractor_multi: Target language (query was translated to X)

Usage:
    python evaluate_language.py --input results/claude-opus-4.5-openrouter/baseline/responses_*.jsonl
    python evaluate_language.py --input results/*/codeswitching/*.jsonl --method llm --workers 64
    python evaluate_language.py --input results/qwen3-235b/ --recursive --workers 64
"""

import os
import sys
import json
import argparse
import glob
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from gpt_batch.batcher import GPTBatcher

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

# Load prompts from files (go up 4 levels: evaluation -> scripts -> src -> project_root)
PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "prompts"

def load_prompt(filename):
    """Load prompt from file."""
    with open(PROMPTS_DIR / filename, 'r', encoding='utf-8') as f:
        return f.read()

LANGUAGE_JUDGE_PROMPT = load_prompt("language_judge_prompt.txt")
LANGUAGE_VERIFY_PROMPT = load_prompt("language_verify_prompt.txt")


import re

def clean_response_text(text):
    """Clean response text before language detection."""
    # Remove <thinking>...</thinking> blocks (Claude extended thinking)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    return text.strip()


def get_openai_client():
    """Get OpenAI client for language judge."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")
    return OpenAI(api_key=api_key)


def detect_language_llm(client, response_text, max_chars=2000):
    """Detect language using LLM judge."""
    # Clean and truncate
    text = clean_response_text(response_text)
    text = text[:max_chars] if len(text) > max_chars else text

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


def verify_language_llm(client, response_text, expected_lang, max_chars=2000):
    """Verify if response is in expected language using LLM judge."""
    # Clean and truncate
    text = clean_response_text(response_text)

    # For very short responses (< 10 chars), assume they match expected language
    # Single words like "No", "Sí", "OK", etc. can't reliably be attributed to any language
    if len(text.strip()) < 10:
        return True, expected_lang

    text = text[:max_chars] if len(text) > max_chars else text

    expected_lang_name = LANG_NAMES.get(expected_lang, expected_lang)

    try:
        result = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": LANGUAGE_VERIFY_PROMPT.format(
                response=text,
                expected_lang_name=expected_lang_name
            )}],
            temperature=0.0,
            max_tokens=10
        )
        answer = result.choices[0].message.content.strip().lower()

        if "yes" in answer:
            return True, expected_lang
        else:
            return False, "other"

    except Exception as e:
        return False, f"error:{str(e)[:50]}"


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
    filename = os.path.basename(filepath).lower()

    if "distractor_multi" in path_lower or ("distractor" in path_lower and "multi" in path_lower):
        return "distractor_multi"
    elif "distractor" in path_lower:
        return "distractor"
    elif "codeswitching_reverse" in path_lower or "_to_en" in path_lower:
        return "codeswitching_reverse"
    elif "en_to_" in filename:
        # EN→X codeswitching (e.g., responses_en_to_de_*.jsonl)
        return "codeswitching"
    elif "codeswitching" in path_lower:
        return "codeswitching"
    elif "full_translation" in path_lower:
        return "full_translation"
    elif "baseline_en" in filename or filename.startswith("responses_2"):
        # English baseline only (baseline_en or just timestamp)
        return "baseline"
    elif "baseline_" in filename:
        # baseline_ar/de/es/zh are full translations (not English baseline)
        return "full_translation"
    elif "baseline" in path_lower:
        return "baseline"
    # Cross-lingual: X→Y where neither is English (e.g., de_to_zh, es_to_ar)
    elif "_to_" in filename:
        return "cross_lingual"
    else:
        return "unknown"


def extract_dataset_name(filepath):
    """Extract dataset name from response filename (e.g., baseline_en, en_to_de, ar_to_en)."""
    filename = os.path.basename(filepath)

    # List of known dataset patterns
    datasets = [
        "baseline_en", "baseline_ar", "baseline_de", "baseline_es", "baseline_zh",
        "en_to_ar", "en_to_de", "en_to_es", "en_to_zh",
        "ar_to_en", "de_to_en", "es_to_en", "zh_to_en"
    ]

    for ds in datasets:
        if ds in filename:
            return ds

    return None


def extract_target_language(filepath):
    """Extract target language from response filename or path.

    For cross-lingual patterns (X_to_Y), returns Y (the query/target language).
    """
    filename = os.path.basename(filepath)

    # Cross-lingual pattern: {lang1}_to_{lang2} - return lang2 (query language)
    import re
    cross_lingual_match = re.search(r'_(de|es|ar|zh)_to_(de|es|ar|zh)', filename)
    if cross_lingual_match:
        return cross_lingual_match.group(2)  # Return the target (query) language

    # EN→X pattern: en_to_{lang} - return lang
    en_to_match = re.search(r'_en_to_(de|es|ar|zh)', filename)
    if en_to_match:
        return en_to_match.group(1)

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


def evaluate_language_fidelity_batch(responses, condition_type, target_lang, output_dir, method="verify", num_workers=64, dataset_name=None):
    """Evaluate language fidelity for all responses using parallel GPTBatcher."""
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Output filename includes dataset name (if provided) or language, plus timestamp
    if dataset_name:
        suffix = f"_{dataset_name}_{timestamp}"
    else:
        suffix = f"_{target_lang}_{timestamp}" if target_lang else f"_{timestamp}"
    output_file = f"{output_dir}/language_eval{suffix}.jsonl"
    summary_file = f"{output_dir}/language_summary{suffix}.json"

    # Get expected language
    expected = get_expected_language(condition_type, target_lang)
    expected_lang_name = LANG_NAMES.get(expected, expected)

    print(f"\nEvaluating language fidelity ({method}, {num_workers} workers)...")
    print(f"Condition: {condition_type}, Target: {target_lang}, Expected: {expected}")
    print(f"Output: {output_file}")
    print("=" * 60)

    # Filter valid responses and prepare prompts
    valid_responses = []
    prompts = []
    max_chars = 2000

    for item in responses:
        if not item.get("success") or not item.get("response"):
            continue

        text = clean_response_text(item["response"])

        # For very short responses, we'll handle them separately
        if len(text.strip()) < 10:
            valid_responses.append((item, "short"))
            prompts.append(None)  # Placeholder
            continue

        text = text[:max_chars] if len(text) > max_chars else text

        if method == "verify":
            prompt = LANGUAGE_VERIFY_PROMPT.format(response=text, expected_lang_name=expected_lang_name)
        else:  # llm
            prompt = LANGUAGE_JUDGE_PROMPT.format(response=text)

        valid_responses.append((item, "normal"))
        prompts.append(prompt)

    skipped = len(responses) - len(valid_responses)

    # Filter out None prompts for batch processing
    batch_indices = [i for i, p in enumerate(prompts) if p is not None]
    batch_prompts = [prompts[i] for i in batch_indices]

    print(f"Processing {len(batch_prompts)} responses with GPTBatcher...")

    # Initialize batcher
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")

    batcher = GPTBatcher(
        api_key=api_key,
        model_name=JUDGE_MODEL,
        system_prompt="",
        temperature=0,
        num_workers=num_workers,
        timeout_duration=30,
        retry_attempts=2,
    )

    # Run batch
    batch_results = batcher.handle_message_list(batch_prompts) if batch_prompts else []

    # Map results back
    batch_result_map = {batch_indices[i]: batch_results[i] for i in range(len(batch_results))}

    # Process results
    stats = {
        "total": 0,
        "match": 0,
        "mismatch": 0,
        "mixed": 0,
        "error": 0,
        "skipped": skipped,
    }
    detected_distribution = defaultdict(int)
    results = []

    for i, (item, item_type) in enumerate(valid_responses):
        stats["total"] += 1

        if item_type == "short":
            # Short responses assumed to match
            detected = expected
            match_status = "match"
            stats["match"] += 1
        else:
            judge_text = batch_result_map.get(i, "")

            if method == "verify":
                if judge_text is None or judge_text.strip() == "":
                    detected = "error"
                    match_status = "error"
                    stats["error"] += 1
                elif "yes" in judge_text.lower():
                    detected = expected
                    match_status = "match"
                    stats["match"] += 1
                else:
                    detected = "other"
                    match_status = "mismatch"
                    stats["mismatch"] += 1
            else:  # llm method
                if judge_text is None or judge_text.strip() == "":
                    detected = "error"
                    match_status = "error"
                    stats["error"] += 1
                else:
                    detected = judge_text.strip().lower()
                    # Normalize
                    if detected in ["en", "english"]:
                        detected = "en"
                    elif detected in ["de", "german", "deutsch"]:
                        detected = "de"
                    elif detected in ["es", "spanish", "español"]:
                        detected = "es"
                    elif detected in ["ar", "arabic"]:
                        detected = "ar"
                    elif detected in ["zh", "zh-cn", "chinese", "中文"]:
                        detected = "zh"
                    elif "mixed" in detected:
                        detected = "mixed"

                    if detected == "mixed":
                        match_status = "mixed"
                        stats["mixed"] += 1
                    elif detected == expected:
                        match_status = "match"
                        stats["match"] += 1
                    else:
                        match_status = "mismatch"
                        stats["mismatch"] += 1

        detected_distribution[detected] += 1

        result = {
            "question_id": item.get("question_id"),
            "expected_language": expected,
            "detected_language": detected,
            "match_status": match_status,
            "response_preview": item["response"][:200] + "..." if len(item["response"]) > 200 else item["response"],
        }
        results.append(result)

    # Write results
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Calculate rates
    fidelity_rate = stats["match"] / max(stats["total"], 1) * 100

    summary = {
        "condition_type": condition_type,
        "target_language": target_lang,
        "expected_response_language": expected,
        "method": method,
        "judge_model": JUDGE_MODEL,
        "num_workers": num_workers,
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


def evaluate_language_fidelity(responses, client, condition_type, target_lang, output_dir, method="llm", dataset_name=None):
    """Evaluate language fidelity for all responses (sequential, deprecated)."""
    os.makedirs(output_dir, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Output filename includes dataset name (if provided) or language, plus timestamp
    if dataset_name:
        suffix = f"_{dataset_name}_{timestamp}"
    else:
        suffix = f"_{target_lang}_{timestamp}" if target_lang else f"_{timestamp}"
    lang_suffix = suffix  # Keep for compatibility
    output_file = f"{output_dir}/language_eval{lang_suffix}.jsonl"
    summary_file = f"{output_dir}/language_summary{lang_suffix}.json"

    # Clear output file before starting (avoid accumulating results from multiple runs)
    open(output_file, 'w').close()

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

        # Detect/verify language
        if method == "verify":
            is_match, detected = verify_language_llm(client, item["response"], expected)
            if isinstance(is_match, bool) and is_match:
                match_status = "match"
                stats["match"] += 1
                detected = expected  # Set detected to expected for consistency
            elif detected.startswith("error"):
                match_status = "error"
                stats["error"] += 1
            else:
                match_status = "mismatch"
                stats["mismatch"] += 1
        elif method == "llm":
            detected = detect_language_llm(client, item["response"])
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
        else:
            detected = detect_language_fasttext(item["response"])
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

        detected_distribution[detected] += 1

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

    parser.add_argument("--method", type=str, default="verify",
                       choices=["llm", "fasttext", "verify"],
                       help="Detection method: verify (recommended), llm, or fasttext")

    parser.add_argument("--condition", type=str, default=None,
                       choices=["baseline", "codeswitching", "codeswitching_reverse",
                               "full_translation", "distractor", "distractor_multi"],
                       help="Override condition type (default: infer from path)")

    parser.add_argument("--recursive", action="store_true",
                       help="Process all response files in directory recursively")

    parser.add_argument("--workers", type=int, default=64,
                       help="Number of parallel workers (default: 64)")

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
    print("LAYER 1 EVALUATION: Language Fidelity (Parallel)")
    print("=" * 60)
    print(f"Files to process: {len(input_files)}")
    print(f"Method: {args.method}")
    print(f"Workers: {args.workers}")

    all_summaries = []

    for input_file in sorted(input_files):
        print(f"\n{'='*60}")
        print(f"Processing: {input_file}")

        condition_type = args.condition or infer_condition_type(input_file)
        target_lang = extract_target_language(input_file)
        dataset_name = extract_dataset_name(input_file)
        print(f"Condition: {condition_type}, Target lang: {target_lang}, Dataset: {dataset_name}")

        responses = load_responses(input_file)
        print(f"Loaded {len(responses)} responses")

        output_dir = os.path.dirname(input_file)

        if args.method == "fasttext":
            # Use sequential for fasttext (no API calls)
            client = None
            summary = evaluate_language_fidelity(
                responses, client, condition_type, target_lang, output_dir, args.method
            )
        else:
            # Use parallel batch processing for LLM methods
            summary = evaluate_language_fidelity_batch(
                responses, condition_type, target_lang, output_dir, args.method, args.workers, dataset_name
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
