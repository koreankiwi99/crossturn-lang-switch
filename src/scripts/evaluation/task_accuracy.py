#!/usr/bin/env python3
"""
Evaluate Model Responses with GPT-4o Judge (Parallel Version)

Reads responses and evaluates each with GPT-4o using parallel workers.
Uses official MultiChallenge evaluation prompt.

Usage:
    python evaluate.py --input results/apertus-70b/responses_20251207_143000.jsonl
    python evaluate.py --input results/gemini-3-pro/codeswitching/*.jsonl --workers 64
"""

import os
import sys
import json
import argparse
import glob
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv
from gpt_batch.batcher import GPTBatcher

load_dotenv()

JUDGE_MODEL = "gpt-4o-2024-08-06"


def load_judge_prompt():
    """Load judge prompt from prompts/judge_prompt.txt."""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    prompt_path = os.path.join(base_dir, "prompts/judge_prompt.txt")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_responses(input_path):
    """Load responses from JSONL file."""
    responses = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                responses.append(json.loads(line))
    return responses


def evaluate_responses_batch(responses, output_dir, num_workers=64):
    """Evaluate all responses using parallel GPTBatcher."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_name = responses[0].get("model", "unknown") if responses else "unknown"
    output_file = f"{output_dir}/evaluated_{timestamp}.jsonl"
    summary_file = f"{output_dir}/summary_{timestamp}.json"

    judge_prompt_template = load_judge_prompt()

    # Filter valid responses
    valid_responses = [r for r in responses if r.get("success")]
    skipped = len(responses) - len(valid_responses)

    print(f"\nEvaluating {len(valid_responses)} responses with {JUDGE_MODEL}...")
    print(f"Workers: {num_workers}")
    print(f"Output: {output_file}")
    print("=" * 60)

    # Prepare prompts for batch processing
    prompts = []
    for item in valid_responses:
        prompt = judge_prompt_template.format(
            response=item["response"],
            target_question=item["target_question"]
        )
        prompts.append(prompt)

    # Initialize batcher with parallel workers
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment")

    batcher = GPTBatcher(
        api_key=api_key,
        model_name=JUDGE_MODEL,
        system_prompt="",
        temperature=0,
        num_workers=num_workers,
        timeout_duration=60,
        retry_attempts=2,
    )

    # Run batch evaluation
    print(f"Running batch evaluation with {num_workers} parallel workers...")
    judge_responses = batcher.handle_message_list(prompts)

    # Process results
    stats = {"total": 0, "passed": 0, "failed": 0, "error": 0, "skipped": skipped}
    axis_stats = defaultdict(lambda: {"total": 0, "passed": 0})
    results = []

    for item, judge_text in zip(valid_responses, judge_responses):
        axis = item.get("axis", "UNKNOWN")
        stats["total"] += 1
        axis_stats[axis]["total"] += 1

        # Parse verdict
        if judge_text is None or judge_text.strip() == "":
            verdict = "ERROR"
            reasoning = "Empty response from judge"
        else:
            judge_text_upper = judge_text.upper().strip()
            if judge_text_upper.endswith("YES"):
                verdict = "YES"
            elif judge_text_upper.endswith("NO"):
                verdict = "NO"
            else:
                last_line = judge_text.strip().split('\n')[-1].upper()
                verdict = "YES" if "YES" in last_line else "NO"
            reasoning = judge_text

        pass_criteria = item.get("pass_criteria", "YES")
        passed = (verdict == pass_criteria)

        if verdict == "ERROR":
            stats["error"] += 1
        elif passed:
            stats["passed"] += 1
            axis_stats[axis]["passed"] += 1
        else:
            stats["failed"] += 1

        result = {
            **item,
            "evaluation": {
                "judge_model": JUDGE_MODEL,
                "verdict": verdict,
                "pass_criteria": pass_criteria,
                "passed": passed,
                "reasoning": reasoning,
            }
        }
        results.append(result)

    # Write results
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    # Summary
    pass_rate = stats["passed"] / max(stats["total"], 1) * 100

    summary = {
        "model": model_name,
        "judge_model": JUDGE_MODEL,
        "timestamp": timestamp,
        "stats": stats,
        "pass_rate": pass_rate,
        "axis_stats": {
            axis: {
                **ax_stats,
                "pass_rate": ax_stats["passed"] / max(ax_stats["total"], 1) * 100
            }
            for axis, ax_stats in axis_stats.items()
        },
        "evaluated_file": output_file,
    }

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nOverall: {stats['passed']}/{stats['total']} passed ({pass_rate:.1f}%)")
    print(f"Errors: {stats['error']}, Skipped: {stats['skipped']}")

    print("\nBy Axis:")
    for axis, ax_stats in axis_stats.items():
        ax_rate = ax_stats["passed"] / max(ax_stats["total"], 1) * 100
        print(f"  {axis}: {ax_stats['passed']}/{ax_stats['total']} ({ax_rate:.1f}%)")

    print(f"\nResults: {output_file}")
    print(f"Summary: {summary_file}")

    return output_file, summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate responses with GPT-4o judge")

    parser.add_argument("--input", type=str, required=True,
                       help="Input responses file (JSONL)")
    parser.add_argument("--workers", type=int, default=64,
                       help="Number of parallel workers (default: 64)")

    args = parser.parse_args()

    input_files = glob.glob(args.input)
    if not input_files:
        print(f"Error: No files found matching {args.input}")
        sys.exit(1)

    input_file = sorted(input_files)[-1]

    print("=" * 60)
    print("EVALUATE: GPT-4o Judge (Parallel)")
    print("=" * 60)
    print(f"Input: {input_file}")
    print(f"Judge: {JUDGE_MODEL}")

    responses = load_responses(input_file)
    print(f"Loaded {len(responses)} responses")

    output_dir = os.path.dirname(input_file)

    evaluate_responses_batch(responses, output_dir, num_workers=args.workers)


if __name__ == "__main__":
    main()
