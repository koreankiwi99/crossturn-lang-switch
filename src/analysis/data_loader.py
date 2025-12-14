"""Data loading functions for analysis."""

import json
from collections import defaultdict
from pathlib import Path

from .config import RESULTS_DIR, MODELS, CONDITIONS, LANGUAGES


def load_jsonl(filepath):
    """Load JSONL file."""
    data = []
    if filepath.exists():
        with open(filepath) as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data


def load_all_data():
    """Load all evaluation and language fidelity data."""
    results = {
        "task_accuracy": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        "language_fidelity": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        "raw_language": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
    }

    for model in MODELS:
        model_dir = RESULTS_DIR / model
        if not model_dir.exists():
            continue

        for condition in CONDITIONS:
            cond_dir = model_dir / condition
            if not cond_dir.exists():
                continue

            # Load task accuracy (evaluated files)
            if condition == "baseline":
                # Try standard naming first
                eval_file = cond_dir / "evaluated_en.jsonl"
                if eval_file.exists():
                    data = load_jsonl(eval_file)
                    for item in data:
                        passed = item.get("evaluation", {}).get("passed", False)
                        turn_count = item.get("turn_count", 0)
                        results["task_accuracy"][model][condition]["en"].append({
                            "passed": passed,
                            "turn_count": turn_count,
                            "question_id": item.get("question_id")
                        })
                else:
                    # Try timestamp-based files (Gemini format)
                    for eval_file in sorted(cond_dir.glob("evaluated_*.jsonl")):
                        data = load_jsonl(eval_file)
                        for item in data:
                            passed = item.get("evaluation", {}).get("passed", False)
                            turn_count = item.get("turn_count", 0)
                            results["task_accuracy"][model][condition]["en"].append({
                                "passed": passed,
                                "turn_count": turn_count,
                                "question_id": item.get("question_id")
                            })
                        break  # Only use first (most recent) file
            else:
                # Try standard naming first
                found_standard = False
                for lang in LANGUAGES:
                    eval_file = cond_dir / f"evaluated_{lang}.jsonl"
                    if eval_file.exists():
                        found_standard = True
                        data = load_jsonl(eval_file)
                        for item in data:
                            passed = item.get("evaluation", {}).get("passed", False)
                            turn_count = item.get("turn_count", 0)
                            results["task_accuracy"][model][condition][lang].append({
                                "passed": passed,
                                "turn_count": turn_count,
                                "question_id": item.get("question_id")
                            })

                # If no standard files found, try timestamp-based (Gemini format)
                if not found_standard:
                    # Process ALL timestamp files (each may contain a different language)
                    for eval_file in sorted(cond_dir.glob("evaluated_*.jsonl")):
                        data = load_jsonl(eval_file)
                        for item in data:
                            qid = item.get("question_id", "")
                            # Extract language from question_id (e.g., "xxx_de" -> "de")
                            lang = None
                            for l in LANGUAGES:
                                if f"_{l}" in qid:
                                    lang = l
                                    break
                            if lang:
                                passed = item.get("evaluation", {}).get("passed", False)
                                turn_count = item.get("turn_count", 0)
                                results["task_accuracy"][model][condition][lang].append({
                                    "passed": passed,
                                    "turn_count": turn_count,
                                    "question_id": qid
                                })

            # Load language fidelity
            if condition != "baseline":
                for lang in LANGUAGES:
                    lang_file = cond_dir / f"language_eval_{lang}.jsonl"
                    if lang_file.exists():
                        data = load_jsonl(lang_file)
                        for item in data:
                            match = item.get("match_status") == "match"
                            qid = item.get("question_id", "")
                            results["raw_language"][model][condition][lang].append({
                                "match": match,
                                "question_id": qid,
                                "detected": item.get("detected_language"),
                                "expected": item.get("expected_language")
                            })

    return results


def build_qid_to_turns(results):
    """Build question_id to turn_count mapping."""
    qid_to_turns = {}
    for model in MODELS:
        for lang_data in results["task_accuracy"][model]["baseline"].values():
            for item in lang_data:
                qid = item.get("question_id", "")
                turn_count = item.get("turn_count", 0)
                if qid and turn_count:
                    qid_to_turns[qid] = turn_count
        # Also check codeswitching conditions
        for cond in ["codeswitching", "codeswitching_reverse"]:
            for lang_data in results["task_accuracy"][model][cond].values():
                for item in lang_data:
                    qid = item.get("question_id", "")
                    turn_count = item.get("turn_count", 0)
                    if qid and turn_count:
                        # Strip language suffix if present
                        base_qid = qid.split("_")[0]
                        qid_to_turns[base_qid] = turn_count
    return qid_to_turns
