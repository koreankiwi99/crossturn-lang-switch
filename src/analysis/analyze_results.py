#!/usr/bin/env python3
"""Comprehensive analysis of cross-turn language switching results.

This script analyzes results from the codeswitching experiments including:
- Layer 2: Task accuracy by model, condition, and language
- Layer 1: Language fidelity by model, condition, and language
- Conversation length effects on X→EN fidelity
- Language × conversation length interaction
- Distractor condition effects
- Sample size summary

Usage:
    python -m src.analysis.analyze_results
    # or
    python src/analysis/analyze_results.py
"""

from .data_loader import load_all_data, build_qid_to_turns
from .task_accuracy import analyze_task_accuracy
from .language_fidelity import analyze_language_fidelity, analyze_by_language
from .conversation_length import analyze_conversation_length, analyze_language_x_length
from .distractor import analyze_distractor
from .sample_sizes import analyze_sample_sizes


def main():
    """Run all analyses."""
    print("Loading all results data...")
    results = load_all_data()

    # Build question ID to turn count mapping
    qid_to_turns = build_qid_to_turns(results)

    # Run analyses
    analyze_task_accuracy(results)
    analyze_language_fidelity(results)
    analyze_conversation_length(results, qid_to_turns)
    analyze_language_x_length(results, qid_to_turns)
    analyze_by_language(results)
    analyze_distractor(results)
    analyze_sample_sizes(results)


if __name__ == "__main__":
    main()
