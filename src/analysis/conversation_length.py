"""Conversation length analysis."""

from collections import defaultdict
import scipy.stats as stats

from .config import MODELS, MODEL_DISPLAY, LANGUAGES
from .stats import compute_correlation, compute_chi_square, format_correlation


def analyze_conversation_length(results, qid_to_turns):
    """Analyze X→EN fidelity by conversation length."""
    print("\n" + "=" * 60)
    print("CONVERSATION LENGTH ANALYSIS (X→EN Fidelity)")
    print("=" * 60)

    print(f"\nQuestion ID to turn count mapping: {len(qid_to_turns)} entries")

    # Analyze X→EN fidelity by conversation length
    print("\n### X→EN Fidelity by Conversation Length")
    print("| Model | Short (2-3) | Medium (4-5) | Long (6+) | Trend |")
    print("|-------|-------------|--------------|-----------|-------|")

    length_stats = {}
    for model in MODELS:
        all_data = []
        for lang in LANGUAGES:
            data = results["raw_language"][model]["codeswitching_reverse"].get(lang, [])
            for item in data:
                qid = item.get("question_id", "")
                # Try to extract base question ID
                base_qid = qid.replace(f"_{lang}_reverse", "").replace(f"_{lang}", "")
                turn_count = qid_to_turns.get(base_qid, 0)
                if turn_count > 0:
                    all_data.append({"match": item["match"], "turn_count": turn_count})

        if not all_data:
            print(f"| {MODEL_DISPLAY[model]} | - | - | - | - |")
            continue

        # Bucket analysis
        short_data = [d for d in all_data if 2 <= d["turn_count"] <= 3]
        medium_data = [d for d in all_data if 4 <= d["turn_count"] <= 5]
        long_data = [d for d in all_data if d["turn_count"] >= 6]

        short_rate = sum(d["match"] for d in short_data) / len(short_data) * 100 if short_data else None
        medium_rate = sum(d["match"] for d in medium_data) / len(medium_data) * 100 if medium_data else None
        long_rate = sum(d["match"] for d in long_data) / len(long_data) * 100 if long_data else None

        length_stats[model] = {
            "short": (short_rate, len(short_data)),
            "medium": (medium_rate, len(medium_data)),
            "long": (long_rate, len(long_data)),
            "all_data": all_data,
            "short_data": short_data,
            "medium_data": medium_data,
            "long_data": long_data
        }

        # Compute trend (correlation)
        corr, p_value = compute_correlation(all_data)
        trend = format_correlation(corr, p_value)

        short_str = f"{short_rate:.1f}% (n={len(short_data)})" if short_rate else "-"
        medium_str = f"{medium_rate:.1f}% (n={len(medium_data)})" if medium_rate else "-"
        long_str = f"{long_rate:.1f}% (n={len(long_data)})" if long_rate else "-"

        print(f"| {MODEL_DISPLAY[model]} | {short_str} | {medium_str} | {long_str} | {trend} |")

    # Chi-square test for all models
    print("\n### Statistical Tests (Conversation Length Effect)")
    for model in MODELS:
        if model not in length_stats or not length_stats[model]["all_data"]:
            continue

        chi2, p = compute_chi_square(
            length_stats[model]["short_data"],
            length_stats[model]["medium_data"],
            length_stats[model]["long_data"]
        )

        if chi2 is not None:
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            short_m = sum(d["match"] for d in length_stats[model]["short_data"])
            medium_m = sum(d["match"] for d in length_stats[model]["medium_data"])
            long_m = sum(d["match"] for d in length_stats[model]["long_data"])
            short_n = len(length_stats[model]["short_data"])
            medium_n = len(length_stats[model]["medium_data"])
            long_n = len(length_stats[model]["long_data"])

            print(f"\n{MODEL_DISPLAY[model]}: χ²={chi2:.2f}, p={p:.6f} ({sig})")
            print(f"  Short (2-3): {short_m}/{short_n} = {short_m/short_n*100:.1f}%")
            print(f"  Medium (4-5): {medium_m}/{medium_n} = {medium_m/medium_n*100:.1f}%")
            print(f"  Long (6+): {long_m}/{long_n} = {long_m/long_n*100:.1f}%")

    return length_stats


def analyze_language_x_length(results, qid_to_turns):
    """Analyze language × conversation length interaction."""
    print("\n" + "=" * 60)
    print("LANGUAGE × CONVERSATION LENGTH ANALYSIS (X→EN Fidelity)")
    print("=" * 60)

    # Build per-language data with turn counts
    lang_length_data = defaultdict(lambda: defaultdict(list))
    for model in MODELS:
        for lang in LANGUAGES:
            data = results["raw_language"][model]["codeswitching_reverse"].get(lang, [])
            for item in data:
                qid = item.get("question_id", "")
                base_qid = qid.replace(f"_{lang}_reverse", "").replace(f"_{lang}", "")
                turn_count = qid_to_turns.get(base_qid, 0)
                if turn_count > 0:
                    lang_length_data[model][lang].append({
                        "match": item["match"],
                        "turn_count": turn_count
                    })

    # Print table for each model
    for model in MODELS:
        print(f"\n### {MODEL_DISPLAY[model]}")
        print("| Language | Short (2-3) | Medium (4-5) | Long (6+) | Trend | χ² | p |")
        print("|----------|-------------|--------------|-----------|-------|----|----|")

        for lang in LANGUAGES:
            data = lang_length_data[model][lang]
            if not data:
                print(f"| {lang.upper()} | - | - | - | - | - | - |")
                continue

            short = [d for d in data if 2 <= d["turn_count"] <= 3]
            medium = [d for d in data if 4 <= d["turn_count"] <= 5]
            long_ = [d for d in data if d["turn_count"] >= 6]

            short_rate = sum(d["match"] for d in short) / len(short) * 100 if short else None
            medium_rate = sum(d["match"] for d in medium) / len(medium) * 100 if medium else None
            long_rate = sum(d["match"] for d in long_) / len(long_) * 100 if long_ else None

            # Correlation
            corr, p_corr = compute_correlation(data)
            trend = format_correlation(corr, p_corr)

            # Chi-square
            short_m = sum(d["match"] for d in short)
            medium_m = sum(d["match"] for d in medium)
            long_m = sum(d["match"] for d in long_)
            if len(short) > 0 and len(medium) > 0 and len(long_) > 0:
                contingency = [
                    [short_m, len(short) - short_m],
                    [medium_m, len(medium) - medium_m],
                    [long_m, len(long_) - long_m]
                ]
                try:
                    chi2, p_chi, _, _ = stats.chi2_contingency(contingency)
                    chi2_str = f"{chi2:.1f}"
                    p_str = f"{p_chi:.4f}" if p_chi >= 0.0001 else "<0.0001"
                except:
                    chi2_str = "-"
                    p_str = "-"
            else:
                chi2_str = "-"
                p_str = "-"

            short_str = f"{short_rate:.1f}% (n={len(short)})" if short_rate is not None else "-"
            medium_str = f"{medium_rate:.1f}% (n={len(medium)})" if medium_rate is not None else "-"
            long_str = f"{long_rate:.1f}% (n={len(long_)})" if long_rate is not None else "-"

            print(f"| {lang.upper()} | {short_str} | {medium_str} | {long_str} | {trend} | {chi2_str} | {p_str} |")

    return lang_length_data
