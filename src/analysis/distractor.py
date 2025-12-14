"""Distractor condition analysis."""

from .config import MODELS, MODEL_DISPLAY, LANGUAGES
from .stats import compute_fidelity_stats


def analyze_distractor(results):
    """Analyze distractor conditions."""
    print("\n" + "=" * 60)
    print("DISTRACTOR CONDITION ANALYSIS")
    print("=" * 60)

    print("\n### Language Fidelity: Distractor vs Base EN→X")
    print("| Model | EN→X | Distractor | Distractor Multi |")
    print("|-------|------|------------|------------------|")

    for model in MODELS:
        en_to_x_rates = []
        distractor_rates = []
        distractor_multi_rates = []

        for lang in LANGUAGES:
            # EN→X
            data = results["raw_language"][model]["codeswitching"].get(lang, [])
            rate, _ = compute_fidelity_stats(data)
            if rate is not None:
                en_to_x_rates.append(rate)

            # Distractor
            data = results["raw_language"][model]["distractor"].get(lang, [])
            rate, _ = compute_fidelity_stats(data)
            if rate is not None:
                distractor_rates.append(rate)

            # Distractor Multi
            data = results["raw_language"][model]["distractor_multi"].get(lang, [])
            rate, _ = compute_fidelity_stats(data)
            if rate is not None:
                distractor_multi_rates.append(rate)

        en_to_x_avg = sum(en_to_x_rates) / len(en_to_x_rates) if en_to_x_rates else None
        dist_avg = sum(distractor_rates) / len(distractor_rates) if distractor_rates else None
        dist_multi_avg = sum(distractor_multi_rates) / len(distractor_multi_rates) if distractor_multi_rates else None

        en_to_x_str = f"{en_to_x_avg:.1f}%" if en_to_x_avg is not None else "-"
        dist_str = f"{dist_avg:.1f}%" if dist_avg is not None else "-"
        dist_multi_str = f"{dist_multi_avg:.1f}%" if dist_multi_avg is not None else "-"

        print(f"| {MODEL_DISPLAY[model]} | {en_to_x_str} | {dist_str} | {dist_multi_str} |")
