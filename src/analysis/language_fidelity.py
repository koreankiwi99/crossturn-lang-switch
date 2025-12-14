"""Language fidelity (Layer 1) analysis."""

from .config import MODELS, MODEL_DISPLAY, LANGUAGES, CONDITION_DISPLAY
from .stats import compute_fidelity_stats


def analyze_language_fidelity(results):
    """Analyze language fidelity by model, condition, and language."""
    print("\n" + "=" * 60)
    print("LAYER 1: LANGUAGE FIDELITY ANALYSIS")
    print("=" * 60)

    # Summary table
    print("\n### Language Fidelity Summary (X→EN is the critical condition)")
    print("| Model | EN→X | X→EN | Behavior |")
    print("|-------|------|------|----------|")

    model_fidelity = {}
    for model in MODELS:
        en_to_x_rates = []
        x_to_en_rates = []

        for lang in LANGUAGES:
            # EN→X (codeswitching)
            data = results["raw_language"][model]["codeswitching"].get(lang, [])
            rate, n = compute_fidelity_stats(data)
            if rate is not None:
                en_to_x_rates.append(rate)

            # X→EN (codeswitching_reverse)
            data = results["raw_language"][model]["codeswitching_reverse"].get(lang, [])
            rate, n = compute_fidelity_stats(data)
            if rate is not None:
                x_to_en_rates.append(rate)

        en_to_x_avg = sum(en_to_x_rates) / len(en_to_x_rates) if en_to_x_rates else None
        x_to_en_avg = sum(x_to_en_rates) / len(x_to_en_rates) if x_to_en_rates else None

        model_fidelity[model] = {"en_to_x": en_to_x_avg, "x_to_en": x_to_en_avg}

        # Classify behavior
        if x_to_en_avg is not None:
            if x_to_en_avg >= 85:
                behavior = "Query-following"
            elif x_to_en_avg <= 20:
                behavior = "Context-anchored"
            else:
                behavior = "Mixed"
        else:
            behavior = "-"

        en_to_x_str = f"{en_to_x_avg:.1f}%" if en_to_x_avg is not None else "-"
        x_to_en_str = f"**{x_to_en_avg:.1f}%**" if x_to_en_avg is not None else "-"
        print(f"| {MODEL_DISPLAY[model]} | {en_to_x_str} | {x_to_en_str} | {behavior} |")

    # Detailed by language
    print("\n### Language Fidelity by Model, Condition, and Language")
    for model in MODELS:
        print(f"\n#### {MODEL_DISPLAY[model]}")
        print("| Condition | Expected | DE | ZH | ES | AR | Avg |")
        print("|-----------|----------|----:|----:|----:|----:|----:|")

        for cond in ["codeswitching", "codeswitching_reverse", "full_translation"]:
            if cond == "codeswitching":
                expected = "X"
            elif cond == "codeswitching_reverse":
                expected = "EN"
            else:
                expected = "X"

            row = [CONDITION_DISPLAY[cond], expected]
            rates = []
            for lang in LANGUAGES:
                data = results["raw_language"][model][cond].get(lang, [])
                rate, n = compute_fidelity_stats(data)
                row.append(f"{rate:.1f}%" if rate is not None else "-")
                if rate is not None:
                    rates.append(rate)
            avg = sum(rates) / len(rates) if rates else None
            row.append(f"{avg:.1f}%" if avg is not None else "-")
            print("| " + " | ".join(row) + " |")

    return model_fidelity


def analyze_by_language(results):
    """Analyze X→EN fidelity by language."""
    print("\n" + "=" * 60)
    print("BY-LANGUAGE ANALYSIS (X→EN Fidelity)")
    print("=" * 60)

    print("\n### X→EN Fidelity by Language")
    print("| Model | DE | ZH | ES | AR | Avg |")
    print("|-------|----:|----:|----:|----:|----:|")

    for model in MODELS:
        row = [MODEL_DISPLAY[model]]
        rates = []
        for lang in LANGUAGES:
            data = results["raw_language"][model]["codeswitching_reverse"].get(lang, [])
            rate, n = compute_fidelity_stats(data)
            row.append(f"{rate:.1f}%" if rate is not None else "-")
            if rate is not None:
                rates.append(rate)
        avg = sum(rates) / len(rates) if rates else None
        row.append(f"{avg:.1f}%" if avg is not None else "-")
        print("| " + " | ".join(row) + " |")
