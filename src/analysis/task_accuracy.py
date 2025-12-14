"""Task accuracy (Layer 2) analysis."""

from .config import MODELS, MODEL_DISPLAY, LANGUAGES, CONDITION_DISPLAY
from .stats import compute_accuracy_stats


def analyze_task_accuracy(results):
    """Analyze task accuracy by model, condition, and language."""
    print("\n" + "=" * 60)
    print("LAYER 2: TASK ACCURACY ANALYSIS")
    print("=" * 60)

    # Summary table
    print("\n### Task Accuracy by Model and Condition (Avg across languages)")
    print("| Model | Baseline | EN→X | X→EN | Full Trans |")
    print("|-------|----------|------|------|------------|")

    for model in MODELS:
        row = [MODEL_DISPLAY[model]]
        for cond in ["baseline", "codeswitching", "codeswitching_reverse", "full_translation"]:
            if cond == "baseline":
                data = results["task_accuracy"][model][cond].get("en", [])
                rate, n = compute_accuracy_stats(data)
                row.append(f"{rate:.1f}%" if rate is not None else "-")
            else:
                rates = []
                for lang in LANGUAGES:
                    data = results["task_accuracy"][model][cond].get(lang, [])
                    rate, n = compute_accuracy_stats(data)
                    if rate is not None:
                        rates.append(rate)
                avg = sum(rates) / len(rates) if rates else None
                row.append(f"{avg:.1f}%" if avg is not None else "-")
        print("| " + " | ".join(row) + " |")

    # Detailed by language
    print("\n### Task Accuracy by Model, Condition, and Language")
    for model in MODELS:
        print(f"\n#### {MODEL_DISPLAY[model]}")
        print("| Condition | DE | ZH | ES | AR | Avg |")
        print("|-----------|----:|----:|----:|----:|----:|")

        for cond in ["baseline", "codeswitching", "codeswitching_reverse", "full_translation"]:
            row = [CONDITION_DISPLAY[cond]]
            rates = []
            if cond == "baseline":
                data = results["task_accuracy"][model][cond].get("en", [])
                rate, n = compute_accuracy_stats(data)
                rate_str = f"{rate:.1f}%" if rate is not None else "-"
                row.extend([rate_str] * 4)
                if rate:
                    rates = [rate] * 4
            else:
                for lang in LANGUAGES:
                    data = results["task_accuracy"][model][cond].get(lang, [])
                    rate, n = compute_accuracy_stats(data)
                    row.append(f"{rate:.1f}%" if rate is not None else "-")
                    if rate is not None:
                        rates.append(rate)
            avg = sum(rates) / len(rates) if rates else None
            row.append(f"{avg:.1f}%" if avg is not None else "-")
            print("| " + " | ".join(row) + " |")
