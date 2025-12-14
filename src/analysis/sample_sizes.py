"""Sample size summary."""

from .config import MODELS, MODEL_DISPLAY, CONDITIONS, CONDITION_DISPLAY, LANGUAGES


def analyze_sample_sizes(results):
    """Print sample sizes for each model and condition."""
    print("\n" + "=" * 60)
    print("SAMPLE SIZE SUMMARY")
    print("=" * 60)

    print("\n### Sample Sizes per Model and Condition")
    for model in MODELS:
        print(f"\n{MODEL_DISPLAY[model]}:")
        for cond in CONDITIONS:
            if cond == "baseline":
                data = results["task_accuracy"][model][cond].get("en", [])
                print(f"  {CONDITION_DISPLAY[cond]}: n={len(data)}")
            else:
                for lang in LANGUAGES:
                    acc_data = results["task_accuracy"][model][cond].get(lang, [])
                    fid_data = results["raw_language"][model][cond].get(lang, [])
                    if acc_data or fid_data:
                        print(f"  {CONDITION_DISPLAY[cond]} ({lang}): task n={len(acc_data)}, fidelity n={len(fid_data)}")
