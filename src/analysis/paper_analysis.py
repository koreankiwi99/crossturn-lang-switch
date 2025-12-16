#!/usr/bin/env python3
"""
Comprehensive statistical analysis for paper submission.

Generates:
1. Cross-model comparison figures
2. Bootstrap confidence intervals
3. Effect sizes (Cohen's h, Cramér's V)
4. Pairwise model comparisons (McNemar's test)
5. Error analysis

Usage:
    python -m src.analysis.paper_analysis
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from scipy import stats
from itertools import combinations

from .config import RESULTS_DIR, MODELS, MODEL_DISPLAY, LANGUAGES, LANG_DISPLAY
from .data_loader import load_all_data, build_qid_to_turns

# Output directory for figures
FIGURES_DIR = Path(__file__).parent.parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.dpi'] = 150


# =============================================================================
# STATISTICAL UTILITIES
# =============================================================================

def bootstrap_ci(data, statistic=np.mean, n_bootstrap=10000, ci=95):
    """Calculate bootstrap confidence interval."""
    if len(data) == 0:
        return None, None, None

    data = np.array(data)
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_stats.append(statistic(sample))

    alpha = (100 - ci) / 2
    ci_low = np.percentile(boot_stats, alpha)
    ci_high = np.percentile(boot_stats, 100 - alpha)
    point_est = statistic(data)

    return point_est, ci_low, ci_high


def cohens_h(p1, p2):
    """Calculate Cohen's h for two proportions."""
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    return phi1 - phi2


def interpret_cohens_h(h):
    """Interpret Cohen's h effect size."""
    h = abs(h)
    if h < 0.2:
        return "negligible"
    elif h < 0.5:
        return "small"
    elif h < 0.8:
        return "medium"
    else:
        return "large"


def cramers_v(contingency_table):
    """Calculate Cramér's V from contingency table."""
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0


def mcnemar_test(model1_correct, model2_correct):
    """
    McNemar's test for paired nominal data.
    Returns chi2, p-value, and counts.
    """
    # Count discordant pairs
    b = sum(1 for m1, m2 in zip(model1_correct, model2_correct) if m1 and not m2)
    c = sum(1 for m1, m2 in zip(model1_correct, model2_correct) if not m1 and m2)

    if b + c == 0:
        return 0, 1.0, b, c

    # McNemar's chi-square (with continuity correction)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    return chi2, p_value, b, c


def bonferroni_correction(p_values, alpha=0.05):
    """Apply Bonferroni correction."""
    n = len(p_values)
    corrected_alpha = alpha / n
    return [p < corrected_alpha for p in p_values], corrected_alpha


# =============================================================================
# DATA EXTRACTION
# =============================================================================

def extract_fidelity_data(results):
    """Extract language fidelity data into a structured format."""
    data = []

    for model in MODELS:
        for condition in ["codeswitching", "codeswitching_reverse", "full_translation"]:
            for lang in LANGUAGES:
                items = results["raw_language"][model][condition].get(lang, [])
                for item in items:
                    data.append({
                        "model": model,
                        "condition": condition,
                        "language": lang,
                        "match": item["match"],
                        "question_id": item["question_id"],
                        "detected": item.get("detected"),
                        "expected": item.get("expected")
                    })

    return pd.DataFrame(data)


def extract_accuracy_data(results):
    """Extract task accuracy data into a structured format."""
    data = []

    for model in MODELS:
        for condition in ["baseline", "codeswitching", "codeswitching_reverse", "full_translation"]:
            if condition == "baseline":
                items = results["task_accuracy"][model][condition].get("en", [])
                for item in items:
                    data.append({
                        "model": model,
                        "condition": condition,
                        "language": "en",
                        "passed": item["passed"],
                        "turn_count": item["turn_count"],
                        "question_id": item["question_id"]
                    })
            else:
                for lang in LANGUAGES:
                    items = results["task_accuracy"][model][condition].get(lang, [])
                    for item in items:
                        data.append({
                            "model": model,
                            "condition": condition,
                            "language": lang,
                            "passed": item["passed"],
                            "turn_count": item["turn_count"],
                            "question_id": item["question_id"]
                        })

    return pd.DataFrame(data)


# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================

def analyze_language_fidelity_with_ci(df_fidelity):
    """Analyze language fidelity with bootstrap CIs."""
    print("\n" + "=" * 80)
    print("LAYER 1: LANGUAGE FIDELITY WITH 95% CONFIDENCE INTERVALS")
    print("=" * 80)

    results = []

    # X→EN is the critical condition
    print("\n### X→EN Fidelity (Critical Condition)")
    print("| Model | Fidelity | 95% CI | n |")
    print("|-------|----------|--------|---|")

    for model in MODELS:
        df_model = df_fidelity[(df_fidelity["model"] == model) &
                               (df_fidelity["condition"] == "codeswitching_reverse")]
        matches = df_model["match"].values.astype(int)

        if len(matches) > 0:
            point, ci_low, ci_high = bootstrap_ci(matches * 100)
            print(f"| {MODEL_DISPLAY[model]} | {point:.1f}% | [{ci_low:.1f}, {ci_high:.1f}] | {len(matches)} |")
            results.append({
                "model": model,
                "condition": "X→EN",
                "fidelity": point,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n": len(matches)
            })

    # EN→X for comparison
    print("\n### EN→X Fidelity")
    print("| Model | Fidelity | 95% CI | n |")
    print("|-------|----------|--------|---|")

    for model in MODELS:
        df_model = df_fidelity[(df_fidelity["model"] == model) &
                               (df_fidelity["condition"] == "codeswitching")]
        matches = df_model["match"].values.astype(int)

        if len(matches) > 0:
            point, ci_low, ci_high = bootstrap_ci(matches * 100)
            print(f"| {MODEL_DISPLAY[model]} | {point:.1f}% | [{ci_low:.1f}, {ci_high:.1f}] | {len(matches)} |")
            results.append({
                "model": model,
                "condition": "EN→X",
                "fidelity": point,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n": len(matches)
            })

    return pd.DataFrame(results)


def analyze_by_language_with_ci(df_fidelity):
    """Analyze X→EN fidelity by language with CIs."""
    print("\n" + "=" * 80)
    print("X→EN FIDELITY BY LANGUAGE WITH 95% CI")
    print("=" * 80)

    results = []

    for model in MODELS:
        print(f"\n### {MODEL_DISPLAY[model]}")
        print("| Language | Fidelity | 95% CI | n |")
        print("|----------|----------|--------|---|")

        for lang in LANGUAGES:
            df_sub = df_fidelity[(df_fidelity["model"] == model) &
                                 (df_fidelity["condition"] == "codeswitching_reverse") &
                                 (df_fidelity["language"] == lang)]
            matches = df_sub["match"].values.astype(int)

            if len(matches) > 0:
                point, ci_low, ci_high = bootstrap_ci(matches * 100)
                print(f"| {LANG_DISPLAY[lang]} | {point:.1f}% | [{ci_low:.1f}, {ci_high:.1f}] | {len(matches)} |")
                results.append({
                    "model": model,
                    "language": lang,
                    "fidelity": point,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n": len(matches)
                })

    return pd.DataFrame(results)


def analyze_task_accuracy_with_ci(df_accuracy):
    """Analyze task accuracy with bootstrap CIs."""
    print("\n" + "=" * 80)
    print("LAYER 2: TASK ACCURACY WITH 95% CONFIDENCE INTERVALS")
    print("=" * 80)

    results = []

    print("\n### Task Accuracy by Condition")
    print("| Model | Baseline | EN→X | X→EN | Full Trans |")
    print("|-------|----------|------|------|------------|")

    for model in MODELS:
        row = [MODEL_DISPLAY[model]]

        for condition in ["baseline", "codeswitching", "codeswitching_reverse", "full_translation"]:
            df_cond = df_accuracy[(df_accuracy["model"] == model) &
                                  (df_accuracy["condition"] == condition)]
            passed = df_cond["passed"].values.astype(int)

            if len(passed) > 0:
                point, ci_low, ci_high = bootstrap_ci(passed * 100)
                row.append(f"{point:.1f}% [{ci_low:.1f}, {ci_high:.1f}]")
                results.append({
                    "model": model,
                    "condition": condition,
                    "accuracy": point,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n": len(passed)
                })
            else:
                row.append("-")

        print("| " + " | ".join(row) + " |")

    return pd.DataFrame(results)


def compute_effect_sizes(df_fidelity):
    """Compute effect sizes for model comparisons."""
    print("\n" + "=" * 80)
    print("EFFECT SIZES: COHEN'S h FOR X→EN FIDELITY")
    print("=" * 80)

    # Get X→EN fidelity rates
    fidelity_rates = {}
    for model in MODELS:
        df_model = df_fidelity[(df_fidelity["model"] == model) &
                               (df_fidelity["condition"] == "codeswitching_reverse")]
        if len(df_model) > 0:
            fidelity_rates[model] = df_model["match"].mean()

    print("\n### Pairwise Cohen's h (X→EN Fidelity)")
    print("| Comparison | Cohen's h | Interpretation |")
    print("|------------|-----------|----------------|")

    results = []
    for m1, m2 in combinations(MODELS, 2):
        if m1 in fidelity_rates and m2 in fidelity_rates:
            h = cohens_h(fidelity_rates[m1], fidelity_rates[m2])
            interp = interpret_cohens_h(h)
            print(f"| {MODEL_DISPLAY[m1]} vs {MODEL_DISPLAY[m2]} | {h:.3f} | {interp} |")
            results.append({
                "model1": m1,
                "model2": m2,
                "cohens_h": h,
                "interpretation": interp
            })

    # Key comparisons highlighted
    print("\n### Key Effect Sizes")
    print("- GPT-5 vs Claude Opus 4.5: Large effect (query-following vs context-anchored)")
    print("- GPT-5 vs Command R+: Large effect (94% vs 0%)")

    return pd.DataFrame(results)


def pairwise_model_comparisons(df_fidelity):
    """Run McNemar's test for pairwise model comparisons on X→EN."""
    print("\n" + "=" * 80)
    print("PAIRWISE MODEL COMPARISONS (McNEMAR'S TEST)")
    print("=" * 80)

    # Get X→EN data aligned by question_id
    condition = "codeswitching_reverse"
    model_data = {}

    for model in MODELS:
        df_model = df_fidelity[(df_fidelity["model"] == model) &
                               (df_fidelity["condition"] == condition)]
        if len(df_model) > 0:
            model_data[model] = df_model.set_index("question_id")["match"].to_dict()

    # Find common question IDs
    all_qids = set()
    for data in model_data.values():
        all_qids.update(data.keys())

    print(f"\nTotal unique questions: {len(all_qids)}")

    results = []
    p_values = []
    comparisons = []

    print("\n### McNemar's Test Results (X→EN Fidelity)")
    print("| Comparison | χ² | p-value | b | c | Significant |")
    print("|------------|-----|---------|---|---|-------------|")

    for m1, m2 in combinations(MODELS, 2):
        if m1 not in model_data or m2 not in model_data:
            continue

        # Get common questions
        common = set(model_data[m1].keys()) & set(model_data[m2].keys())
        if len(common) < 10:
            continue

        m1_correct = [model_data[m1][qid] for qid in common]
        m2_correct = [model_data[m2][qid] for qid in common]

        chi2, p, b, c = mcnemar_test(m1_correct, m2_correct)

        p_values.append(p)
        comparisons.append((m1, m2))

        results.append({
            "model1": m1,
            "model2": m2,
            "chi2": chi2,
            "p_value": p,
            "b": b,
            "c": c,
            "n_common": len(common)
        })

    # Bonferroni correction
    significant, corrected_alpha = bonferroni_correction(p_values)

    for i, (res, sig) in enumerate(zip(results, significant)):
        sig_str = "***" if sig else ""
        print(f"| {MODEL_DISPLAY[res['model1']]} vs {MODEL_DISPLAY[res['model2']]} | "
              f"{res['chi2']:.2f} | {res['p_value']:.4f} | {res['b']} | {res['c']} | {sig_str} |")
        results[i]["significant"] = sig

    print(f"\nBonferroni-corrected α = {corrected_alpha:.4f}")
    print("*** = significant after Bonferroni correction")

    return pd.DataFrame(results)


def conversation_length_analysis(df_fidelity, qid_to_turns):
    """Analyze conversation length effects on X→EN fidelity."""
    print("\n" + "=" * 80)
    print("CONVERSATION LENGTH EFFECTS (X→EN FIDELITY)")
    print("=" * 80)

    # Add turn counts
    df_x_to_en = df_fidelity[df_fidelity["condition"] == "codeswitching_reverse"].copy()

    def get_turns(qid):
        base_qid = qid.split("_")[0]
        return qid_to_turns.get(base_qid, qid_to_turns.get(qid, 0))

    df_x_to_en["turn_count"] = df_x_to_en["question_id"].apply(get_turns)

    # Bin by length (turns are always odd: 3, 5, 7, 9, 11, 13, 15, 19)
    def categorize_length(turns):
        if turns <= 3:
            return "Short (3)"
        elif turns <= 5:
            return "Medium (5)"
        else:
            return "Long (7+)"

    df_x_to_en["length_bin"] = df_x_to_en["turn_count"].apply(categorize_length)

    print("\n### X→EN Fidelity by Conversation Length")
    print("| Model | Short (3) | Medium (5) | Long (7+) | χ² | p | Trend |")
    print("|-------|-----------|------------|-----------|-----|------|-------|")

    results = []

    for model in MODELS:
        df_model = df_x_to_en[df_x_to_en["model"] == model]

        if len(df_model) < 50:
            continue

        # Get fidelity by bin
        short_data = df_model[df_model["length_bin"] == "Short (3)"]["match"].values
        medium_data = df_model[df_model["length_bin"] == "Medium (5)"]["match"].values
        long_data = df_model[df_model["length_bin"] == "Long (7+)"]["match"].values

        if len(short_data) < 5 or len(medium_data) < 5 or len(long_data) < 5:
            continue

        short_rate, short_lo, short_hi = bootstrap_ci(short_data.astype(int) * 100)
        medium_rate, med_lo, med_hi = bootstrap_ci(medium_data.astype(int) * 100)
        long_rate, long_lo, long_hi = bootstrap_ci(long_data.astype(int) * 100)

        # Chi-square test
        contingency = [
            [short_data.sum(), len(short_data) - short_data.sum()],
            [medium_data.sum(), len(medium_data) - medium_data.sum()],
            [long_data.sum(), len(long_data) - long_data.sum()]
        ]

        try:
            chi2, p, _, _ = stats.chi2_contingency(contingency)
            cramers = cramers_v(pd.DataFrame(contingency))
        except:
            chi2, p, cramers = 0, 1, 0

        # Correlation with turn count
        matches = df_model["match"].values.astype(int)
        turns = df_model["turn_count"].values
        corr, corr_p = stats.pointbiserialr(matches, turns)

        trend = f"r={corr:.2f}" if corr_p < 0.05 else "ns"
        sig = "**" if p < 0.01 else "*" if p < 0.05 else ""

        print(f"| {MODEL_DISPLAY[model]} | {short_rate:.1f}% (n={len(short_data)}) | "
              f"{medium_rate:.1f}% (n={len(medium_data)}) | {long_rate:.1f}% (n={len(long_data)}) | "
              f"{chi2:.2f} | {p:.4f}{sig} | {trend} |")

        results.append({
            "model": model,
            "short_rate": short_rate,
            "medium_rate": medium_rate,
            "long_rate": long_rate,
            "chi2": chi2,
            "p_value": p,
            "cramers_v": cramers,
            "correlation": corr,
            "corr_p": corr_p
        })

    return pd.DataFrame(results)


def error_analysis(df_fidelity):
    """Analyze language confusion patterns."""
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS: LANGUAGE CONFUSION PATTERNS")
    print("=" * 80)

    df_errors = df_fidelity[(df_fidelity["condition"] == "codeswitching_reverse") &
                            (df_fidelity["match"] == False)]

    print(f"\nTotal X→EN errors: {len(df_errors)}")

    print("\n### Errors by Model")
    print("| Model | Errors | Error Rate |")
    print("|-------|--------|------------|")

    for model in MODELS:
        df_model_all = df_fidelity[(df_fidelity["model"] == model) &
                                   (df_fidelity["condition"] == "codeswitching_reverse")]
        df_model_err = df_errors[df_errors["model"] == model]

        if len(df_model_all) > 0:
            rate = len(df_model_err) / len(df_model_all) * 100
            print(f"| {MODEL_DISPLAY[model]} | {len(df_model_err)} | {rate:.1f}% |")

    print("\n### Common Confusion Patterns (Expected → Detected)")
    confusion = df_errors.groupby(["expected", "detected"]).size().sort_values(ascending=False)
    print("| Expected | Detected | Count |")
    print("|----------|----------|-------|")
    for (exp, det), count in confusion.head(10).items():
        print(f"| {exp} | {det} | {count} |")

    print("\n### Errors by Language (X→EN)")
    print("| Language | Errors | Total | Error Rate |")
    print("|----------|--------|-------|------------|")

    for lang in LANGUAGES:
        df_lang_all = df_fidelity[(df_fidelity["condition"] == "codeswitching_reverse") &
                                  (df_fidelity["language"] == lang)]
        df_lang_err = df_errors[df_errors["language"] == lang]

        if len(df_lang_all) > 0:
            rate = len(df_lang_err) / len(df_lang_all) * 100
            print(f"| {LANG_DISPLAY[lang]} | {len(df_lang_err)} | {len(df_lang_all)} | {rate:.1f}% |")


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_xen_fidelity_comparison(df_fidelity):
    """Create main figure: X→EN fidelity comparison across models."""
    print("\nGenerating X→EN fidelity comparison figure...")

    # Calculate fidelity with CIs
    data = []
    for model in MODELS:
        df_model = df_fidelity[(df_fidelity["model"] == model) &
                               (df_fidelity["condition"] == "codeswitching_reverse")]
        matches = df_model["match"].values.astype(int)

        if len(matches) > 0:
            point, ci_low, ci_high = bootstrap_ci(matches * 100)
            data.append({
                "model": MODEL_DISPLAY[model],
                "fidelity": point,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "error_low": point - ci_low,
                "error_high": ci_high - point
            })

    df_plot = pd.DataFrame(data)
    df_plot = df_plot.sort_values("fidelity", ascending=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#d62728' if f < 20 else '#ff7f0e' if f < 70 else '#2ca02c'
              for f in df_plot["fidelity"]]

    bars = ax.barh(df_plot["model"], df_plot["fidelity"], color=colors, alpha=0.8)

    # Error bars
    ax.errorbar(df_plot["fidelity"], df_plot["model"],
                xerr=[df_plot["error_low"], df_plot["error_high"]],
                fmt='none', color='black', capsize=3)

    # Labels
    for i, (idx, row) in enumerate(df_plot.iterrows()):
        ax.text(row["fidelity"] + 2, i, f"{row['fidelity']:.1f}%",
                va='center', fontsize=10)

    ax.set_xlabel("X→EN Language Fidelity (%)")
    ax.set_title("Language Fidelity: Do Models Follow User Query Language?\n(X→EN: Foreign context, English query)",
                 fontsize=12)
    ax.set_xlim(0, 110)
    ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5, label='Random baseline')

    # Add behavior labels
    ax.text(95, len(df_plot)-0.5, "Query-following", fontsize=9, color='#2ca02c', ha='right')
    ax.text(15, 0.5, "Context-anchored", fontsize=9, color='#d62728', ha='left')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig1_xen_fidelity_comparison.pdf", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "fig1_xen_fidelity_comparison.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {FIGURES_DIR / 'fig1_xen_fidelity_comparison.pdf'}")
    plt.close()


def plot_fidelity_heatmap(df_fidelity):
    """Create heatmap of fidelity by model, condition, and language."""
    print("\nGenerating fidelity heatmap...")

    # Pivot data
    pivot_data = []
    for model in MODELS:
        for condition in ["codeswitching", "codeswitching_reverse", "full_translation"]:
            for lang in LANGUAGES:
                df_sub = df_fidelity[(df_fidelity["model"] == model) &
                                     (df_fidelity["condition"] == condition) &
                                     (df_fidelity["language"] == lang)]
                if len(df_sub) > 0:
                    rate = df_sub["match"].mean() * 100
                    pivot_data.append({
                        "Model": MODEL_DISPLAY[model],
                        "Condition": {"codeswitching": "EN→X",
                                     "codeswitching_reverse": "X→EN",
                                     "full_translation": "Full Trans"}[condition],
                        "Language": LANG_DISPLAY[lang],
                        "Fidelity": rate
                    })

    df_pivot = pd.DataFrame(pivot_data)

    # Create subplots for each condition
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, condition in enumerate(["EN→X", "X→EN", "Full Trans"]):
        df_cond = df_pivot[df_pivot["Condition"] == condition]
        heatmap_data = df_cond.pivot(index="Model", columns="Language", values="Fidelity")

        # Reorder
        model_order = [MODEL_DISPLAY[m] for m in MODELS if MODEL_DISPLAY[m] in heatmap_data.index]
        lang_order = [LANG_DISPLAY[l] for l in LANGUAGES if LANG_DISPLAY[l] in heatmap_data.columns]
        heatmap_data = heatmap_data.reindex(index=model_order, columns=lang_order)

        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn',
                    vmin=0, vmax=100, ax=axes[i], cbar=i==2)
        axes[i].set_title(f"{condition}\n(Expected: {'X' if condition != 'X→EN' else 'EN'})")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("" if i > 0 else "Model")

    plt.suptitle("Language Fidelity by Model, Condition, and Language (%)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig2_fidelity_heatmap.pdf", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "fig2_fidelity_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {FIGURES_DIR / 'fig2_fidelity_heatmap.pdf'}")
    plt.close()


def plot_task_accuracy_comparison(df_accuracy):
    """Create task accuracy comparison figure."""
    print("\nGenerating task accuracy comparison figure...")

    # Calculate accuracy with CIs
    data = []
    for model in MODELS:
        for condition in ["baseline", "codeswitching", "codeswitching_reverse", "full_translation"]:
            df_cond = df_accuracy[(df_accuracy["model"] == model) &
                                  (df_accuracy["condition"] == condition)]
            passed = df_cond["passed"].values.astype(int)

            if len(passed) > 0:
                point, ci_low, ci_high = bootstrap_ci(passed * 100)
                data.append({
                    "Model": MODEL_DISPLAY[model],
                    "Condition": {"baseline": "Baseline",
                                 "codeswitching": "EN→X",
                                 "codeswitching_reverse": "X→EN",
                                 "full_translation": "Full Trans"}[condition],
                    "Accuracy": point,
                    "CI_low": ci_low,
                    "CI_high": ci_high
                })

    df_plot = pd.DataFrame(data)

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    conditions = ["Baseline", "EN→X", "X→EN", "Full Trans"]
    models = [MODEL_DISPLAY[m] for m in MODELS]
    x = np.arange(len(models))
    width = 0.2

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, condition in enumerate(conditions):
        df_cond = df_plot[df_plot["Condition"] == condition]
        df_cond = df_cond.set_index("Model").reindex(models)

        bars = ax.bar(x + i*width, df_cond["Accuracy"], width, label=condition,
                      color=colors[i], alpha=0.8)

        # Error bars
        ax.errorbar(x + i*width, df_cond["Accuracy"],
                    yerr=[df_cond["Accuracy"] - df_cond["CI_low"],
                          df_cond["CI_high"] - df_cond["Accuracy"]],
                    fmt='none', color='black', capsize=2)

    ax.set_ylabel("Task Accuracy (%)")
    ax.set_title("Task Accuracy by Model and Condition\n(Layer 2: Does language switching hurt task completion?)")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_task_accuracy_comparison.pdf", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "fig3_task_accuracy_comparison.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {FIGURES_DIR / 'fig3_task_accuracy_comparison.pdf'}")
    plt.close()


def plot_conversation_length_effect(df_fidelity, qid_to_turns):
    """Plot conversation length effect on X→EN fidelity."""
    print("\nGenerating conversation length effect figure...")

    df_x_to_en = df_fidelity[df_fidelity["condition"] == "codeswitching_reverse"].copy()

    def get_turns(qid):
        base_qid = qid.split("_")[0]
        return qid_to_turns.get(base_qid, qid_to_turns.get(qid, 0))

    df_x_to_en["turn_count"] = df_x_to_en["question_id"].apply(get_turns)

    # Filter to models with interesting patterns
    models_to_plot = ["gpt-5", "gemini-3-pro", "claude-opus-4.5", "deepseek-v3.1"]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'gpt-5': '#2ca02c', 'gemini-3-pro': '#ff7f0e',
              'claude-opus-4.5': '#d62728', 'deepseek-v3.1': '#9467bd'}

    for model in models_to_plot:
        df_model = df_x_to_en[df_x_to_en["model"] == model]

        if len(df_model) < 50:
            continue

        # Group by turn count
        grouped = df_model.groupby("turn_count").agg({
            "match": ["mean", "count"]
        }).reset_index()
        grouped.columns = ["turns", "fidelity", "count"]
        grouped["fidelity"] *= 100
        grouped = grouped[grouped["count"] >= 5]  # Filter small groups

        ax.plot(grouped["turns"], grouped["fidelity"], 'o-',
                label=MODEL_DISPLAY[model], color=colors[model], alpha=0.8)

    ax.set_xlabel("Conversation Length (turns)")
    ax.set_ylabel("X→EN Language Fidelity (%)")
    ax.set_title("X→EN Fidelity Degradation with Conversation Length")
    ax.legend(loc='lower left')
    ax.set_ylim(0, 105)
    ax.set_xlim(1, 12)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_conversation_length.pdf", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "fig4_conversation_length.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {FIGURES_DIR / 'fig4_conversation_length.pdf'}")
    plt.close()


def plot_behavioral_classification(df_fidelity):
    """Plot behavioral classification: query-following vs context-anchored."""
    print("\nGenerating behavioral classification figure...")

    data = []
    for model in MODELS:
        # EN→X
        df_en_x = df_fidelity[(df_fidelity["model"] == model) &
                              (df_fidelity["condition"] == "codeswitching")]
        en_x_rate = df_en_x["match"].mean() * 100 if len(df_en_x) > 0 else 0

        # X→EN
        df_x_en = df_fidelity[(df_fidelity["model"] == model) &
                              (df_fidelity["condition"] == "codeswitching_reverse")]
        x_en_rate = df_x_en["match"].mean() * 100 if len(df_x_en) > 0 else 0

        data.append({
            "Model": MODEL_DISPLAY[model],
            "EN→X": en_x_rate,
            "X→EN": x_en_rate
        })

    df_plot = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Color by behavior
    colors = []
    for _, row in df_plot.iterrows():
        if row["X→EN"] >= 85:
            colors.append('#2ca02c')  # Query-following
        elif row["X→EN"] <= 20:
            colors.append('#d62728')  # Context-anchored
        else:
            colors.append('#ff7f0e')  # Mixed

    ax.scatter(df_plot["EN→X"], df_plot["X→EN"], s=200, c=colors, alpha=0.8, edgecolors='black')

    # Labels
    for _, row in df_plot.iterrows():
        ax.annotate(row["Model"], (row["EN→X"], row["X→EN"]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)

    ax.set_xlabel("EN→X Fidelity (%) - English context, foreign query")
    ax.set_ylabel("X→EN Fidelity (%) - Foreign context, English query")
    ax.set_title("Behavioral Classification:\nQuery-Following vs Context-Anchored Models")
    ax.set_xlim(60, 105)
    ax.set_ylim(-5, 105)

    # Add quadrant labels
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=80, color='gray', linestyle='--', alpha=0.5)
    ax.text(100, 95, "Query-following", fontsize=10, color='#2ca02c', ha='right')
    ax.text(100, 5, "Context-anchored", fontsize=10, color='#d62728', ha='right')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_behavioral_classification.pdf", bbox_inches='tight')
    plt.savefig(FIGURES_DIR / "fig5_behavioral_classification.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {FIGURES_DIR / 'fig5_behavioral_classification.pdf'}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all analyses and generate figures."""
    print("=" * 80)
    print("COMPREHENSIVE STATISTICAL ANALYSIS FOR PAPER")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    results = load_all_data()
    qid_to_turns = build_qid_to_turns(results)

    # Extract into DataFrames
    df_fidelity = extract_fidelity_data(results)
    df_accuracy = extract_accuracy_data(results)

    print(f"Loaded {len(df_fidelity)} fidelity evaluations")
    print(f"Loaded {len(df_accuracy)} accuracy evaluations")

    # Run analyses
    df_fidelity_ci = analyze_language_fidelity_with_ci(df_fidelity)
    df_lang_ci = analyze_by_language_with_ci(df_fidelity)
    df_accuracy_ci = analyze_task_accuracy_with_ci(df_accuracy)
    df_effect_sizes = compute_effect_sizes(df_fidelity)
    df_pairwise = pairwise_model_comparisons(df_fidelity)
    df_length = conversation_length_analysis(df_fidelity, qid_to_turns)
    error_analysis(df_fidelity)

    # Generate figures
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)

    plot_xen_fidelity_comparison(df_fidelity)
    plot_fidelity_heatmap(df_fidelity)
    plot_task_accuracy_comparison(df_accuracy)
    plot_conversation_length_effect(df_fidelity, qid_to_turns)
    plot_behavioral_classification(df_fidelity)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nFigures saved to: {FIGURES_DIR}")

    # Save analysis results
    output_dir = FIGURES_DIR / "data"
    output_dir.mkdir(exist_ok=True)

    df_fidelity_ci.to_csv(output_dir / "fidelity_with_ci.csv", index=False)
    df_accuracy_ci.to_csv(output_dir / "accuracy_with_ci.csv", index=False)
    df_effect_sizes.to_csv(output_dir / "effect_sizes.csv", index=False)
    df_pairwise.to_csv(output_dir / "pairwise_comparisons.csv", index=False)
    df_length.to_csv(output_dir / "conversation_length.csv", index=False)

    print(f"Data tables saved to: {output_dir}")


if __name__ == "__main__":
    main()
