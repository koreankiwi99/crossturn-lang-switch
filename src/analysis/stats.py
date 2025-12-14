"""Statistical analysis functions."""

import scipy.stats as stats


def compute_accuracy_stats(data):
    """Compute pass rate and count."""
    if not data:
        return None, 0
    passed = sum(1 for d in data if d["passed"])
    return passed / len(data) * 100, len(data)


def compute_fidelity_stats(data):
    """Compute fidelity rate and count."""
    if not data:
        return None, 0
    matches = sum(1 for d in data if d["match"])
    return matches / len(data) * 100, len(data)


def compute_correlation(data):
    """Compute point-biserial correlation between match and turn_count."""
    if len(data) < 10:
        return None, None
    matches = [1 if d["match"] else 0 for d in data]
    turns = [d["turn_count"] for d in data]
    corr, p_value = stats.pointbiserialr(matches, turns)
    return corr, p_value


def compute_chi_square(short_data, medium_data, long_data):
    """Compute chi-square test for conversation length effect."""
    if not short_data or not medium_data or not long_data:
        return None, None

    short_match = sum(1 for d in short_data if d["match"])
    medium_match = sum(1 for d in medium_data if d["match"])
    long_match = sum(1 for d in long_data if d["match"])

    contingency = [
        [short_match, len(short_data) - short_match],
        [medium_match, len(medium_data) - medium_match],
        [long_match, len(long_data) - long_match]
    ]

    try:
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        return chi2, p
    except:
        return None, None


def format_significance(p_value):
    """Format p-value with significance markers."""
    if p_value is None:
        return "-"
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"


def format_correlation(corr, p_value):
    """Format correlation with significance markers."""
    if corr is None:
        return "-"
    sig = format_significance(p_value)
    if sig == "ns":
        return "ns"
    return f"r={corr:.2f}{sig}"
