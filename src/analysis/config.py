"""Configuration constants for analysis."""

from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"

MODELS = ["gpt-5", "gemini-3-pro", "claude-opus-4.5", "deepseek-v3.1", "command-r-plus"]

MODEL_DISPLAY = {
    "gpt-5": "GPT-5",
    "gemini-3-pro": "Gemini 3 Pro",
    "claude-opus-4.5": "Claude Opus 4.5",
    "deepseek-v3.1": "DeepSeek-V3.1",
    "command-r-plus": "Command R+"
}

CONDITIONS = ["baseline", "codeswitching", "codeswitching_reverse", "full_translation", "distractor", "distractor_multi"]

CONDITION_DISPLAY = {
    "baseline": "Baseline (EN)",
    "codeswitching": "EN→X",
    "codeswitching_reverse": "X→EN",
    "full_translation": "Full Translation",
    "distractor": "Distractor",
    "distractor_multi": "Distractor Multi"
}

LANGUAGES = ["de", "zh", "es", "ar"]

LANG_DISPLAY = {
    "de": "German",
    "zh": "Chinese",
    "es": "Spanish",
    "ar": "Arabic"
}
