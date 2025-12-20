# Query-Following vs Context-Anchoring: How LLMs Handle Cross-Turn Language Switching

When multilingual users switch languages mid-conversation, how should LLMs respond? We extend MultiChallenge to evaluate cross-turn language switching across five frontier models and four languages.

## Key Findings

| Model | EN→X | X→EN | Behavior |
|-------|------|------|----------|
| GPT-5 | 98.6% | 95.1% | Query-following |
| Gemini 3 Pro | 98.3% | 73.8% | Mixed |
| Claude Opus 4.5 | 96.1% | 7.7% | Context-anchoring |
| DeepSeek-V3.1 | 88.3% | 51.9% | Mixed |
| Command R+ | 89.3% | 0.8% | Context-anchoring |

**Main finding**: All models follow the query language when switching into a foreign language (EN→X: 88–99%). But when switching back to English (X→EN), models diverge dramatically—GPT-5 follows the query (95%), while Claude and Command R+ continue in the context language (<8%).

Task accuracy remains stable across all conditions, indicating this is a behavioral choice, not a comprehension limitation.

## Experimental Conditions

| Condition | Description |
|-----------|-------------|
| Baseline (EN) | All turns in English (original) |
| Baseline (X) | All turns in X (translated) |
| EN→X | English context, final query in X |
| X→EN | X context, final query in English |

Where X ∈ {German, Chinese, Spanish, Arabic}

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Environment Variables

```bash
OPENAI_API_KEY=...      # For GPT models and evaluation judges
ANTHROPIC_API_KEY=...   # For Claude models
OPENROUTER_API_KEY=...  # For Gemini, DeepSeek, Command R+
```

### Generate Datasets

```bash
# Generate all experiment datasets from verified translations
python src/scripts/data-generation/generate_clean_datasets.py

# Generate distractor datasets (optional)
python src/scripts/data-generation/generate_distractor.py --lang de
```

### Run Experiments

```bash
# Generate model responses
python src/scripts/run_experiment.py --model gpt-5 --data data/experiments/en_to_de.jsonl --workers 32

# Evaluate - Layer 1: Language Fidelity
python src/scripts/evaluation/language_fidelity.py --input results/responses/gpt-5/responses_*.jsonl

# Evaluate - Layer 2: Task Accuracy
python src/scripts/evaluation/task_accuracy.py --input results/responses/gpt-5/responses_*.jsonl
```

## Project Structure

```
├── data/
│   ├── multi-challenge/           # Original benchmark (git submodule)
│   ├── translations/              # GPT-4o verified translations
│   └── experiments/               # Final experiment datasets
│       ├── baseline_en.jsonl      # English baseline
│       ├── baseline_{lang}.jsonl  # Full translation
│       ├── en_to_{lang}.jsonl     # EN→X condition
│       └── {lang}_to_en.jsonl     # X→EN condition
│
├── results/
│   ├── layer1/                    # Language fidelity evaluations
│   ├── layer2/                    # Task accuracy evaluations
│   └── responses/                 # Raw model responses
│
├── src/scripts/
│   ├── run_experiment.py          # Main experiment runner
│   ├── data-generation/           # Dataset generation scripts
│   └── evaluation/                # Evaluation scripts
│       ├── language_fidelity.py   # Layer 1: Language detection
│       └── task_accuracy.py       # Layer 2: GPT-4o judge
│
├── notebooks/                     # Analysis notebooks
│   ├── consistency_analysis.ipynb # Cross-run variance analysis
│   └── full_results_tables.ipynb  # Complete results tables
│
└── figures/                       # Paper figures
```

## Evaluation Metrics

**Layer 1 (Language Fidelity)**: Does the model respond in the user's query language?
- Judge: GPT-4o-mini
- Metric: Match rate with expected language

**Layer 2 (Task Accuracy)**: Does the model correctly complete the task?
- Judge: GPT-4o with MultiChallenge rubrics
- Metric: Pass rate on instance-level criteria

## Models

| Model | Provider | API |
|-------|----------|-----|
| GPT-5 | OpenAI | Direct |
| Gemini 3 Pro | Google | OpenRouter |
| Claude Opus 4.5 | Anthropic | Direct |
| DeepSeek-V3.1 | DeepSeek | OpenRouter |
| Command R+ | Cohere | OpenRouter |

## Languages

| Code | Language | Script |
|------|----------|--------|
| de | German | Latin |
| zh | Chinese (Simplified) | Han |
| es | Spanish | Latin |
| ar | Arabic | Arabic |

## Citation

```bibtex
@article{anonymous2025queryfollowing,
  title={Query-Following vs Context-Anchoring: How LLMs Handle Cross-Turn Language Switching},
  author={Anonymous},
  journal={ACL 2025 Submission},
  year={2025}
}
```

## License

This project extends the [MultiChallenge](https://arxiv.org/abs/2501.17399) benchmark.
