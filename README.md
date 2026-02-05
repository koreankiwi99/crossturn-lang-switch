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
git clone --recursive https://github.com/koreankiwi99/crossturn-lang-switch.git
cd crossturn-lang-switch

# If you already cloned without --recursive:
git submodule update --init

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
```

### Run Experiments

```bash
# Generate model responses (outputs to results/{model}/)
python src/scripts/run_experiment.py --model gpt-5 --data data/experiments/en_to_de.jsonl --workers 32

# Evaluate - Layer 1: Language Fidelity
python src/scripts/evaluation/language_fidelity.py --input results/gpt-5/responses_en_to_de_*.jsonl

# Evaluate - Layer 2: Task Accuracy
python src/scripts/evaluation/task_accuracy.py --input results/gpt-5/responses_en_to_de_*.jsonl
```

## Project Structure

```
├── data/
│   ├── multi-challenge/           # Original benchmark (git submodule)
│   ├── translations/              # GPT-4o verified translations
│   └── experiments/               # Final experiment datasets
│       ├── baseline_en.jsonl      # English baseline
│       ├── baseline_{lang}.jsonl  # Full translation baselines
│       ├── en_to_{lang}.jsonl     # EN→X conditions
│       └── {lang}_to_en.jsonl     # X→EN conditions
│
├── results/
│   ├── responses/                 # Raw model responses (5 models)
│   ├── layer1/                    # Language fidelity evaluations
│   ├── layer2/                    # Task accuracy evaluations
│   ├── cross-lingual/             # X→Y cross-lingual transfer results
│   └── sysprompt-ablation/        # System prompt ablation results
│
├── src/
│   ├── scripts/
│   │   ├── run_experiment.py      # Main experiment runner
│   │   ├── data-generation/       # Dataset generation scripts
│   │   └── evaluation/            # Evaluation scripts
│   │       ├── language_fidelity.py   # Layer 1: Language detection
│   │       └── task_accuracy.py       # Layer 2: GPT-4o judge
│   └── analysis/                  # Statistical analysis modules
│       ├── config.py              # Model lists, conditions, display names
│       └── paper_analysis.py      # Paper statistics generation
│
├── prompts/                       # Externalized evaluation prompts
├── notebooks/                     # Analysis notebooks
│   ├── full_results_tables.ipynb          # Complete results tables
│   ├── conversation_length_analysis.ipynb # Length effect analysis
│   ├── cross_lingual_analysis.ipynb       # X→Y transfer analysis
│   ├── sysprompt_ablation.ipynb           # System prompt ablation
│   └── consistency_analysis.ipynb         # Cross-run variance analysis
│
├── pdf/                           # Paper PDF
└── LICENSE                        # MIT License
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

## Paper

**Query-Following vs Context-Anchoring: How LLMs Handle Cross-Turn Language Switching**
Kyuhee Kim, Chengheng Li Chen, Anna Sotnikova
MME Workshop @ EACL 2026

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project extends the [MultiChallenge](https://arxiv.org/abs/2501.17399) benchmark.
