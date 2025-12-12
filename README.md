# Cross-Turn Language Switching in LLMs

Evaluating whether language models maintain task performance when users switch languages mid-conversation.

**Benchmark Extension**: [MultiChallenge](https://scale.com/leaderboard/multichallenge)

## Introduction

Large language models increasingly serve multilingual users worldwide. These users often switch between languages naturally within a single conversation—starting a query in English, then continuing in their native language, or vice versa. This cross-turn language switching reflects how multilingual speakers actually communicate: fluidly moving between languages based on comfort, context, or expression.

Existing benchmarks evaluate multilingual and multi-turn capabilities separately:
- **Multilingual benchmarks** (MMMLU, Multi-IF) test cross-lingual understanding in single-turn setups
- **Code-switching benchmarks** (LinCE, GLUECoS) evaluate mixed-language text but focus on intra-sentential mixing
- **Multi-turn benchmarks** (MT-Bench, MultiChallenge) assess context retention but operate monolingually

**Gap**: No existing benchmark tests whether models maintain task performance when users switch languages across conversation turns.

We address this gap by extending MultiChallenge to evaluate cross-turn language switching.

## Research Questions

- **Layer 1 (Language Fidelity)**: When users switch languages, do LLMs respond in the correct language?
- **Layer 2 (Task Accuracy)**: Even when models respond correctly, does switching hurt task completion?

## Experimental Conditions

We test 6 conditions across 4 language pairs (EN↔DE, EN↔ZH, EN↔ES, EN↔AR):

| Condition | Description | Example Pattern |
|-----------|-------------|-----------------|
| **Baseline** | Original English-only MultiChallenge | EN → EN → EN |
| **EN→X** | English context, last query in target language | EN → EN → DE |
| **X→EN** | Foreign context, English final query | DE → DE → EN |
| **Full Translation** | Entire conversation in target language | DE → DE → DE |
| **Distractor** | Foreign noise embedded in first turn, foreign query | EN[+DE noise] → EN → DE |
| **Distractor Multi** | Foreign noise in all user turns (except last), foreign query | EN[+DE] → EN[+DE] → DE |

## Task Design

We focus on two MultiChallenge axes that test memory across turns:

- **INFERENCE_MEMORY** (113 questions): Model must remember implicit constraints from earlier turns
- **INSTRUCTION_RETENTION** (69 questions): Model must follow explicit instructions given earlier

Total: **182 questions** per condition × 4 languages × 6 conditions = **4,368 evaluations per model**

## Models

| Model | Type | Notes |
|-------|------|-------|
| Claude Opus 4.5 | Closed | Best on MultiChallenge |
| GPT-5 | Closed | OpenAI flagship |
| Gemini 2.5 Pro | Closed | Google flagship |
| Llama 4 Maverick | Open | Meta MoE |
| Qwen3-235B-A22B | Open | Chinese-origin, 119 languages |
| Apertus 70B | Multilingual | Swiss, 1000+ languages |

## Results

### Layer 2: Task Accuracy

#### Qwen3-235B-A22B

| Condition | DE | ZH | ES | AR | Avg |
|-----------|----:|----:|----:|----:|----:|
| Baseline (EN) | 36.3% | 36.3% | 36.3% | 36.3% | 36.3% |
| EN→X | 33.5% | 31.3% | 36.8% | 29.7% | 32.8% |
| X→EN | 29.7% | 33.0% | 30.2% | 26.9% | 30.0% |
| Full Translation | 30.2% | 29.7% | 32.4% | 23.6% | 29.0% |
| Distractor | 33.0% | 30.2% | 31.9% | 27.5% | 30.6% |
| Distractor Multi | 37.9% | 34.1% | 35.7% | 32.4% | 35.0% |

#### Claude Opus 4.5

| Condition | DE | ZH | ES | AR | Avg |
|-----------|----:|----:|----:|----:|----:|
| Baseline (EN) | 52.0% | 52.0% | 52.0% | 52.0% | 52.0% |
| EN→X | 47.8% | 47.8% | 53.8% | 52.7% | 50.5% |
| X→EN | 45.1% | 46.2% | 47.8% | 45.6% | 46.2% |
| Full Translation | 45.6% | 51.1% | 46.2% | 49.5% | 48.1% |
| Distractor | 50.5% | 47.3% | 57.1% | 47.8% | 50.7% |
| Distractor Multi | 51.6% | 48.4% | 54.4% | 51.6% | 51.5% |

### Layer 1: Language Fidelity

#### Qwen3-235B-A22B

| Condition | Expected | DE | ZH | ES | AR | Avg |
|-----------|----------|----:|----:|----:|----:|----:|
| Baseline | EN | 100% | 100% | 100% | 100% | 100% |
| EN→X | X | 92.5% | 79.8% | 92.9% | 78.0% | 85.8% |
| X→EN | EN | 32.4% | 28.6% | 21.7% | 56.8% | 34.9% |
| Full Translation | X | 100% | 99.1% | 99.1% | 100% | 99.6% |
| Distractor | X | 94.6% | 83.1% | 92.9% | 65.5% | 84.0% |
| Distractor Multi | X | 86.9% | 78.5% | 89.4% | 42.4% | 74.3% |

#### Claude Opus 4.5

| Condition | Expected | DE | ZH | ES | AR | Avg |
|-----------|----------|----:|----:|----:|----:|----:|
| Baseline | EN | 100% | 100% | 100% | 100% | 100% |
| EN→X | X | 97.8% | 94.5% | 96.7% | 96.7% | 96.4% |
| X→EN | EN | 1.6% | 3.3% | 3.3% | 2.7% | 2.7% |
| Full Translation | X | 99.5% | 99.5% | 99.5% | 100% | 99.6% |
| Distractor | X | 94.5% | 95.1% | 95.1% | 96.2% | 95.2% |
| Distractor Multi | X | 93.9% | 92.9% | 92.3% | 94.0% | 93.3% |

### Key Findings

**Task Accuracy (Layer 2):**
- Claude shows smaller performance gaps than Qwen3 (max ~6% drop vs ~13%)
- X→EN (foreign context, English query) shows the largest drop for Claude (-5.8%)
- Arabic shows the largest degradation for Qwen across all conditions
- Distractors barely affect Claude (-0.5% to -1.3% on average)

**Language Fidelity (Layer 1):**
- X→EN is broken for both models: When context is foreign but query is English, both respond in context language (Claude: 97%, Qwen: 65%)
- Claude has better language fidelity: 96% vs 86% on EN→X codeswitching
- Arabic is problematic for Qwen: Only 78% EN→X, 42% distractor_multi
- **Key insight**: Models prioritize conversation context language over query language

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:
```bash
OPENAI_API_KEY=...      # For GPT models and GPT-4o judge
ANTHROPIC_API_KEY=...   # For Claude models
APERTUS_API_KEY=...     # For Apertus models (EPFL)
OPENROUTER_API_KEY=...  # For Gemini, Llama, Qwen via OpenRouter
```

### Generate Datasets

```bash
# EN→X: Translate last user turn
python src/scripts/generate_codeswitching.py --lang de
python src/scripts/generate_codeswitching.py --lang zh

# X→EN: Translate context, keep English query
python src/scripts/generate_codeswitching_reverse.py --lang de
python src/scripts/generate_codeswitching_reverse.py --lang zh

# Full translation: All turns in target language
python src/scripts/generate_full_translation.py --lang de

# Distractor: Embed foreign noise in context
python src/scripts/generate_distractor.py --lang zh
python src/scripts/generate_distractor.py --lang zh --mix-all-turns  # Multi-turn variant
```

### Run Experiments

```bash
# Baseline (English-only MultiChallenge)
python src/scripts/run_experiment.py --model qwen3-235b

# Code-switching experiments
python src/scripts/run_experiment.py --model qwen3-235b --data data/codeswitching/en_to_x/codeswitching_de.jsonl
python src/scripts/run_experiment.py --model qwen3-235b --data data/codeswitching/x_to_en/codeswitching_de_to_en.jsonl

# Test with limited samples
python src/scripts/run_experiment.py --model gpt-4o --samples 20
```

### Evaluate Responses

```bash
python src/scripts/evaluate.py --input results/qwen3-235b/responses_*.jsonl
```

## Project Structure

```
├── data/
│   ├── multi-challenge/          # Original MultiChallenge (submodule)
│   └── codeswitching/
│       ├── en_to_x/              # EN→X datasets
│       ├── x_to_en/              # X→EN datasets
│       ├── full_translation/     # Fully translated datasets
│       └── distractor/           # Distractor datasets
├── src/scripts/
│   ├── generate_codeswitching.py         # EN→X generation
│   ├── generate_codeswitching_reverse.py # X→EN generation
│   ├── generate_full_translation.py      # Full translation
│   ├── generate_distractor.py            # Distractor injection
│   ├── run_experiment.py                 # Model inference
│   ├── evaluate.py                       # GPT-4o judge (Layer 2)
│   └── evaluate_language.py              # Language fidelity (Layer 1)
├── results/{model}/              # Model responses and evaluations
└── prompts/judge_prompt.txt      # Evaluation prompt template
```

## Supported Languages

| Code | Language | Script |
|------|----------|--------|
| de | German | Latin |
| zh | Chinese (Simplified) | Han |
| es | Spanish | Latin |
| ar | Arabic | Arabic |

## Related Work

- [IFEval](https://arxiv.org/pdf/2311.07911) (2023): Verifiable instructions (single-turn)
- [Multi-IF](https://arxiv.org/abs/2410.15553) (2024): Multilingual extension of IFEval
- [MultiChallenge](https://arxiv.org/pdf/2501.17399) (2025): Multi-turn challenge benchmark
- [LoCoMo](https://arxiv.org/pdf/2402.17753) (2024): Very long conversation dataset (300+ turns)

## Limitations

- Translation quality: Uses Google Translate, which may introduce artifacts
- Limited language coverage: Only 4 languages tested
- Distractor design: Current distractors are benign small talk - may need adversarial variants to stress-test robust models

## Citation

```bibtex
@misc{codeswitching2025,
  title={Cross-Turn Language Switching in LLMs},
  author={},
  year={2025},
  note={Extension of MultiChallenge benchmark}
}
```
