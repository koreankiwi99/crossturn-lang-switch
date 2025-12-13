# Cross-Turn Language Switching in LLMs

Evaluating whether language models maintain task performance when users switch languages mid-conversation.

**Benchmark Extension**: [MultiChallenge](https://scale.com/leaderboard/multichallenge)

1. Introduction (0.5 page)
   - Multilingual users switch languages
   - No benchmark tests this
   - We find models diverge dramatically

2. Related Work (0.5 page)
   - Language confusion (single-turn)
   - Multi-turn benchmarks (monolingual)
   - Gap: cross-turn switching

3. Methodology (0.75 page)
   - MultiChallenge extension
   - 3 conditions: Baseline, EN→X, X→EN
   - 6 models, 2-4 language pairs
   - Two-layer evaluation

4. Results (1 page)
   - Main finding: Fidelity divergence
   - Group A (user-following) vs Group B (context-anchoring)
   - Table + figure

5. Discussion (0.5 page)
   - Implications for multilingual UX
   - Null results: accuracy stable, distractors ineffective
   - Limitations

6. Conclusion (0.25 page)

References + Appendix (as needed)


## Introduction

Large language models increasingly serve multilingual users worldwide. These users often switch between languages naturally within a single conversation—starting a query in English, then continuing in their native language, or vice versa. This cross-turn language switching reflects how multilingual speakers actually communicate: fluidly moving between languages based on comfort, context, or expression.

Existing benchmarks evaluate multilingual and multi-turn capabilities separately:
- **Multilingual benchmarks** (MMMLU, Multi-IF) test cross-lingual understanding in single-turn setups
- **Code-switching benchmarks** (LinCE, GLUECoS) evaluate mixed-language text but focus on intra-sentential mixing
- **Multi-turn benchmarks** (MT-Bench, MultiChallenge) assess context retention but operate monolingually

**Gap**: No existing benchmark tests whether models maintain task performance when users switch languages across conversation turns.

We address this gap by extending MultiChallenge to evaluate cross-turn language switching.

## Research Questions

**RQ1 (Main)**: Do LLMs follow user query language or conversation context language?
- When a user switches languages mid-conversation, which language does the model respond in?
- Measured by Layer 1: Language Fidelity

**RQ2**: Does language fidelity degrade with longer conversations?
- Do models become more "anchored" to context language as conversation length increases?

**RQ3**: Do models maintain task performance when users switch languages across turns?
- Does language switching hurt task completion?
- Measured by Layer 2: Task Accuracy (null result - no significant difference)

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
| GPT-5 | Closed | OpenAI flagship |
| Gemini 3 Pro | Closed | Google flagship |
| Claude Opus 4.5 | Closed | Anthropic flagship |

See Appendix for additional models (Qwen3-235B) and distractor conditions.

## Results

### Layer 2: Task Accuracy

#### Gemini 3 Pro

| Condition | DE | ZH | ES | AR | Avg |
|-----------|----:|----:|----:|----:|----:|
| Baseline (EN) | 71.4% | 71.4% | 71.4% | 71.4% | 71.4% |
| EN→X | 70.3% | 69.8% | 70.3% | 71.4% | 70.5% |
| X→EN | 67.6% | 68.7% | 67.6% | 69.8% | 68.4% |
| Full Translation | 69.8% | 74.7% | 69.8% | 69.8% | 71.0% |
| Distractor | 71.4% | 70.3% | 70.9% | 70.3% | 70.7% |
| Distractor Multi | 70.9% | 69.2% | 73.6% | 69.8% | 70.9% |

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
| Baseline (EN) | 54.4% | 54.4% | 54.4% | 54.4% | 54.4% |
| EN→X | 47.8% | 47.8% | 53.8% | 52.7% | 50.5% |
| X→EN | 45.1% | 46.2% | 47.8% | 45.6% | 46.2% |
| Full Translation | 45.6% | 51.1% | 46.2% | 49.5% | 48.1% |
| Distractor | 50.5% | 47.3% | 57.1% | 47.8% | 50.7% |
| Distractor Multi | 51.6% | 48.4% | 54.4% | 51.6% | 51.5% |

#### GPT-5

| Condition | DE | ZH | ES | AR | Avg |
|-----------|----:|----:|----:|----:|----:|
| Baseline (EN) | 57.1% | 57.1% | 57.1% | 57.1% | 57.1% |
| EN→X | 57.7% | 59.3% | 58.2% | 57.1% | 58.1% |
| X→EN | 53.8% | 53.8% | 57.1% | 54.1% | 54.7% |
| Full Translation | 56.0% | 57.7% | 57.1% | 58.6% | 57.4% |
| Distractor | 59.9% | 57.1% | 63.2% | 60.4% | 60.2% |
| Distractor Multi | 54.9% | 57.7% | 58.2% | 58.8% | 57.4% |

#### DeepSeek-V3.1

| Condition | DE | ZH | ES | AR | Avg |
|-----------|----:|----:|----:|----:|----:|
| Baseline (EN) | 47.3% | 47.3% | 47.3% | 47.3% | 47.3% |
| EN→X | 40.1% | 39.6% | 51.6% | 40.7% | 43.0% |
| X→EN | 38.5% | 33.5% | 45.1% | 40.1% | 39.3% |

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
| EN→X | X | 98.4% | 95.1% | 96.2% | 97.8% | 96.9% |
| X→EN | EN | 2.7% | 3.8% | 3.8% | 3.8% | 3.5% |
| Full Translation | X | 100% | 100% | 100% | 100% | 100% |
| Distractor | X | 94.5% | 96.2% | 95.1% | 96.7% | 95.6% |
| Distractor Multi | X | 93.4% | 95.1% | 92.9% | 95.1% | 94.1% |

#### GPT-5

| Condition | Expected | DE | ZH | ES | AR | Avg |
|-----------|----------|----:|----:|----:|----:|----:|
| Baseline | EN | 100% | 100% | 100% | 100% | 100% |
| EN→X | X | 98.9% | 97.8% | 100% | 98.9% | 98.9% |
| X→EN | EN | 92.3% | 94.5% | 95.1% | 95.1% | 94.2% |
| Full Translation | X | 100% | 100% | 100% | 100% | 100% |
| Distractor | X | 98.9% | 98.4% | 99.5% | 98.4% | 98.8% |
| Distractor Multi | X | 94.5% | 95.6% | 95.6% | 96.2% | 95.5% |

#### Gemini 3 Pro

| Condition | Expected | DE | ZH | ES | AR | Avg |
|-----------|----------|----:|----:|----:|----:|----:|
| Baseline | EN | 100% | 100% | 100% | 100% | 100% |
| EN→X | X | 97.8% | 99.4% | 97.8% | 99.4% | 98.6% |
| X→EN | EN | 78.3% | 70.6% | 72.2% | 64.4% | 71.4% |
| Full Translation | X | 100% | 100% | 100% | 100% | 100% |
| Distractor | X | 97.8% | 98.9% | 97.8% | 98.9% | 98.4% |
| Distractor Multi | X | 96.7% | 97.2% | 96.7% | 97.8% | 97.1% |

#### DeepSeek-V3.1

| Condition | Expected | DE | ZH | ES | AR | Avg |
|-----------|----------|----:|----:|----:|----:|----:|
| Baseline | EN | 100% | 100% | 100% | 100% | 100% |
| EN→X | X | 89.0% | 68.1% | 90.1% | 84.6% | 83.0% |
| X→EN | EN | 36.8% | 73.1% | 45.1% | 68.7% | 55.9% |

### Key Findings

#### RQ1: Query Language vs Context Language

Models diverge dramatically on X→EN (foreign context, English query):

| Model | EN→X | X→EN | Behavior |
|-------|------|------|----------|
| GPT-5 | 98.9% | **94.2%** | Follows query language |
| Gemini 3 Pro | 98.6% | **71.4%** | Mixed |
| Claude Opus 4.5 | 96.9% | **3.5%** | Follows context language |

- **EN→X**: All models respond in target language (97-99%)
- **X→EN**: Models split into two groups:
  - **Query-following**: GPT-5 responds in English when queried in English
  - **Context-anchored**: Claude responds in context language, ignoring English query

#### RQ2: Conversation Length Effect

Gemini's X→EN fidelity decreases with conversation length (χ²=32.5, p<0.0001):

| Length | Fidelity | n |
|--------|----------|---|
| Short (2-3 turns) | 82.6% | 132 |
| Medium (4-5 turns) | 70.6% | 296 |
| Long (6+ turns) | **49.1%** | 112 |

Per-language analysis confirms trend (all p<0.05). GPT-5 shows no degradation (~94% stable).

#### RQ3: Task Accuracy (Null Result)

No significant accuracy degradation from language switching:

| Model | Baseline | EN→X | X→EN | Full Trans |
|-------|----------|------|------|------------|
| Gemini 3 Pro | 71.4% | 70.5% | 68.4% | 71.0% |
| GPT-5 | 57.1% | 58.1% | 54.7% | 57.4% |
| Claude Opus 4.5 | 54.4% | 50.5% | 46.2% | 48.1% |

Task performance remains stable across language conditions.

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
- Qwen3-235B empty responses: ~40% of responses are empty due to reasoning token exhaustion (model uses hidden thinking tokens that don't appear in output). This is a known issue with reasoning models.

## Citation

```bibtex
@misc{codeswitching2025,
  title={Cross-Turn Language Switching in LLMs},
  author={},
  year={2025},
  note={Extension of MultiChallenge benchmark}
}
```
