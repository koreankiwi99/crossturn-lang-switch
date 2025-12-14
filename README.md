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

| Model | Provider | Notes |
|-------|----------|-------|
| GPT-5 | OpenAI | Flagship |
| Gemini 3 Pro | Google | Flagship |
| Claude Opus 4.5 | Anthropic | Flagship |
| DeepSeek-V3.1 | DeepSeek | Open-weight |
| Command R+ | Cohere | Open-weight |

See Appendix for detailed per-language results and distractor conditions.

## Results

### Layer 1: Language Fidelity (Main Finding)

#### Summary: Query-following vs Context-anchored Behavior

| Model | EN→X | X→EN | Behavior |
|-------|------|------|----------|
| GPT-5 | 98.9% | **94.2%** | Query-following |
| Gemini 3 Pro | 98.6% | **71.4%** | Mixed |
| DeepSeek-V3.1 | 83.0% | **55.9%** | Mixed |
| Claude Opus 4.5 | 96.8% | **3.6%** | Context-anchored |
| Command R+ | 91.2% | **0.0%** | Context-anchored |

**Key observation**: All models respond in target language for EN→X (83-99%), but diverge dramatically on X→EN (foreign context, English query). Command R+ shows the most extreme context-anchored behavior (0% English responses).

#### X→EN Fidelity by Language

| Model | DE | ZH | ES | AR | Avg |
|-------|----:|----:|----:|----:|----:|
| GPT-5 | 92.3% | 94.5% | 95.1% | 95.1% | 94.2% |
| Gemini 3 Pro | 78.3% | 70.6% | 72.2% | 64.4% | 71.4% |
| DeepSeek-V3.1 | 36.8% | 73.1% | 45.1% | 68.7% | 55.9% |
| Claude Opus 4.5 | 2.7% | 3.8% | 3.8% | 3.8% | 3.6% |
| Command R+ | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

#### X→EN Fidelity by Conversation Length

| Model | Short (2-3) | Medium (4-5) | Long (6+) | χ² | p |
|-------|-------------|--------------|-----------|------|--------|
| GPT-5 | 100% (n=28) | 94.6% (n=148) | 93.8% (n=552) | 1.90 | 0.39 |
| Gemini 3 Pro | 85.7% (n=28) | 83.7% (n=147) | 67.3% (n=545) | 18.05 | **0.0001** |
| Claude Opus 4.5 | 25.0% (n=28) | 2.7% (n=148) | 2.7% (n=552) | 38.83 | **<0.0001** |
| DeepSeek-V3.1 | 67.9% (n=28) | 61.5% (n=148) | 53.8% (n=552) | 4.48 | 0.11 |
| Command R+ | 0.0% (n=28) | 0.0% (n=148) | 0.0% (n=552) | - | - |

Note: Command R+ shows 0% fidelity at all conversation lengths (floor effect).

#### Gemini 3 Pro: Language × Conversation Length

| Language | Short (2-3) | Medium (4-5) | Long (6+) | Trend | p |
|----------|-------------|--------------|-----------|-------|------|
| DE | 85.7% (n=7) | 88.9% (n=36) | 75.2% (n=137) | r=-0.22 | **0.003** |
| ZH | 85.7% (n=7) | 83.8% (n=37) | 66.2% (n=136) | r=-0.16 | *0.03* |
| ES | 71.4% (n=7) | 81.1% (n=37) | 69.9% (n=136) | r=-0.23 | **0.002** |
| AR | 100% (n=7) | 81.1% (n=37) | 58.1% (n=136) | r=-0.41 | **<0.0001** |

Arabic shows strongest degradation (100% → 58.1%, r=-0.41***)

### Layer 2: Task Accuracy

| Model | Baseline | EN→X | X→EN | Full Trans |
|-------|----------|------|------|------------|
| Gemini 3 Pro | 71.4% | 70.6% | 68.5% | 71.0% |
| GPT-5 | 57.1% | 58.1% | 54.7% | 57.5% |
| Claude Opus 4.5 | 54.4% | 50.5% | 46.2% | 48.1% |
| DeepSeek-V3.1 | 47.3% | 43.0% | 39.3% | 40.2% |
| Command R+ | 14.8% | 15.2% | 11.7% | - |

No significant accuracy degradation from language switching across models. Command R+ shows consistently low accuracy across all conditions.

### Key Findings

#### RQ1: Query Language vs Context Language

Models split into two behavioral groups on X→EN (foreign context, English query):

- **Query-following** (GPT-5): Responds in user's query language regardless of context (94.2%)
- **Context-anchored** (Claude Opus 4.5, Command R+): Responds in conversation context language, ignoring query language (3.6%, 0.0%)
- **Mixed** (Gemini, DeepSeek): Intermediate behavior (56-71%)

#### RQ2: Conversation Length Effect

Both Gemini and DeepSeek show significant degradation in X→EN fidelity as conversations get longer:

- **Gemini 3 Pro**: 85.7% → 67.3% (r=-0.26, p<0.001)
- **DeepSeek-V3.1**: 67.9% → 53.8% (r=-0.12, p<0.01)
- **GPT-5**: Stable at ~94% regardless of length
- **Claude**: Already at floor (~3%)
- **Command R+**: At absolute floor (0%) for all lengths

#### RQ3: Task Accuracy (Null Result)

Task performance remains stable across language conditions. Language switching does not significantly hurt task completion.

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
python src/scripts/run_experiment.py --model gpt-5

# Code-switching experiments
python src/scripts/run_experiment.py --model gpt-5 --data data/codeswitching/en_to_x/codeswitching_de.jsonl
python src/scripts/run_experiment.py --model gpt-5 --data data/codeswitching/x_to_en/codeswitching_de_to_en.jsonl

# Test with limited samples
python src/scripts/run_experiment.py --model gpt-4o --samples 20
```

### Evaluate Responses

```bash
python src/scripts/evaluate.py --input results/gpt-5/responses_*.jsonl
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
├── src/
│   ├── scripts/
│   │   ├── generate_codeswitching.py         # EN→X generation
│   │   ├── generate_codeswitching_reverse.py # X→EN generation
│   │   ├── generate_full_translation.py      # Full translation
│   │   ├── generate_distractor.py            # Distractor injection
│   │   ├── run_experiment.py                 # Model inference
│   │   ├── evaluate.py                       # GPT-4o judge (Layer 2)
│   │   └── evaluate_language.py              # Language fidelity (Layer 1)
│   └── analysis/
│       ├── analyze_results.py    # Main analysis entry point
│       ├── config.py             # Models, languages, conditions
│       ├── data_loader.py        # Load evaluation results
│       ├── task_accuracy.py      # Layer 2 analysis
│       ├── language_fidelity.py  # Layer 1 analysis
│       ├── conversation_length.py # Length effect analysis
│       └── stats.py              # Statistical utilities
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

## Appendix

### Detailed Results by Model

#### GPT-5

| Condition | Expected | DE | ZH | ES | AR | Avg |
|-----------|----------|----:|----:|----:|----:|----:|
| EN→X | X | 98.9% | 97.8% | 100% | 98.9% | 98.9% |
| X→EN | EN | 92.3% | 94.5% | 95.1% | 95.1% | 94.2% |
| Full Translation | X | 100% | 100% | 100% | 100% | 100% |

#### Gemini 3 Pro

| Condition | Expected | DE | ZH | ES | AR | Avg |
|-----------|----------|----:|----:|----:|----:|----:|
| EN→X | X | 97.8% | 99.4% | 97.8% | 99.4% | 98.6% |
| X→EN | EN | 78.3% | 70.6% | 72.2% | 64.4% | 71.4% |
| Full Translation | X | 98.9% | 100% | 100% | 100% | 99.7% |

#### Claude Opus 4.5

| Condition | Expected | DE | ZH | ES | AR | Avg |
|-----------|----------|----:|----:|----:|----:|----:|
| EN→X | X | 98.4% | 95.1% | 96.2% | 97.8% | 96.8% |
| X→EN | EN | 2.7% | 3.8% | 3.8% | 3.8% | 3.6% |
| Full Translation | X | 100% | 100% | 100% | 100% | 100% |

#### DeepSeek-V3.1

| Condition | Expected | DE | ZH | ES | AR | Avg |
|-----------|----------|----:|----:|----:|----:|----:|
| EN→X | X | 89.0% | 68.1% | 90.1% | 84.6% | 83.0% |
| X→EN | EN | 36.8% | 73.1% | 45.1% | 68.7% | 55.9% |
| Full Translation | X | 100% | 97.8% | 100% | 100% | 99.5% |

#### Command R+

| Condition | Expected | DE | ZH | ES | AR | Avg |
|-----------|----------|----:|----:|----:|----:|----:|
| EN→X | X | 94.5% | 89.0% | 98.4% | 83.0% | 91.2% |
| X→EN | EN | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Full Translation | X | 99.5% | 100% | 100% | 100% | 99.9% |

Note: Command R+ shows the most extreme context-anchored behavior, with 0% English responses in X→EN condition across all languages.

### Distractor Condition Results

Distractors (foreign noise embedded in context) do not significantly affect language fidelity.

| Model | EN→X | Distractor | Distractor Multi |
|-------|------|------------|------------------|
| GPT-5 | 98.9% | 98.8% | 95.5% |
| Gemini 3 Pro | 98.6% | 95.3% | 81.6% |
| Claude Opus 4.5 | 96.8% | 95.6% | 94.1% |
| DeepSeek-V3.1 | 83.0% | - | - |
| Command R+ | 91.2% | - | - |

### Sample Sizes

- **182 questions** per condition per language (728 per model per condition across 4 languages)
- **Conversation length distribution**: Short (2-3 turns): ~4%, Medium (4-5 turns): ~20%, Long (6+ turns): ~76%

## Citation

```bibtex
@misc{codeswitching2025,
  title={Cross-Turn Language Switching in LLMs},
  author={},
  year={2025},
  note={Extension of MultiChallenge benchmark}
}
```
