# MultiChallenge Dataset

## Overview

**Total: 182 questions** (INFERENCE_MEMORY + INSTRUCTION_RETENTION)

Source: [MultiChallenge Benchmark](https://arxiv.org/pdf/2501.17399) (2025)

## Distribution by Axis

| Axis | Count | % | Description |
|------|-------|---|-------------|
| INFERENCE_MEMORY | 113 | 62.1% | Tests if model remembers constraints from earlier turns |
| INSTRUCTION_RETENTION | 69 | 37.9% | Tests if model follows formatting rules throughout conversation |

## Distribution by Turn Count

### INFERENCE_MEMORY (113 questions)

| Turns | Count | Description |
|-------|-------|-------------|
| 3 | ~95 | Short: constraint → response → test |
| 5 | ~12 | Medium |
| 7 | ~6 | Longer conversations |

**Average: ~3.4 turns**

### INSTRUCTION_RETENTION (69 questions)

| Turns | Count | Description |
|-------|-------|-------------|
| 3 | ~18 | Short |
| 5 | ~35 | Medium: rule set → follow-ups |
| 7 | ~12 | Multiple follow-up requests |
| 9 | ~4 | Extended conversations |

**Average: ~4.8 turns**

## Example Questions

### INFERENCE_MEMORY

**Turn 1 (user):** "I am an International relations expert at UN. I hate taxis. I prefer venues within **5-minute walk** from UN headquarters."

**Turn 2 (assistant):** "Here are places within 5-minute walk: 1. Dag Plaza..."

**Turn 3 (user):** "I am meeting a German diplomat. Recommend a restaurant."

**TARGET_QUESTION:** "Are the restaurants chosen within a 5-minute walk from the UN headquarters?"

**PASS_CRITERIA:** YES

---

### INSTRUCTION_RETENTION

**Turn 1 (user):** "Please respond to each question using only **2 sentences**. Keep each sentence **under 15 words**."

**Turn 2 (assistant):** "A match becomes memorable through high-stakes storytelling..."

**Turn 3 (user):** "What are the key elements of storytelling, match variety, and crowd interaction?"

**TARGET_QUESTION:** "Does the model remember to respond with two sentences, both of which use 15 words or less?"

**PASS_CRITERIA:** YES

## Data Format

Each question in `benchmark_questions.jsonl`:

```json
{
  "QUESTION_ID": "674552683acc22154b07a598",
  "AXIS": "INFERENCE_MEMORY",
  "CONVERSATION": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "TARGET_QUESTION": "Are the restaurants within 5-minute walk?",
  "PASS_CRITERIA": "YES"
}
```

## Directory Structure

```
data/
├── multi-challenge/              # Original MultiChallenge benchmark (git submodule)
│   └── data/
│       └── benchmark_questions.jsonl   # 182 questions (EN baseline)
│
└── codeswitching/               # Generated translated datasets
    ├── en_to_x/                  # Code-switching: EN context → X query
    │   ├── codeswitching_de.jsonl   # 182 questions
    │   ├── codeswitching_zh.jsonl   # 182 questions
    │   ├── codeswitching_es.jsonl   # 182 questions
    │   └── codeswitching_ar.jsonl   # 182 questions
    │
    ├── x_to_en/                  # Reverse: X context → EN query
    │   ├── codeswitching_de_to_en.jsonl   # 182 questions
    │   ├── codeswitching_zh_to_en.jsonl   # 182 questions
    │   ├── codeswitching_es_to_en.jsonl   # 182 questions
    │   └── codeswitching_ar_to_en.jsonl   # 182 questions
    │
    └── full_translation/         # Fully translated (monolingual non-EN)
        ├── full_translation_de.jsonl   # 182 questions (all turns in German)
        ├── full_translation_zh.jsonl   # 182 questions (all turns in Chinese)
        ├── full_translation_es.jsonl   # 182 questions (all turns in Spanish)
        └── full_translation_ar.jsonl   # 182 questions (all turns in Arabic)
```

## Dataset Types

### 1. English Baseline (Original MultiChallenge)
- **Location:** `data/multi-challenge/data/benchmark_questions.jsonl`
- **Count:** 182 questions
- **Description:** Original English conversations from MultiChallenge benchmark

### 2. EN→X Code-Switching
- **Location:** `data/codeswitching/en_to_x/`
- **Pattern:** English context, foreign language final query
- **Use case:** Tests if model maintains constraints when user switches TO a foreign language

### 3. X→EN Reverse Code-Switching
- **Location:** `data/codeswitching/x_to_en/`
- **Pattern:** Foreign language context, English final query
- **Use case:** Tests if model maintains constraints when context is foreign but query is English

### 4. Full Translation
- **Location:** `data/codeswitching/full_translation/`
- **Pattern:** All turns translated to target language (monolingual non-English)
- **Use case:** Baseline for pure multilingual capability without code-switching

## Languages

| Code | Language | Script |
|------|----------|--------|
| DE | German | Latin |
| ZH | Chinese (Simplified) | Hanzi |
| ES | Spanish | Latin |
| AR | Arabic | Arabic |

## Code-Switching Extension

We extend this dataset by:

1. **Selecting turn N** for language switch
2. **Translating user message** at turn N to target language (DE, ZH, ES, AR)
3. **Evaluating**:
   - Layer 1: Language Fidelity (did model switch language?)
   - Layer 2: Task Accuracy (did model retain memory/instructions?)
