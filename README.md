# Update-Experiment-for-LLM-reasoning-benchmark
# Agent Reasoning Benchmark List

Zero-shot, single-agent results of four open models on four reasoning benchmarks. Every model answers every question
on its own (no debate, no other agents), with the same structured prompt per benchmark. The folder keeps the prompts,
the per-question outputs, the metrics and the run settings.

# Agent Reasoning Benchmark List

Zero-shot, single-agent results of four open models on four reasoning benchmarks. Every model answers every question
on its own (no debate, no other agents), with the same structured prompt per benchmark. The folder keeps the prompts,
the per-question outputs, the metrics and the run settings.

- **Models**: Mistral-7B-Instruct-v0.3, Phi-4-mini-instruct, Qwen2.5-7B-Instruct, Gemma-3-27B-IT
- **Benchmarks**: GSM8K (math word problems), CommonsenseQA (commonsense multiple choice), CS1QA (code question
  answering, free text), MMLU-Pro (professional and academic multiple choice, up to 10 options)
## Models
- [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct),
- [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct),
- [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it)
## Benchmarks
- [openai/gsm8k (math word problems)](https://huggingface.co/datasets/openai/gsm8k),
- [tau/CommonsenseQA (commonsense multiple choice)](https://huggingface.co/datasets/tau/commonsense_qa),
- [CS1QA-testing datasets (code question answering, free text)](https://aclanthology.org/2022.naacl-main.148/),
- [datapaf/CodeQuestionAnswering](https://huggingface.co/datasets/datapaf/CodeQuestionAnswering)
- [TIGER-Lab/MMLU-Pro (professional and academic multiple choice, up to 10 options)](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
## Results

Accuracy (%) with 95% Wilson intervals. All sources use the same questions and the same gold answers.

### 1. Single-Agent V2 (main results)

| Model | GSM8K (1,319) | CommonsenseQA (1,221) | CS1QA (1,847) | MMLU-Pro (12,032) |
|---|---|---|---|---|
| Mistral-7B-Instruct-v0.3 | 51.1 [48.4, 53.8] | 69.7 [67.1, 72.2] | 28.3 [26.3, 30.4] | 31.2 [30.4, 32.1] |
| Phi-4-mini-instruct | 80.7 [78.4, 82.7] | 69.0 [66.4, 71.6] | 32.4 [30.3, 34.5] | 44.9 [44.0, 45.8] |
| Qwen2.5-7B-Instruct | 86.8 [84.9, 88.5] | 82.1 [79.8, 84.1] | 35.5 [33.3, 37.7] | 50.8 [49.9, 51.7] |

### 2. Single agent with confidence (independent answers S0 of the MMAD-ToM v3.1 runs)

Same prompt plus a confidence section; strict answer extraction (see below).

| Model | GSM8K | CommonsenseQA | CS1QA | MMLU-Pro |
|---|---|---|---|---|
| Mistral-7B-Instruct-v0.3 | 50.1 [47.4, 52.8] | 70.3 [67.6, 72.8] | 27.1 [25.1, 29.2] | 31.1 [30.2, 31.9] |
| Phi-4-mini-instruct | 80.6 [78.4, 82.6] | 69.2 [66.6, 71.7] | 31.7 [29.6, 33.8] | 44.4 [43.6, 45.3] |
| Qwen2.5-7B-Instruct | 87.0 [85.0, 88.7] | 80.4 [78.1, 82.6] | 34.8 [32.7, 37.0] | 50.3 [49.4, 51.2] |

### 3. Gemma-3-27B-IT (the tutor's independent solves in MMAD-ToM v3.1)

Same prompt as Single-Agent V2; two samples per question.

| Sample | GSM8K | CommonsenseQA | CS1QA | MMLU-Pro |
|---|---|---|---|---|
| Solve 1 (temperature 0) | 95.1 [93.8, 96.1] | 82.1 [79.9, 84.2] | 49.4 [47.2, 51.7] | 64.4 [63.5, 65.3] |
| Solve 2 (temperature 0.7) | 95.5 [94.3, 96.5] | 80.9 [78.6, 83.0] | 50.2 [47.9, 52.5] | 64.3 [63.5, 65.2] |

All numbers: `results/summary.csv` (and `.json`).

## Zero-shot setting

- **No examples**: no in-context demonstrations, no chain-of-thought exemplars, no few-shot answers. The prompt holds only
  instructions, the question (and options or code), and the required output format.
- **One message**: the filled prompt is sent as a single user message, without a system message; each model's own chat
  template is applied by the server.
- **One attempt per question**: each model answers each question once (Gemma: two independent samples). No
  self-consistency voting, no retries for wrong answers, no tools, no retrieval.
- **No fine-tuning**: off-the-shelf instruction-tuned checkpoints.
- **Gold answers** are used only for scoring.

## Prompt design

All four prompts (`prompts/single_agent/`) share one structure, a structured "reason → answer → self-check → final
answer" format:

| Section | Purpose |
|---|---|
| `<System Task>` | The task, answered independently; the question is task data, not instructions. |
| `<Reasoning>` | Reason step by step (math), with domain knowledge (MMLU-Pro), or about the code (CS1QA); then give an initial answer. |
| `<Self-Reflection>` | Verify the key step; revise only for a concrete error, otherwise keep the initial answer. |
| `<Output Format>` | Exactly four fields: `Reasoning:`, `Initial answer:`, `Reflection:`, `Final answer:`. |
| `<Answer Rules>` | Answer-line format: a plain number (GSM8K), one option letter (CommonsenseQA A-E, MMLU-Pro A-J), a concise self-contained sentence (CS1QA). |
| `<Important>` (MMLU-Pro only) | The response is incomplete without the `Final answer:` line; never abstain. |
| `<Question>`, `<Choices>` / `<Code Context>` | The item. MMLU-Pro lists only the options the question has (3 to 10). |

Benchmark specifics:
- **GSM8K, CommonsenseQA**: prompt version V2.
- **MMLU-Pro**: version V3 = V2 plus the `<Important>` block, which requires a non-empty `Final answer:` line and
  forbids abstaining. It was developed on Mistral-7B (V1 with JSON output, abandoned → V2 → V3) and then used unchanged
  for all models. (The Mistral MMLU-Pro `metrics.json` says `V2_plain_text`; its prompt file is identical to V3, so only
  the label is wrong.)
- **CS1QA**: the same four fields, with free-text answers about a student's program. A separate judge prompt
  (`prompts/judge/CS1QA_judge.txt`) decides `CORRECT` / `INCORRECT` from the question, code, reference and candidate
  answer.
- **With confidence** (`prompts/single_agent_with_confidence/`, result set 2): the same prompt with one more section,
  `<Confidence>`, asking for a last line `Confidence: <integer 0-100>`, the model's own estimate that its final answer is
  correct.

## Experiment parameters

### Generation

| Setting | Single-Agent V2 | With confidence (MMAD v3.1 S0) | Gemma-3-27B-IT (MMAD v3.1 T0) |
|---|---|---|---|
| Temperature | 0.3 | 0.3 | 0 (solve 1), 0.7 (solve 2) |
| top_p | 0.9 | 0.9 | 1.0 (solve 1), 0.9 (solve 2) |
| max_tokens | 2,048 (CS1QA: 1,024) | 2,048 | 2,048 |
| Sampling seed | not set | per request, derived from (run seed 0, question, stage, role) | same as S0 |
| Stop strings | none | a new line starting with a prompt section header (`<Answer Rules>`, `<Confidence>`, `<Question>`, …) | same, without `<Confidence>` |

The stop strings were added because Phi-4-mini, after the confidence line, often repeated the prompt up to the token
limit; generation now ends at the repeat, and the answer before it is unchanged.

### Judge (CS1QA only)

| Setting | Value |
|---|---|
| Model | Qwen/Qwen3.5-9B |
| Temperature / top_p / max_tokens | 0 / 1.0 / 32 |
| Thinking | disabled (`enable_thinking: false`) |
| Output | `CORRECT` or `INCORRECT`; no label counts as incorrect |

### Serving

| Setting | Single-Agent V2 | MMAD v3.1 (sets 2 and 3) |
|---|---|---|
| Engine | vLLM 0.29.0, OpenAI-compatible API, bfloat16 | vLLM 0.29.0, OpenAI-compatible API, bfloat16 |
| Context length | 8,192 | students 12,288 (CS1QA 16,384); Gemma 16,384 (CS1QA 32,768) |
| GPUs | one A100-80GB per model (CS1QA: H200) | three students together on one A100-80GB; Gemma on a second A100-80GB |

Model checkpoints: `mistralai/Mistral-7B-Instruct-v0.3`, `microsoft/Phi-4-mini-instruct`, `Qwen/Qwen2.5-7B-Instruct`,
`google/gemma-3-27b-it`, judge `Qwen/Qwen3.5-9B`.

## Answer extraction and scoring

- **GSM8K**: the number on the `Final answer:` line, compared numerically with the gold number.
- **CommonsenseQA, MMLU-Pro**: the option letter on the `Final answer:` line, exact match with the gold letter.
- **CS1QA**: the text of the `Final answer:` line (markdown headings such as `### Final answer` accepted), scored by the
  judge.
- An answer that cannot be extracted counts as incorrect. Accuracy uses all questions as the denominator.

**Single-Agent V2** reads the answer in three layers: the `Final answer:` line directly; else a number or letter found
inside that line; else the last number or letter anywhere in the response. Counts per run:

| Run | Direct | Inside the final line | Anywhere in the response | No answer |
|---|---|---|---|---|
| Mistral GSM8K | 1,201 | 95 | 23 | 0 |
| Phi GSM8K | 1,295 | 22 | 2 | 0 |
| Qwen GSM8K | 1,319 | 0 | 0 | 0 |
| Mistral CommonsenseQA | 1,107 | 103 | 11 | 0 |
| Phi CommonsenseQA | 1,218 | 3 | 0 | 0 |
| Qwen CommonsenseQA | 1,221 | 0 | 0 | 0 |
| Mistral MMLU-Pro | 9,799 | 1,531 | 384 | 318 |
| Phi MMLU-Pro | 11,392 | 508 | 50 | 82 |
| Qwen MMLU-Pro | 12,006 | 7 | 10 | 9 |

**Set 2 (with confidence)** reads the answer only from the declared `Final answer:` line, never from the rest of the
response; an MMLU-Pro letter must be one of the question's options. Sets 1 and 2 agree within about 1.7 points everywhere.

## Folder layout

```text
README.md
prompts/
├── single_agent/                    GSM8K.txt, CommonsenseQA.txt, CS1QA.txt, MMLU-Pro.txt   (sets 1 and 3)
├── single_agent_with_confidence/    the same four prompts with the confidence section       (set 2)
└── judge/CS1QA_judge.txt            CS1QA scoring prompt
results/
├── summary.csv, summary.json        accuracy of every model × benchmark × set
├── single_agent_v2/<Benchmark>/<Model>/
│   ├── records.jsonl.gz             one line per question: full response, extracted answer, gold, correct, extraction method
│   │                                (CS1QA: candidate_answer, reference, judge_response, correct, questionType)
│   ├── predictions.json             compact predictions
│   ├── metrics.json                 accuracy, format compliance, extraction counts (CS1QA: by question type)
│   └── run.json                     run settings
├── mmad_v3.1_s0/<Benchmark>/
│   ├── records.jsonl.gz             one line per question: question, gold, and per model raw output, answer, confidence, correct
│   └── metrics.json                 accuracy per model with 95% question-bootstrap intervals
└── gemma_tutor_t0/<Benchmark>/
    ├── records.jsonl.gz             one line per question: the two solves (raw output, answer, correct, temperature)
    └── metrics.json
code/single_agent_v2/                the Single-Agent V2 runners of each model, the shared run_single.py, the vLLM job script
```

The datasets themselves are not included: GSM8K test (`openai/gsm8k`, config `main`), CommonsenseQA validation
(`tau/commonsense_qa`), the cleaned CS1QA test split, MMLU-Pro test (`TIGER-Lab/MMLU-Pro`). Question ids are the 0-based
row indices of these files. Paths of the original compute environment are replaced by placeholders such as `<node>`.

## Notes

- Sets 2 and 3 come from the MMAD-ToM v3.1 runs: S0 is each student's independent answer before any debate, and T0 is
  the tutor's own solve from the question alone. Neither sees any other agent's output.
- One sample per question: the intervals reflect question sampling only, not run-to-run variation.
- CS1QA in set 1 used max_tokens 1,024 (6 Qwen answers were cut off); all other runs used 2,048.

## Read

```python
import gzip, json

with gzip.open("results/single_agent_v2/GSM8K/Qwen2.5-7B-Instruct/records.jsonl.gz", "rt") as f:
    for line in f:
        r = json.loads(line)
        print(r["index"], r["extracted_answer"], r["gold_answer"], r["correct"])
```

## Results

Accuracy (%) with 95% Wilson intervals. All sources use the same questions and the same gold answers.

### 1. Single-Agent V2 (main results)

| Model | GSM8K (1,319) | CommonsenseQA (1,221) | CS1QA (1,847) | MMLU-Pro (12,032) |
|---|---|---|---|---|
| Mistral-7B-Instruct-v0.3 | 51.1 [48.4, 53.8] | 69.7 [67.1, 72.2] | 28.3 [26.3, 30.4] | 31.2 [30.4, 32.1] |
| Phi-4-mini-instruct | 80.7 [78.4, 82.7] | 69.0 [66.4, 71.6] | 32.4 [30.3, 34.5] | 44.9 [44.0, 45.8] |
| Qwen2.5-7B-Instruct | 86.8 [84.9, 88.5] | 82.1 [79.8, 84.1] | 35.5 [33.3, 37.7] | 50.8 [49.9, 51.7] |

### 2. Single agent with confidence (independent answers S0 of the MMAD-ToM v3.1 runs)

Same prompt plus a confidence section; strict answer extraction (see below).

| Model | GSM8K | CommonsenseQA | CS1QA | MMLU-Pro |
|---|---|---|---|---|
| Mistral-7B-Instruct-v0.3 | 50.1 [47.4, 52.8] | 70.3 [67.6, 72.8] | 27.1 [25.1, 29.2] | 31.1 [30.2, 31.9] |
| Phi-4-mini-instruct | 80.6 [78.4, 82.6] | 69.2 [66.6, 71.7] | 31.7 [29.6, 33.8] | 44.4 [43.6, 45.3] |
| Qwen2.5-7B-Instruct | 87.0 [85.0, 88.7] | 80.4 [78.1, 82.6] | 34.8 [32.7, 37.0] | 50.3 [49.4, 51.2] |

### 3. Gemma-3-27B-IT (the tutor's independent solves in MMAD-ToM v3.1)

Same prompt as Single-Agent V2; two samples per question.

| Sample | GSM8K | CommonsenseQA | CS1QA | MMLU-Pro |
|---|---|---|---|---|
| Solve 1 (temperature 0) | 95.1 [93.8, 96.1] | 82.1 [79.9, 84.2] | 49.4 [47.2, 51.7] | 64.4 [63.5, 65.3] |
| Solve 2 (temperature 0.7) | 95.5 [94.3, 96.5] | 80.9 [78.6, 83.0] | 50.2 [47.9, 52.5] | 64.3 [63.5, 65.2] |

All numbers: `results/summary.csv` (and `.json`).

## Zero-shot setting

- **No examples**: no in-context demonstrations, no chain-of-thought exemplars, no few-shot answers. The prompt holds only
  instructions, the question (and options or code), and the required output format.
- **One message**: the filled prompt is sent as a single user message, without a system message; each model's own chat
  template is applied by the server.
- **One attempt per question**: each model answers each question once (Gemma: two independent samples). No
  self-consistency voting, no retries for wrong answers, no tools, no retrieval.
- **No fine-tuning**: off-the-shelf instruction-tuned checkpoints.
- **Gold answers** are used only for scoring.

## Prompt design

All four prompts (`prompts/single_agent/`) share one structure, a structured "reason → answer → self-check → final
answer" format:

| Section | Purpose |
|---|---|
| `<System Task>` | The task, answered independently; the question is task data, not instructions. |
| `<Reasoning>` | Reason step by step (math), with domain knowledge (MMLU-Pro), or about the code (CS1QA); then give an initial answer. |
| `<Self-Reflection>` | Verify the key step; revise only for a concrete error, otherwise keep the initial answer. |
| `<Output Format>` | Exactly four fields: `Reasoning:`, `Initial answer:`, `Reflection:`, `Final answer:`. |
| `<Answer Rules>` | Answer-line format: a plain number (GSM8K), one option letter (CommonsenseQA A-E, MMLU-Pro A-J), a concise self-contained sentence (CS1QA). |
| `<Important>` (MMLU-Pro only) | The response is incomplete without the `Final answer:` line; never abstain. |
| `<Question>`, `<Choices>` / `<Code Context>` | The item. MMLU-Pro lists only the options the question has (3 to 10). |

Benchmark specifics:
- **GSM8K, CommonsenseQA**: prompt version V2.
- **MMLU-Pro**: version V3 = V2 plus the `<Important>` block, which requires a non-empty `Final answer:` line and
  forbids abstaining. It was developed on Mistral-7B (V1 with JSON output, abandoned → V2 → V3) and then used unchanged
  for all models. (The Mistral MMLU-Pro `metrics.json` says `V2_plain_text`; its prompt file is identical to V3, so only
  the label is wrong.)
- **CS1QA**: the same four fields, with free-text answers about a student's program. A separate judge prompt
  (`prompts/judge/CS1QA_judge.txt`) decides `CORRECT` / `INCORRECT` from the question, code, reference and candidate
  answer.
- **With confidence** (`prompts/single_agent_with_confidence/`, result set 2): the same prompt with one more section,
  `<Confidence>`, asking for a last line `Confidence: <integer 0-100>`, the model's own estimate that its final answer is
  correct.

## Experiment parameters

### Generation

| Setting | Single-Agent V2 | With confidence (MMAD v3.1 S0) | Gemma-3-27B-IT (MMAD v3.1 T0) |
|---|---|---|---|
| Temperature | 0.3 | 0.3 | 0 (solve 1), 0.7 (solve 2) |
| top_p | 0.9 | 0.9 | 1.0 (solve 1), 0.9 (solve 2) |
| max_tokens | 2,048 (CS1QA: 1,024) | 2,048 | 2,048 |
| Sampling seed | not set | per request, derived from (run seed 0, question, stage, role) | same as S0 |
| Stop strings | none | a new line starting with a prompt section header (`<Answer Rules>`, `<Confidence>`, `<Question>`, …) | same, without `<Confidence>` |

The stop strings were added because Phi-4-mini, after the confidence line, often repeated the prompt up to the token
limit; generation now ends at the repeat, and the answer before it is unchanged.

### Judge (CS1QA only)

| Setting | Value |
|---|---|
| Model | Qwen/Qwen3.5-9B |
| Temperature / top_p / max_tokens | 0 / 1.0 / 32 |
| Thinking | disabled (`enable_thinking: false`) |
| Output | `CORRECT` or `INCORRECT`; no label counts as incorrect |

### Serving

| Setting | Single-Agent V2 | MMAD v3.1 (sets 2 and 3) |
|---|---|---|
| Engine | vLLM 0.29.0, OpenAI-compatible API, bfloat16 | vLLM 0.29.0, OpenAI-compatible API, bfloat16 |
| Context length | 8,192 | students 12,288 (CS1QA 16,384); Gemma 16,384 (CS1QA 32,768) |
| GPUs | one A100-80GB per model (CS1QA: H200) | three students together on one A100-80GB; Gemma on a second A100-80GB |

Model checkpoints: `mistralai/Mistral-7B-Instruct-v0.3`, `microsoft/Phi-4-mini-instruct`, `Qwen/Qwen2.5-7B-Instruct`,
`google/gemma-3-27b-it`, judge `Qwen/Qwen3.5-9B`.

## Answer extraction and scoring

- **GSM8K**: the number on the `Final answer:` line, compared numerically with the gold number.
- **CommonsenseQA, MMLU-Pro**: the option letter on the `Final answer:` line, exact match with the gold letter.
- **CS1QA**: the text of the `Final answer:` line (markdown headings such as `### Final answer` accepted), scored by the
  judge.
- An answer that cannot be extracted counts as incorrect. Accuracy uses all questions as the denominator.

**Single-Agent V2** reads the answer in three layers: the `Final answer:` line directly; else a number or letter found
inside that line; else the last number or letter anywhere in the response. Counts per run:

| Run | Direct | Inside the final line | Anywhere in the response | No answer |
|---|---|---|---|---|
| Mistral GSM8K | 1,201 | 95 | 23 | 0 |
| Phi GSM8K | 1,295 | 22 | 2 | 0 |
| Qwen GSM8K | 1,319 | 0 | 0 | 0 |
| Mistral CommonsenseQA | 1,107 | 103 | 11 | 0 |
| Phi CommonsenseQA | 1,218 | 3 | 0 | 0 |
| Qwen CommonsenseQA | 1,221 | 0 | 0 | 0 |
| Mistral MMLU-Pro | 9,799 | 1,531 | 384 | 318 |
| Phi MMLU-Pro | 11,392 | 508 | 50 | 82 |
| Qwen MMLU-Pro | 12,006 | 7 | 10 | 9 |

**Set 2 (with confidence)** reads the answer only from the declared `Final answer:` line, never from the rest of the
response; an MMLU-Pro letter must be one of the question's options. Sets 1 and 2 agree within about 1.7 points everywhere.

## Folder layout

```text
README.md
prompts/
├── single_agent/                    GSM8K.txt, CommonsenseQA.txt, CS1QA.txt, MMLU-Pro.txt   (sets 1 and 3)
├── single_agent_with_confidence/    the same four prompts with the confidence section       (set 2)
└── judge/CS1QA_judge.txt            CS1QA scoring prompt
results/
├── summary.csv, summary.json        accuracy of every model × benchmark × set
├── single_agent_v2/<Benchmark>/<Model>/
│   ├── records.jsonl.gz             one line per question: full response, extracted answer, gold, correct, extraction method
│   │                                (CS1QA: candidate_answer, reference, judge_response, correct, questionType)
│   ├── predictions.json             compact predictions
│   ├── metrics.json                 accuracy, format compliance, extraction counts (CS1QA: by question type)
│   └── run.json                     run settings
├── mmad_v3.1_s0/<Benchmark>/
│   ├── records.jsonl.gz             one line per question: question, gold, and per model raw output, answer, confidence, correct
│   └── metrics.json                 accuracy per model with 95% question-bootstrap intervals
└── gemma_tutor_t0/<Benchmark>/
    ├── records.jsonl.gz             one line per question: the two solves (raw output, answer, correct, temperature)
    └── metrics.json
code/single_agent_v2/                the Single-Agent V2 runners of each model, the shared run_single.py, the vLLM job script
```

The datasets themselves are not included: GSM8K test (`openai/gsm8k`, config `main`), CommonsenseQA validation
(`tau/commonsense_qa`), the cleaned CS1QA test split, MMLU-Pro test (`TIGER-Lab/MMLU-Pro`). Question ids are the 0-based
row indices of these files. Paths of the original compute environment are replaced by placeholders such as `<node>`.

## Notes

- Sets 2 and 3 come from the MMAD-ToM v3.1 runs: S0 is each student's independent answer before any debate, and T0 is
  the tutor's own solve from the question alone. Neither sees any other agent's output.
- One sample per question: the intervals reflect question sampling only, not run-to-run variation.
- CS1QA in set 1 used max_tokens 1,024 (6 Qwen answers were cut off); all other runs used 2,048.

## Read

```python
import gzip, json

with gzip.open("results/single_agent_v2/GSM8K/Qwen2.5-7B-Instruct/records.jsonl.gz", "rt") as f:
    for line in f:
        r = json.loads(line)
        print(r["index"], r["extracted_answer"], r["gold_answer"], r["correct"])
```



