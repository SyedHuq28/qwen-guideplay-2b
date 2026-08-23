# Qwen-GuidePlay-2B

Official code and model release for:

**First Make It Playable, Then Make It Good: Staged Interaction Learning for Small Dialogue-Game Agents**

**Syed Mahbubul Huq · Pranava Madhyastha**
City St George's, University of London · The Alan Turing Institute

[🤗 Hugging Face](https://huggingface.co/syedhuq/qwen-guideplay-2b)

---

## Overview

**Qwen-GuidePlay-2B** is a 2B-parameter dialogue-game agent built from **Qwen3.5-2B** using a staged interaction-learning pipeline for the Playpen benchmark.

The central idea is simple:

> **First make the model playable, then improve how well it plays.**

The training pipeline separates learning valid interaction behaviour from improving local decision quality:

1. **Stage 1 — Success SFT**
   Fine-tune on complete successful Playpen trajectories to teach valid participation, turn structure, action formatting, and stopping behaviour.

2. **Stage 2 — Value-weighted turn-level SFT**
   Factorise successful trajectories into local state-action examples and train directly on the next-action prediction problem.

3. **Stage 3 — Teacher-guided repair**
   Use a stronger teacher model to score selected examples and generate realistic invalid near-misses for repair training. The teacher never creates new gold actions; all training targets remain grounded in successful Playpen trajectories.

---

## Results

### Public Playpen validation

| Model / Method           | Avg. % Played | Avg. Quality | Clemscore | Statscore |
| ------------------------ | ------------: | -----------: | --------: | --------: |
| Qwen3.5-2B official base |             – |            – |     13.05 |     44.02 |
| Qwen3.5-2B local base    |         28.57 |        41.70 |     11.91 |         – |
| Stage 1: Success SFT     |         85.60 |        53.46 |     45.76 |     39.74 |
| Stage 2: Weighted-turn   |         83.77 |        63.65 |     53.32 |     41.61 |
| **Qwen-GuidePlay-2B**    |     **83.30** |    **68.57** | **57.12** | **42.68** |

Compared with the official Qwen3.5-2B base score, the final model improves Playpen clemscore by **+44.07** on the public validation setup.

The progression across stages shows a clear division of labour:

* **Stage 1** primarily improves playability.
* **Stage 2** primarily improves decision quality.
* **Stage 3** further improves decision precision while maintaining high playability.

### Official LM Playschool Challenge results

In the official challenge evaluation, Qwen-GuidePlay-2B achieves:

* **+35.99 Playpen clemscore** over the Qwen3.5-2B base model
* **Second-highest Playpen clemscore improvement** among submitted systems
* **+32.85 clemscore** on the held-out in-domain evaluation
* **+6.53 clemscore** on the held-out out-of-domain evaluation
* **−1.94 Playpen statscore** relative to the base model

---

## Method

### Stage 1: Success SFT

We begin with successful trajectories from the `colab-potsdam/playpen-data` interaction training split.

Only episodes with:

```text
Outcome == "success"
```

are retained.

The model is trained on complete successful dialogue-game trajectories using standard causal language-model loss, with sequences truncated to a maximum of **1,024 tokens**.

This stage is intended to teach the model the interactional grammar of Playpen before focusing on local decision quality.

---

### Stage 2: Value-weighted turn-level SFT

Each successful trajectory is factorised into local state-action examples:

```text
prompt     = dialogue/game history before the assistant action
completion = next assistant action
```

Each completion receives a lightweight heuristic value:

```text
v = 0.70
  + 0.15 * I[1 <= len(c) <= 400]
  + 0.05 * I[no stop-token leakage]
  + 0.05 * I[no role-marker leakage]
  + 0.05 * I[len(c) > 0]
```

with the value capped at `1.0`.

Training uses completion-only weighted SFT:

```text
L = sum_i(w_i * l_i) / sum_i(w_i)
```

where `l_i` is the mean completion-token loss and `w_i` is the example value.

---

### Stage 3: Teacher-guided repair

The final stage keeps Stage 2 examples dominant while adding a small amount of teacher-guided supervision.

A stronger teacher model, **Gemma-4-31B-it**, is used to:

* judge selected Stage 2 examples;
* adjust their training weights;
* generate realistic invalid near-misses for repair training.

The teacher **does not invent new gold actions**.

For repair examples, the input contains an invalid version of an existing action, while the supervised target remains the original gold action from a successful Playpen trajectory.

Final training mixture:

| Component                           |      Count |    Share |
| ----------------------------------- | ---------: | -------: |
| Original Stage 2 weighted-turn rows |     30,000 |    96.5% |
| Teacher-judged/reweighted rows      |      1,000 |     3.2% |
| Teacher-generated repair rows       |        100 |     0.3% |
| **Total**                           | **31,100** | **100%** |

---

## Data

All training supervision is derived from the:

```text
colab-potsdam/playpen-data
```

interaction training split.

The training data contains:

* **20,202 successful trajectories**
* **16 games**
* **105,972 factorised turn-level examples**

Validation data is used only for public evaluation and model selection.

No private evaluation data is used to construct training examples.

---

## Training configuration

| Setting              | Stage 1              | Stage 2         | Final                        |
| -------------------- | -------------------- | --------------- | ---------------------------- |
| Initialisation       | Base + new LoRA      | Stage 1 adapter | Stage 1 adapter              |
| Data                 | Successful dialogues | ~106k turn rows | 31,100-row mixture           |
| Loss tokens          | All                  | Completion-only | Completion-only              |
| Example weights      | None                 | Heuristic       | Heuristic / teacher / repair |
| Learning rate        | 2e-4                 | 5e-5            | 5e-5                         |
| Epochs               | 1.0                  | 0.25            | 0.25                         |
| Effective batch size | 8                    | 16              | 8–16                         |

LoRA configuration:

```text
rank = 16
alpha = 32
dropout = 0.05
target_modules = "all-linear"
```

Training uses:

* PyTorch AdamW
* cosine learning-rate schedule
* warmup ratio `0.03`
* bfloat16 precision
* no weight quantisation

---

## Model

The final model is released as a **fully merged Hugging Face checkpoint**, rather than only as a LoRA adapter.

🤗 **Hugging Face:**
https://huggingface.co/syedhuq/qwen-guideplay-2b

---

## Evaluation

Evaluation uses **Playpen 3.7.0** and the public Playpen validation pipeline.

The standard evaluation command is:

```bash
playpen eval <model> --suite all
```

For local Hugging Face evaluation, the model uses:

```text
left padding
eos_to_cull=<|im_end|>
```

The merged model's `generation_config.json` uses:

```json
{
  "eos_token_id": [248046, 248044],
  "pad_token_id": 248044
}
```

All reported experimental results correspond to single training runs rather than averages across multiple random seeds.

---

## Reproducing the experiments

The overall reproduction pipeline is:

```text
Playpen training data
        ↓
successful trajectory filtering
        ↓
Stage 1: Success SFT
        ↓
trajectory factorisation
        ↓
Stage 2: Value-weighted turn-level SFT
        ↓
teacher judging + repair generation
        ↓
Stage 3: Teacher-guided SFT
        ↓
merge LoRA adapter
        ↓
Playpen evaluation
```

Exact commands should be added here according to the scripts included in this repository.

---

## Ablations

We also evaluated several alternative training strategies:

| Method                            | Clemscore | Statscore |
| --------------------------------- | --------: | --------: |
| Stage 2 Weighted-turn             |     53.32 |     41.61 |
| Stage 2 Weighted-turn, 0.50 epoch |     53.07 |     40.62 |
| FAIPD-RR                          |     54.86 |     38.66 |
| FAIPD-Q                           |     51.06 |     40.84 |
| HEM-Mix                           |     51.41 |     40.60 |
| **Qwen-GuidePlay-2B**             | **57.12** | **42.68** |



This work uses the Playpen dialogue-game benchmark and builds on Qwen3.5-2B as the base model.
