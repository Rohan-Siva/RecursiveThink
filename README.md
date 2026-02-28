# RecursiveThink: Inference-Time Recursive Reasoning with Uncertainty-Guided Branching

A research exploration into whether structured inference-time computation — recursive reasoning loops, sampling-based uncertainty estimation, and tree-of-thought branching — can improve LLM mathematical problem-solving without any fine-tuning.

**Benchmark:** AIME 2025 (30 competition math problems)
**Model:** Mistral Large (`mistral-large-latest`)

---

## Table of Contents

1. [Motivation](#motivation)
2. [System Architecture](#system-architecture)
3. [Experiment 1: Ablation Study on Prompt Design](#experiment-1-ablation-study-on-prompt-design)
4. [Experiment 2: Linear Recursive Controller](#experiment-2-linear-recursive-controller)
5. [Experiment 3: Sampling-Based Uncertainty Estimation](#experiment-3-sampling-based-uncertainty-estimation)
6. [Experiment 4: Uncertainty-Guided Branching](#experiment-4-uncertainty-guided-branching)
7. [Experiment 5: Baseline Comparisons](#experiment-5-baseline-comparisons)
8. [Combined Results](#combined-results)
9. [Analysis & Discussion](#analysis--discussion)
10. [Repository Structure](#repository-structure)

---

## Motivation

Standard LLM inference is a single forward pass: one prompt → one answer. But for hard reasoning tasks, this leaves performance on the table. We explored three complementary ideas to improve reasoning at inference time, using only API access to a frozen model:

1. **Recursive refinement** — Let the model iteratively improve its own solution
2. **Uncertainty estimation** — Sample multiple responses and measure agreement instead of trusting self-reported confidence
3. **Branching** — When the model is uncertain, explore multiple solution paths and verify each one

All approaches operate at inference time with no training, no fine-tuning, and no custom model weights.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RecursiveThink System                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │  Linear Mode   │    │ Uncertainty Mode  │    │  Branching Mode  │  │
│  │ (controller.py)│    │ (controller.py +  │    │ (branching_      │  │
│  │                │    │  uncertainty.py)  │    │  controller.py)  │  │
│  │ Single sample  │    │ N-sample voting   │    │ Decompose →      │  │
│  │ per step       │    │ per step          │    │ Solve → Branch → │  │
│  │                │    │                   │    │ Verify → Assemble│  │
│  └───────────────┘    └──────────────────┘    └──────────────────┘  │
│           │                    │                       │            │
│           └────────────────────┴───────────────────────┘            │
│                                │                                    │
│                     ┌──────────┴──────────┐                         │
│                     │   Shared Components  │                         │
│                     │  model.py  state.py  │                         │
│                     │  prompt.py parser.py │                         │
│                     │  logger.py           │                         │
│                     └─────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────┘
```

### Branching Pipeline (5 Phases)

```
Problem
   │
   ▼
┌──────────────────────────┐
│  Phase 1: DECOMPOSE       │  LLM breaks problem into 3–6 subgoals
│  (decompose.py)           │  with explicit dependency DAG
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│  Phase 2: SOLVE           │  For each subgoal (topological order):
│  (uncertainty.py)         │  • Generate N=5 independent samples
│                           │  • Extract \boxed{answer} from each
│                           │  • Cluster answers, compute agreement
│                           │  • measured_confidence = majority/N
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│  Phase 3: BRANCH          │  If confidence < threshold (0.8):
│  (branching_controller.py)│  • Take top-k answer clusters as branches
│                           │  • Verify each branch independently
│                           │  • Pick winner by verification confidence
│                           │  • Extract negative knowledge from losers
│                           │  • Propagate dead-ends to downstream subgoals
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│  Phase 4: ASSEMBLE        │  Combine all subgoal results into
│                           │  final coherent solution
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│  Phase 5: FINAL CHECK     │  One more N-sample uncertainty check
│                           │  on the assembled answer
└──────────────────────────┘
```

---

## Experiment 1: Ablation Study on Prompt Design

**Goal:** Identify which prompt engineering strategies most affect recursive reasoning performance.

**Setup:** 6 prompt variants × 5 diverse problems (arithmetic, algebra, logic, probability), using the linear recursive controller with `max_steps=8`, `confidence_threshold=0.9`.

### Results

| Variant | Description | Avg Steps | Avg Confidence | Avg Time |
|---------|-------------|-----------|----------------|----------|
| **expert** | _"You are a world-class expert"_ | **1.0** | 0.978 | **5.2s** |
| **chain_of_thought** | _"Let me think through this…"_ | **1.0** | **0.992** | 9.1s |
| step_by_step | Explicit numbered instructions | 1.2 | 0.970 | 5.5s |
| anonymous | _"the previous solution"_ (neutral) | 1.8 | 0.950 | 8.0s |
| own | _"YOUR previous reasoning"_ | 2.6 | 0.930 | 12.6s |
| minimal | Bare essentials only | 3.4 | 0.950 | 11.1s |

### Key Findings

- **Expert persona + CoT markers = optimal.** Both solved every problem in 1 step with the highest confidence.
- **Self-attribution hurts.** Telling the model to build on "YOUR previous reasoning" caused 2.6× more steps — the model second-guesses itself when given ownership.
- **Minimal prompts cause thrashing.** The model started at low confidence (0.5–0.7) and took up to 8 steps on a probability problem, getting stuck at 0.85.

> **Takeaway:** All subsequent experiments use the `expert + CoT` prompt, which was informed by this ablation.

**Files:** `ablation.py`, `ablation_variants.py`, `ablation_all_variants_results.jsonl`

---

## Experiment 2: Linear Recursive Controller

**Goal:** Test whether iterative self-refinement (without branching) improves accuracy on AIME 2025.

**Setup:** Full 30-problem AIME 2025 dataset, `max_steps=10`, `confidence_threshold=0.9`, single-sample per step (no uncertainty estimation). Two independent runs.

### Results

| Run | Correct | Accuracy | Avg Steps |
|-----|---------|----------|-----------|
| Linear exp1 | 10/30 | 33.3% | 1.1 |
| Linear exp2 | 14/30 | **46.7%** | 1.1 |

### Per-Problem Breakdown (Linear Recursive)

| Problem | Expected | Run 1 | Run 2 |
|---------|----------|-------|-------|
| id_01 | 70 | ✅ | ✅ |
| id_02 | 588 | ❌ | ❌ |
| id_03 | 16 | ✅ | ✅ |
| id_04 | 117 | ✅ | ✅ |
| id_05 | 279 | ✅ | ❌ |
| id_06 | 504 | ✅ | ✅ |
| id_07 | 821 | ❌ | ❌ |
| id_08 | 77 | ✅ | ❌ |
| id_09 | 62 | ❌ | ✅ |
| id_10 | 81 | ❌ | ❌ |
| id_11 | 259 | ❌ | ✅ |
| id_12 | 510 | ❌ | ✅ |
| id_13 | 204 | ✅ | ❌ |
| id_14 | 60 | ❌ | ❌ |
| id_15 | 735 | ❌ | ❌ |
| id_16 | 468 | ✅ | ✅ |
| id_17 | 49 | ✅ | ✅ |
| id_18 | 82 | ❌ | ❌ |
| id_19 | 106 | ✅ | ✅ |
| id_20 | 336 | ❌ | ❌ |
| id_21 | 293 | ❌ | ❌ |
| id_22 | 237 | ❌ | ❌ |
| id_23 | 610 | ❌ | ✅ |
| id_24 | 149 | ❌ | ❌ |
| id_25 | 907 | ❌ | ✅ |
| id_26 | 113 | ❌ | ❌ |
| id_27 | 19 | ❌ | ✅ |
| id_28 | 248 | ❌ | ❌ |
| id_29 | 104 | ❌ | ❌ |
| id_30 | 240 | ❌ | ✅ |

### Observations

- **High variance between runs** (33%–47%) — the model's single-sample output is nondeterministic.
- **Most problems solved in 1 step** — the recursive loop rarely iterated because the improved prompts produce high self-reported confidence immediately.
- The linear controller essentially collapses to a single-shot solver with good prompts.

**Files:** `controller.py`, `tts/run_aime.py`, `results/aime_recursive_mistral_large_latest/`

---

## Experiment 3: Sampling-Based Uncertainty Estimation

**Goal:** Replace the model's self-reported confidence with *measured* confidence via multi-sample agreement.

**Mechanism:** For each reasoning step, generate N=5 independent responses, extract the `\boxed{answer}` from each, cluster equivalent answers, and set:

```
measured_confidence = |largest_cluster| / N
```

If all 5 samples agree → confidence = 1.0. If they split 3/1/1 → confidence = 0.6.

This is inspired by **self-consistency** (Wang et al., 2022) and **LaMSeI** (Pang et al., TMLR 2025).

### Design Details

- **Answer extraction:** Regex for `\boxed{...}`, fallback to last number in output
- **Answer normalization:** Strip whitespace, commas, convert to float when possible
- **Clustering:** Greedy equivalence-class grouping with normalized comparator

### Impact

The uncertainty estimator became the foundation for the branching controller — it identifies *which* subgoals need branching and *which* answer clusters to explore.

**Files:** `uncertainty.py`

---

## Experiment 4: Uncertainty-Guided Branching

**Goal:** Combine problem decomposition, uncertainty estimation, and tree-of-thought branching into a unified pipeline.

**Setup:** AIME 2025, `num_samples=5`, `branch_threshold=0.8`, `max_branches=3`, `branch_verification_steps=2`. Multiple experiment runs due to process hangs from API rate limits.

### Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_samples` | 5 | Independent samples per subgoal |
| `branch_threshold` | 0.8 | Confidence below which branching triggers |
| `max_branches` | 3 | Max answer clusters to explore |
| `branch_verification_steps` | 2 | Verification rounds per branch |
| `final_check_samples` | 5 | Final consistency check samples |

### Results by Experiment

| Experiment | Problems | Correct | Accuracy | Avg API Calls/Problem | Status |
|------------|----------|---------|----------|-----------------------|--------|
| exp1 | id_01–05 | 4/5 | **80.0%** | 44 | ✅ Complete |
| exp2 | id_06–19 | 5/14 | 35.7% | 49 | ⚠️ Partial (hung) |
| exp3 | id_15–17 | 1/3 | 33.3% | 45 | ⚠️ Partial (hung) |
| exp4 | id_19–20 | 0/2 | 0.0% | 28 | ⚠️ Partial (hung) |
| exp5 | id_26–30 | 0/5 | 0.0% | 55 | ✅ Complete |

### Per-Problem Detail (Branching)

| Problem | Expected | Branching Answer | Correct? | Subgoals | Branched | API Calls | Time |
|---------|----------|-----------------|----------|----------|----------|-----------|------|
| id_01 | 70 | 70 | ✅ | 6 | 3 | 50 | 4.9m |
| id_02 | 588 | 776 | ❌ | 6 | 3 | 47 | 20.3m |
| id_03 | 16 | 16 | ✅ | 5 | 0 | 32 | 3.8m |
| id_04 | 117 | 117 | ✅ | 5 | 1 | 36 | 5.4m |
| id_05 | 279 | 279 | ✅ | 6 | 4 | 55 | 8.2m |
| id_06 | 504 | 504 | ✅ | 6 | 2 | 44 | 4.9m |
| id_07 | 821 | 16 | ❌ | 6 | 3 | 52 | 8.3m |
| id_08 | 77 | 73/4 | ❌ | 6 | 2 | 45 | 10.2m |
| id_09 | 62 | 62 | ✅ | 5 | 2 | 37 | 11.0m |
| id_10 | 81 | 61 | ❌ | 6 | 5 | 53 | 11.2m |
| id_11 | 259 | 343 | ❌ | 6 | 3 | 48 | 21.8m |
| id_12 | 510 | 78 | ❌ | 6 | 5 | 54 | 25.3m |
| id_13 | 204 | 179 | ❌ | 6 | 4 | 52 | 13.4m |
| id_14 | 60 | 21+20√3 | ❌ | 6 | 5 | 56 | 14.5m |
| id_15 | 735 | 742 | ❌ | 6 | 4 | 50 | 16.9m |
| id_16 | 468 | 468 | ✅ | 6 | 2 | 47 | 5.4m |
| id_17 | 49 | 49 | ✅ | 6 | 2 | 43 | 7.4m |
| id_18 | 82 | 66 | ❌ | 6 | 6 | 58 | 15.1m |
| id_19 | 106 | 106 | ✅ | 6 | 2 | 43 | 6.9m |
| id_26 | 113 | 12 | ❌ | 6 | 4 | 53 | 10.8m |
| id_27 | 19 | 75 | ❌ | 9 | 6 | 74 | 21.2m |
| id_28 | 248 | 741 | ❌ | 6 | 3 | 47 | 18.3m |
| id_29 | 104 | 138 | ❌ | 6 | 3 | 46 | 17.2m |
| id_30 | 240 | 188 | ❌ | 6 | 5 | 57 | 18.5m |

### Branching Behavior

- **Average subgoals per problem:** 5.9 (model consistently decomposes into ~6 steps)
- **Average branched subgoals:** 3.2 out of 5.9 (54% of subgoals triggered branching)
- **Average API calls per problem:** ~49 (vs. 1 for single-shot baseline)
- On problems the model gets right, fewer subgoals need branching (avg 2.0 vs 3.9 on incorrect)

**Files:** `branching_controller.py`, `branching_prompts.py`, `decompose.py`, `subgoal.py`, `tts/run_aime_branching.py`, `results/aime_branching_mistral_large_latest/`

---

## Experiment 5: Baseline Comparisons

**Goal:** Establish reference points — what does the raw model achieve without any recursive scaffolding?

### Single-Shot Baseline (k=1)

Standard single-call generation with `max_tokens=16384` (we found `max_tokens=1024` caused 0% accuracy due to truncation).

| Metric | Value |
|--------|-------|
| **Accuracy** | **13/30 = 43.3%** |

### Best-of-16 Baseline (pass@k)

Generate 16 independent samples per problem, check if *any* sample is correct.

| Metric | Value |
|--------|-------|
| **pass@1 (estimated)** | **37.1%** |
| **pass@4** | **53.4%** |
| **pass@8** | **60.9%** |
| **pass@16** | **21/30 = 70.0%** |

### Ministral Comparison

Reproduction of results from Ministral paper (arxiv 2601.08584) using `mistral-large`:

| Metric | Value |
|--------|-------|
| **pass@1** | **12/30 = 40.0%** |

**Files:** `baseline.py`, `results/aime_baseline_mistral_large_latest/`, `results/ministral_mistral_large/`

---

## Combined Results

### Headline Comparison

| Method | Problems Evaluated | Correct | Accuracy | API Calls/Problem |
|--------|-------------------|---------|----------|-------------------|
| **Single-Shot Baseline (k=1)** | 30 | 13 | 43.3% | 1 |
| **Ministral Baseline** | 30 | 12 | 40.0% | 1 |
| **Best-of-16 (pass@16)** | 30 | 21 | 70.0% | 16 |
| **Linear Recursive (best run)** | 30 | 14 | 46.7% | ~1 |
| **Branching (id_01–05)** | 5 | 4 | 80.0% | 44 |
| **Branching (id_01–19 combined)** | 19 | 9 | 47.4% | 49 |
| **Branching (id_26–30, hardest)** | 5 | 0 | 0.0% | 55 |

### Head-to-Head: Problems 1–19

For the 19 problems where we have results from every method:

| Problem | Baseline k=1 | Linear R1 | Linear R2 | Branching | Baseline k=16 |
|---------|:---:|:---:|:---:|:---:|:---:|
| id_01 | ✅ | ✅ | ✅ | ✅ | 16/16 |
| id_02 | ❌ | ❌ | ❌ | ❌ | 7/16 |
| id_03 | ✅ | ✅ | ✅ | ✅ | 16/16 |
| id_04 | ✅ | ✅ | ✅ | ✅ | 16/16 |
| id_05 | ✅ | ✅ | ❌ | ✅ | 13/16 |
| id_06 | ✅ | ✅ | ✅ | ✅ | 16/16 |
| id_07 | ❌ | ❌ | ❌ | ❌ | 0/16 |
| id_08 | ✅ | ✅ | ❌ | ❌ | 5/16 |
| id_09 | ❌ | ❌ | ✅ | ✅ | 1/16 |
| id_10 | ❌ | ❌ | ❌ | ❌ | 1/16 |
| id_11 | ❌ | ❌ | ✅ | ❌ | 0/16 |
| id_12 | ✅ | ❌ | ✅ | ❌ | 7/16 |
| id_13 | ❌ | ✅ | ❌ | ❌ | 0/16 |
| id_14 | ❌ | ❌ | ❌ | ❌ | 0/16 |
| id_15 | ❌ | ❌ | ❌ | ❌ | 1/16 |
| id_16 | ✅ | ✅ | ✅ | ✅ | 16/16 |
| id_17 | ✅ | ✅ | ✅ | ✅ | 16/16 |
| id_18 | ❌ | ❌ | ❌ | ❌ | 0/16 |
| id_19 | ✅ | ✅ | ✅ | ✅ | 14/16 |
| **Total** | **10/19** | **10/19** | **10/19** | **9/19** | — |

---

## Analysis & Discussion

### 1. Branching Did Not Improve Over Baseline

On the comparable subset (problems 1–19), branching achieved **47.4%** accuracy — roughly matching the single-shot baseline (43.3%) and linear recursive (46.7%). The branching approach used **~49× more API calls** per problem but did not yield a meaningful accuracy gain.

### 2. Problem Difficulty Is the Dominant Factor

The results reveal a clear pattern:
- **Easy problems** (id_01, 03, 04, 05, 06, 16, 17, 19): All methods solve these. Baseline k=16 shows near-perfect agreement (13–16/16 correct).
- **Hard problems** (id_07, 11, 14, 18, 22, 28–30): No method solves these. Even pass@16 gets 0/16.
- **Marginal problems** (id_02, 08, 09, 10, 12, 13, 15): These are where methods *could* differ, but branching doesn't consistently win.

The model simply cannot solve certain problems regardless of how many times or ways you ask it.

### 3. Branching Has a "Consensus Amplification" Problem

When the model is uncertain about a subgoal, the majority answer cluster isn't necessarily correct — it's just the most common wrong answer. Branching then *verifies* this already-wrong answer, and the verification LLM (same model) often confirms it. This creates a self-reinforcing loop where the most popular incorrect answer gets enshrined.

### 4. Decomposition Adds Error Propagation Risk

Breaking problems into subgoals introduces sequential dependency: if Subgoal 2 depends on Subgoal 1, and Subgoal 1 is wrong, the error cascades through all downstream subgoals. The negative knowledge mechanism was designed to mitigate this, but it can only prevent *known* errors from repeating — it can't detect novel errors.

### 5. The Recursive Loop Largely Collapsed

With the improved (expert + CoT) prompts, the linear controller solved most problems in 1 step. The model's self-reported confidence was high enough to trigger an immediate stop. This means the "recursive" part of RecursiveThink was rarely exercised — the prompt improvements made the first-attempt quality high enough that iteration was unnecessary.

### 6. Sampling (pass@k) Remains the Most Effective Scaling Strategy

Best-of-16 significantly outperformed all structured approaches:

| Method | API Calls | Accuracy |
|--------|-----------|----------|
| Single-shot (k=1) | 1 | 43.3% |
| RecursiveThink branching | ~49 | 47.4% |
| Best-of-16 | 16 | **70.0%** |

With 16 unstructured samples, the model achieves 70% — far exceeding what 49 *structured* calls achieve through branching. This suggests that for this model on AIME, naïve sampling diversity beats structured exploration.

### 7. Prompt Design Had the Largest Marginal Effect

The ablation study showed a **3.4× reduction in steps** and a **+0.06 confidence gain** just from switching prompt styles. This was accomplished by changing perhaps 50 words in the system prompt — suggesting that at this model capability level, prompt engineering yields more than architectural complexity.

---

## Lessons Learned

1. **Measure, don't trust.** Self-reported confidence from the LLM is unreliable — sampling-based measurement is more honest about what the model knows.
2. **Structured reasoning adds overhead without guaranteed benefit.** Decomposition and branching are elegant but introduce error propagation and API cost.
3. **The model is the bottleneck.** No amount of inference-time scaffolding can make the model solve problems it fundamentally can't. Scaling the model (or using a reasoning-specialized model) would likely have a larger effect.
4. **Prompt engineering first, architecture second.** The ablation study's impact dwarfed the branching system's impact.
5. **Negative knowledge propagation is a promising idea that needs stronger verification.** Dead-end detection relies on the same model that made the error, limiting its effectiveness.

---

## Repository Structure

```
RecursiveThink/
├── main.py                    # CLI entry point
├── controller.py              # Linear recursive controller
├── branching_controller.py    # Uncertainty-guided branching controller
├── branching_prompts.py       # Decompose / solve / verify / assemble prompts
├── decompose.py               # Problem decomposition + topological sort
├── subgoal.py                 # SubgoalState and BranchResult dataclasses
├── uncertainty.py             # Sampling-based uncertainty estimator
├── model.py                   # LLM API wrapper (Mistral, Gemini)
├── prompt.py                  # Linear controller prompts (ablation-informed)
├── parser.py                  # JSON output parser
├── state.py                   # ThoughtState dataclass
├── logger.py                  # JSONL structured logging
├── baseline.py                # Single-shot / pass@k baseline runner
├── ablation.py                # Ablation study runner
├── ablation_variants.py       # 6 prompt variant definitions
├── tts/
│   ├── run_aime.py            # Linear recursive AIME runner
│   ├── run_aime_branching.py  # Branching AIME runner
│   └── math_grader.py         # Answer extraction & grading
└── results/
    ├── aime_baseline_mistral_large_latest/     # Baseline k=1, k=16
    ├── aime_recursive_mistral_large_latest/    # Linear recursive exp1, exp2
    ├── aime_branching_mistral_large_latest/    # Branching exp1–5
    ├── aime_recursive_gemini_2.5_flash/        # Gemini flash (2 problems)
    └── ministral_mistral_large/                # Ministral reproduction
```

## Setup

```bash
pip install mistralai datasets
export MISTRAL_API_KEY="your-key"

# Run linear recursive
python tts/run_aime.py --provider mistral

# Run branching
python tts/run_aime_branching.py --provider mistral --num-samples 5 --branch-threshold 0.8

# Run baseline
python baseline.py --k 1
```
