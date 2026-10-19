# P3 Sweep #6 — stale_obs recovery training (NET-NEGATIVE)

**Date:** 2026-10-17 → 2026-10-18
**Branch:** `stale-obs` (commit `02720c7`)
**Result:** negative on both axes. Explicit stale_obs recovery training degraded
standard performance *and* stale_obs recovery itself.

## TL;DR

| Metric | p3-formfix (102 clean) | p3-staleobs (84 clean + 55 stale_obs) | Δ |
|---|---|---|---|
| Normal benchmark (51) | 48/51 | **27/51** | **−21** |
| Held-out (10) | 9/10 | **7/10** | −2 |
| stale_obs recovery | 24/26 (92%) | **26/44 (59%)** | **−33 pt** |

- **Search collapsed 8/9 → 0/9** — diagnosed as a `CLICK` loop (model emits
  `CLICK(ref=4)` five times, never `TYPE`s into the box).
- **Recovery got *worse*** (92% → 59%) — the training hurt the exact capability
  it was designed to add. This mirrors the P2 finding that a clean-only student
  already recovers via natural recovery examples; explicit recovery data added
  noise, not signal.

## Hypothesis

Adding `stale_obs` (state-changing) recovery trajectories to the SFT set would
teach the sub-1B student to recover when its observation goes stale (an element
is removed and refs shift). This is the last "recovery" variable in the P3
single-variable sweep series, and the first *state-changing* injection type
(every prior injection — `tool_error` / `missing_element` / `timeout` — was
signal-only, leaving the page untouched).

## Training config (identical to p3-formfix except where noted)

| Field | Value |
|---|---|
| Base model | Qwen3-0.6B (base, no instruct) |
| Method | LoRA SFT, `fine_tune_type=lora` |
| LoRA | rank 8, 16 layers, dropout 0.0, scale 20.0 |
| Optimizer | adam |
| lr | 1e-4 |
| batch size | 4 |
| iterations | 200 |
| seed | 0 |
| `mask_prompt` | true (assistant-token-only loss) |
| `max_seq_length` | 2048 |
| `valid_frac` | 0.1 |
| thinking | OFF (`NoThinkChatDataset`, `enable_thinking=False`) |
| `grad_checkpoint` | **true** (formfix was false — benign infra fix for Metal "Impacting Interactivity" timeouts) |
| train data | `data/dataset_v1_staleobs.jsonl` |

**Split:** 139 examples → 13 valid / 126 train (valid_frac 0.1).
**val loss:** 0.450 (formfix-era healthy runs were ~0.14–0.26 — this run is
noticeably noisier/less converged).

## Dataset manifest — `data/dataset_v1_staleobs.jsonl`

139 examples = **84 clean + 55 stale_obs**.

The raw collection produced 84 clean + 75 stale_obs = 159; **20 stale_obs
examples were dropped as duplicates** of the clean set (the step-1 pre-injection
example of a stale_obs recovery trajectory is byte-identical to the clean
trajectory's step-1 — it carries no recovery signal). 75 − 20 = 55.

Composition of the 55 stale_obs examples:

| Kind | Count | Meaning |
|---|---|---|
| post-injection (recovery signal) | 21 | observation contains `ERROR: CLICK error: stale observation`; assistant learns to re-click with the shifted ref |
| pre-injection (clean-like) | 34 | no error in observation; these are effectively extra clean examples |

So only **21 of 139 examples (15%)** carry the actual stale_obs recovery signal.
The rest are clean or clean-like.

Clean portion by category: form 33 / multi_step 24 / navigation 11 / search 16
(41 unique tasks). **Critically, form-014..018 are absent** (see Confounds).

Per-task example counts are heavily imbalanced: form-001 = 9, multi-013 = 9,
form-002/008/012 = 8, multi-001 = 8 — but search-001..005 and search-007 = 1
each.

## Results

### Normal benchmark — 51 tasks (`experiments/p3_staleobs.csv`)

| Category | p3-formfix | p3-staleobs |
|---|---|---|
| navigation | 10/10 | 10/10 |
| search | 8/9 | **0/9** |
| form | 18/19 | 12/19 |
| multi_step | 12/13 | **5/13** |
| **Total** | **48/51** | **27/51** |

### Held-out — 10 tasks (`experiments/p3_staleobs_heldout.csv`)

p3-formfix 9/10 → p3-staleobs **7/10**. New failures: `ho-search-001`,
`ho-search-002` (search collapse transfers to the unseen site); `ho-form-001`
(3-field + textarea) still fails, unchanged.

### Recovery eval — stale_obs injection

Run via `collect.py --adapter-path` (student adapters, not the 4B teacher).
Injection: `stale_obs` on the first `CLICK` of each trajectory.

| Student | injected | recovered | rate |
|---|---|---|---|
| p3-formfix (baseline) | 26 | 24 | **92%** |
| p3-staleobs | 44 | 26 | **59%** |

Note the **different denominators** (26 vs 44): injection triggers on the first
`CLICK`, and the degraded p3-staleobs model `CLICK`s on far more tasks (form /
search tasks that should start with `TYPE`), so it is injected more often and
fails more of those. This is not a clean like-for-like recovery comparison; it
must be read alongside the normal-benchmark collapse.

## Failure analysis

- **Regressions** (formfix passed → staleobs failed), **22 tasks**:
  - form (7): form-001, form-010, form-014, form-016, form-017, form-018, form-019
  - multi (7): multi-004, multi-005, multi-007, multi-008, multi-010, multi-011, multi-012
  - search (8): search-001..005, search-007..009
- **Fixed** (1): form-008 (the "fill username+password then submit" login — the
  value-splitting gap that P3 sweep #5's form-state also closed; likely spurious,
  not a real capability gain).
- **Both fail** (2): multi-002, search-006.

**Search collapse mechanism** (re-run search-001/002/006 with the p3-staleobs
adapter): the model emits `CLICK(ref=4)` (the search box) on all 5 allowed
steps, never `TYPE`. Valid-but-wrong — `tool_errors=0`. It locates the box but
"forgets" to type into it. Same `CLICK`-loop signature seen in P3 sweep #3
(additive observation format).

## Confounds — this is NOT a clean single-variable test

1. **Clean portion shrank and changed.** p3-formfix was trained on **102 clean**
   (form 51); p3-staleobs on **84 clean** (form 33). The 18 missing clean
   examples are exactly **form-014..018** — the specific-value form tasks added
   in P3 sweep #1. The teacher failed to recover from stale_obs on those tasks,
   so their trajectories were excluded by the "only-successful-trajectories-emit
   SFT" gate. The regression therefore confounds two effects: (a) the stale_obs
   examples, and (b) the loss of the form-014..018 clean data.
2. **Imbalanced coverage.** A few tasks contribute 8–9 examples; most search and
   nav tasks contribute 1. This is not the uniform coverage of earlier datasets.
3. **`grad_checkpoint` differs** (true vs false) — benign, but noted for
   completeness.

A clean **control** (84-clean-only, same hyperparams, same eval protocol) was
run — see below; it **resolved** this confound.

## Control experiment (2026-10-18) — confound RESOLVED

> P3 Sweep #6 — Negative result with controlled data ablation. The initial
> stale_obs experiment regressed from 48/51 to 27/51 on the normal benchmark and
> from 24/26 to 14/26 on the common recovery evaluation set. Follow-up controls
> showed that reducing the clean dataset from 102 to 84 examples accounted for
> only a small performance decrease (48/51 to 45/51), while adding 34 clean-like
> examples improved performance to 49/51. Adding the 21 post-injection recovery
> examples then reduced performance to 27/51. The regression is therefore
> associated with the recovery-example addition, not primarily with the missing
> clean form examples. The search collapse and repeated CLICK behavior suggest
> an action-distribution interference mechanism, which remains a hypothesis
> rather than a proven causal mechanism. The stale_obs training intervention is
> rejected in its current form.

**Note on the first control runs (invalidated):** the first A/B1 trainings
copied `--num-layers -1` from train.py's stale docstring. `-1` applies LoRA to
*all* 28 layers (Qwen3-0.6B), whereas every training in this project uses
`--num-layers 16`. Those runs were invalidated; **all corrected results below
come from `--num-layers 16` trainings.**

### Results (normal benchmark, 51 tasks, all `--num-layers 16`)

| Experiment | Dataset | Normal |
|---|---|---|
| P3 formfix | 102 clean | 48/51 |
| A — clean-only | 84 clean | 45/51 |
| B1 — clean-like | 84 + 34 clean-like | 49/51 |
| B2 — full stale_obs | 84 + 34 clean-like + 21 recovery | 27/51 |

### Findings

1. **Reducing clean data is not the main cause.** 102→84 examples lowered
   48/51 → 45/51 (−3) — a regression, but it does not explain B2's collapse.
2. **The 34 clean-like examples helped.** Adding them raised 45/51 → 49/51
   (+4).
3. **Adding the 21 recovery examples caused the collapse.** B1 → B2 dropped
   49/51 → 27/51 (−22). Search collapsed 9/9 → 0/9, multi 12/13 → 5/13.

The 21 recovery examples are 71% CLICK (15/21) vs 56% for the 34 clean-like
examples (19/34), and B2 loops CLICK on search tasks — supporting the hypothesis
that the recovery examples shifted the model's action distribution toward CLICK.
This is a **hypothesis, not a proven causal mechanism**; B2's training dynamics
and data distribution may also contribute.

## Next step

The stale_obs recovery intervention is **rejected in its current form**. Move to
the quantization sweep: hold the p3-formfix adapter fixed and compare FP16 /
8-bit / 4-bit on success + model size + memory + inference latency.
