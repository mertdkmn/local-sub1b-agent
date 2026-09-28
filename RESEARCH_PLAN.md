# Sub-1B Local Agent — Research Plan

## 1. Core Research Question

> **How far can we push a sub-1B parameter browser agent under strict compute and memory constraints on consumer hardware locally?**

The project is not simply about building a small browser agent. The goal is to systematically investigate how much browser-agent capability can be obtained **per parameter, per MB of memory, and per unit of compute**, and which techniques move that capability frontier.

This is the **unchanged core**. Everything below is the *method* for reaching it, not a replacement of it.

## 2. The Key Conceptual Distinction

Local inference itself is no longer the research problem — a 0.5–1B model can run locally. The open question is whether that small model can remain **reliable** when it must perform many sequential actions, recover from mistakes, operate with limited context, and stay within a strict memory/latency budget.

- A chatbot: one wrong step → one bad answer.
- An agent: errors **accumulate** across a long action sequence. At 95% per-step accuracy, a 20-step task succeeds only ~64% of the time.

Therefore the project centers on **reliability, recovery, and efficiency** — not merely whether a model can run.

## 3. Terminology (locked, to avoid confusion)

| Term | Meaning | In this project |
|---|---|---|
| **Local** | Model runs on-device, no cloud API | ✅ Required |
| **Consumer hardware** | Ordinary hardware (Apple M2 Pro, 16GB), not data-center GPUs | ✅ Required |
| **Offline / no internet** | No network at all | ❌ NOT what we mean |
| **Cloud inference** | Sending thinking to a remote API | ❌ Never |
| **Cloud training** | Using remote GPU to *train* | ⚠️ Optional, not required |

"Local" ≠ "consumer hardware", and "local" ≠ "offline". They are different axes. This project needs **both local AND consumer hardware** at the same time — that is what produces the "strict compute and memory constraint" we measure. And because it is a *browser* agent, **internet access is required** (for the browser to navigate); only the model's *thinking* is local.

## 4. Locked Design Decisions

1. **Fine-tune a pretrained sub-1B Transformer FIRST.** Start with Qwen3-0.6B-Instruct. SAN/Needle-style architecture is a later phase (P4), not a Phase-1 variable.
2. **"Local" = inference/deployment constraint only.** Training may use cloud GPU. Inference runs on Apple Silicon via MLX.
3. **Dataset/trajectory pipeline is the center of P0/P1.** The ~4B teacher (Qwen3-4B) is a *data generator*, not just an oracle.
4. **Teacher and student share the same observation/tool interface** (a11y tree). Otherwise trajectories are out-of-distribution and distillation fails.
5. **Single-variable sweeps**, never combinatorial. 30–50 task pilots before scaling to 100–300.
6. **Student frontier stays strictly sub-1B.** Qwen3-1.7B is excluded (auxiliary/teacher only).
7. **Benchmark uses programmatic oracles**, not an LLM-judge (which is both expensive and circular).

## 5. Why pretrained models (not "doing it differently")

A "language model" is two layers:

1. **Language/world understanding** — what "add to cart", "form", "button" mean; sentence structure; how web pages work. Trained on trillions of tokens on huge clusters. *Expensive and solved.*
2. **Browser-agent behavior** — "given an a11y tree, produce the right `CLICK(ref)`", "after an error, re-observe". A thin layer on top of (1). *This is our research.*

Using a pretrained model gives us layer (1) for free; we research layer (2). "Doing it differently" happens on two levels, in order:

- **Now (P0–P3):** we differ in layer 2 — observation representation, recovery training, action-weighted loss, tool space.
- **Later (P4):** we differ in layer 1 — SAN/Needle-style architecture. But only after we have a Transformer baseline to compare against. You cannot claim "SAN is better" without answering "better than what?"

## 6. Model-size reality: the checkpoint set

Off-the-shelf sub-1B checkpoints come in fixed sizes. There is **no 0.8B checkpoint** — Qwen3 has 0.6B and 1.7B; SmolLM2 has 135M and 360M. So the "model size" variable is really a **checkpoint sweep** over:

```
{ SmolLM2-135M, SmolLM2-360M, Qwen2.5-0.5B, Qwen3-0.6B }
```

Getting an 0.8B model would require pruning (1.7B → 0.8B) or from-scratch training — a later, separate experiment (P4).

## 7. System Flow

```
TASK GENERATOR → Browser (Playwright) → TEACHER (Qwen3-4B, 4-bit)
                                              ↓
                                      TRAJECTORY COLLECTOR
                              ┌───────────────┴───────────────┐
                              ↓                               ↓
                       Successful trace                Failure trace (injected)
                              │                               ↓
                              │                       Recovery trace
                              └───────────────┬───────────────┘
                                              ↓
                                        DATASET v1 (each trace has failure_type)
                                              ↓
                                        STUDENT TRAINING (SFT/LoRA)
                                              ↓
                                       Sub-1B Agent → EVAL HARNESS → experiments.csv → frontier plot
```

### Component glossary

- **Task Generator** — a script that produces ~100 tasks (`task_id`, `goal`, programmatic oracle).
- **Browser (Playwright)** — a real, controllable browser. The model never touches it directly; all actions go through Playwright. *Controllability is what enables failure injection.*
- **Teacher (Qwen3-4B, 4-bit)** — a big model whose job is to produce correct step-by-step actions. It is the data generator.
- **Trajectory Collector** — a script that runs the teacher on tasks and records every (observation → action) step.
- **Trajectory / trace** — the step-by-step record of one task: OBSERVE → ACT → OBSERVE → ACT → … → SUCCESS/FAIL. This record *is* the training data.
- **Dataset v1** — many traces, converted to SFT format (input → expected action), each tagged with `failure_type`.
- **Student Training (SFT)** — fine-tune Qwen3-0.6B on Dataset v1. **Distillation** = big model (teacher) produces data, small model (student) learns from it.
- **Eval Harness** — runs a model across all benchmark tasks and logs ~14 metrics to `experiments.csv`.

## 8. The Agent Loop

```
GOAL
 ↓
OBSERVE  →  compact a11y tree (only interactive elements, NOT full HTML)
 ↓
ACTION   →  one tool call:  CLICK(ref=17)
 ↓
OBSERVE  →  new a11y tree (did the page change?)
 ↓
ACTION   →  TYPE(ref=3, "X")
 ↓
 ... repeat ...
 ↓
SUCCESS / FAIL  (max steps exceeded, wrong result, timeout)
```

Initial tool space (8 tools):

```
CLICK(ref)   TYPE(ref, text)   SCROLL(dir)   GO_BACK()
GOTO(url)    SEARCH(query)     SUBMIT()      WAIT()
```

Each tool references an element in the a11y tree via `ref`. "8 → 12 → 20 → 40 tools" is tested later as a single variable.

## 9. Quantization (for the teacher)

A model's knowledge is stored in billions of **weights** (decimal numbers). More bits per weight = more precise but bigger:

- **FP16** = 16 bits (2 bytes) per weight
- **INT8** = 8 bits (1 byte)
- **INT4** = 4 bits (0.5 byte)

**Quantization** = rounding these numbers to fewer bits, trading a tiny accuracy loss for big memory/speed gains.

For a 4B model (4 billion weights):
- FP16: 4B × 2 bytes = ~8 GB
- INT4: 4B × 0.5 bytes = ~2 GB

The teacher only *produces data*; it is not deployed. So we run it at **4-bit** to fit the 16GB Mac alongside the browser. (Student is ~1.2GB FP16 / ~0.3GB INT4 — trivial.)

## 10. Recovery Training (the most technical part)

The student must learn to **recover from mistakes**. But to teach that, the training data must contain *mistake + correction* examples. A naturally-successful teacher never produces them. So we **inject failures**:

```
Teacher intends:  CLICK(search_button)          ← correct
      ↓
We intercept:      "element not found" / redirect
      ↓
Teacher re-observes → state changed
      ↓
Teacher corrects:  CLICK(search_button)          ← tries again
      ↓
SUCCESS
```

This "failure → re-observe → corrective action → success" sequence is recorded as a **recovery trace**. The student learns "on error, don't panic — re-observe and retry."

### Failure taxonomy (each injected failure is tagged)

```
wrong_target, stale_obs, wrong_arg, tool_error, navigation, popup, timeout, missing_element
```

This enables a stronger research question than "recovery is weak":
> **"Which failure types can a sub-1B model learn to recover from, and which can it not?"**

## 11. Eval Harness & Metrics

The harness measures, per run:

```
success, steps, latency_ms, tokens_per_sec, peak_ram_mb, model_size_mb,
tool_errors, recovery_success
```

Plus run metadata:

```
model_name, param_count, quantization, observation_type, context_length,
tool_count, teacher, training_method, loss, recovery_training,
task_id, task_category, failure_type
```

All of it lands in **one row of `experiments.csv`**. This file is the project's heart — the frontier plot and every variable analysis derive from it.

### Two kinds of tests (do not confuse)

| Kind | Checks | Example |
|---|---|---|
| **Unit tests** | Your *code* works | "Does the CLICK tool actually click the right element?" "Does failure injection trigger the right error?" |
| **Benchmark eval** | The *model's* quality | "Zero-shot 0.6B succeeds on 40 tasks at 40%" |

## 12. Benchmark Design

~100 tasks, categorized:

```
20 Navigation, 20 Search, 20 Form filling, 20 Multi-step, 20 Recovery
```

Each task defined by: `task_id, goal, initial state, expected result, max_steps, difficulty, failure injection points`.

Tasks run against **static/local fixtures or a pinned container**, not live websites (live sites = flaky eval, fatal for recovery experiments). Success is judged by a **programmatic oracle** (final URL / DOM state / form value).

Pilot protocol: every experiment starts with 30–50 tasks; if promising, scale to 100–300.

## 13. Phases (Mac-only, zero budget through P3)

| Phase | Work | Output |
|---|---|---|
| **P0** | Eval harness + ~40-task benchmark + zero-shot Qwen3-0.6B-Instruct baseline | Origin point of the frontier |
| **P1** | Qwen3-4B (4-bit) teacher + trajectory collector + failure injection | Dataset v1 |
| **P2** | Qwen3-0.6B LoRA SFT on Dataset v1 | First result: did distillation move the frontier? |
| **P3** | Single-variable sweeps (observation, loss, tool space, recovery, context, quant) | Causal contribution of each variable |
| **P4** | Checkpoint sweep {135M→600M} + SAN/Needle | Frontier curve + architecture comparison |

**P0–P2 finish on a 16GB M2 Pro at zero cost.** P3 also zero-cost but slow (optional cloud acceleration). P4's from-scratch work needs cloud budget — months later.

## 14. Hardware Reality (16GB M2 Pro)

| Workload | On 16GB M2 Pro? | Cost |
|---|---|---|
| Inference — sub-1B student eval | ✅ Trivial | 0₺ |
| Inference — 4B teacher trajectory gen (4-bit ~2.5GB) | ✅ Comfortable | 0₺ |
| LoRA fine-tune — sub-1B student | ✅ Feasible (~4–6GB), but slow | 0₺ |
| Full fine-tune — 0.6B | ⚠️ Tight, OOM risk | 0₺ |
| From-scratch pretraining (SAN/Needle) | ❌ Not realistic | 💸 cloud |

### Practical locks for 16GB

1. **Student training = LoRA** (avoid full-FT OOM risk).
2. **Teacher = 4-bit** (fit alongside the browser).
3. **Use MLX for both training and inference** (`mlx-lm`); HF Transformers' MPS backend is buggy/slow on Apple Silicon.
4. **Eval one task at a time** (browser-agent eval is naturally sequential — works in our favor).

## 15. Scope Guardrails (what this project is NOT)

1. NOT "build a sub-1B browser agent" → it is "characterize and extend the sub-1B frontier."
2. SAN/Needle is NOT in Phase 1 → baseline first, then comparison.
3. NOT 12 variables at once → single-variable sweeps with pilots.
4. 1.7B is NOT a student → frontier is strictly sub-1B.
5. NOT dependent on live websites → static fixtures, programmatic oracles.
6. Teacher and student MUST share the interface → else distillation is out-of-distribution.
