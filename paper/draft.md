# The Instability of Recovery-Trajectory Supervision in Sub-1B Browser Agents

*Draft v0.4*

## Abstract

Training a small browser agent on its own successful trajectories produces a competent but brittle policy: a single early mistake compounds across the remaining steps because the model has never been shown how to correct course. A widely recommended remedy is to supplement the behavioral cloning objective with recovery trajectories, episodes in which a failure was injected and a teacher model demonstrated the correction. We evaluate this recipe on a 0.6B browser agent using a controlled two by two ablation over the composition of recovery data, which separates error free steps from error containing steps, together with a sweep over the fraction of the training set derived from recovery. Across three to seven seeds per condition, recovery supervision neither reliably improves the agent nor leaves it unchanged. The full recovery condition is bimodal across seeds: roughly two of seven runs collapse to a degenerate policy that issues CLICK actions almost exclusively and never issues SEARCH or TYPE actions, while the remaining runs track the clean baseline. The instability is not monotonic in the recovery fraction. It concentrates in a narrow window near 36 to 40 percent and disappears at both lower and higher fractions. Validation loss fails to rank checkpoints by task success, and a collapsed run can achieve the lowest validation loss, whereas a single action distribution statistic, the fraction of CLICK actions, cleanly separates healthy from collapsed checkpoints. Checkpoint analysis further shows that the collapse is a late training bifurcation rather than an early divergence: at iteration 100 the runs destined to collapse look healthier than several that eventually recover. We conclude that naive recovery supervision for sub-1B agents is not merely ineffective but unstable, and that practitioners should select models for local deployment using multiple seeds and action distribution monitoring rather than validation loss.

## 1 Introduction

Local browser agents, models that run on a laptop without cloud calls, API costs, or data leaving the device, are a natural fit for sub-1B parameter models. A 0.6B model fine-tuned on a teacher's successful trajectories reaches roughly 90 percent on a scoped benchmark, yet the policy it learns remains brittle. When the model commits an early mistake it rarely recovers, and the error propagates through the remaining steps of the episode.

The obvious remedy is to teach recovery directly. One collects recovery trajectories by injecting a failure into an episode, retains the episodes in which a teacher corrected the failure, and appends these episodes to the behavioral cloning objective. The appeal is straightforward: the model learns what to do after an error, not only what to do when everything proceeds as expected.

Our experiments show that this remedy fails in a specific way at the sub-1B scale. Recovery supervision has a small and unreliable effect on average accuracy, but a large effect on variance. A substantial fraction of training seeds collapse to a degenerate policy, and the collapse is invisible to validation loss while remaining clearly visible in the distribution of actions the model emits. We make four contributions.

First, we run a controlled composition ablation that separates error free recovery steps from error containing recovery steps, and we show that the full mixture induces bimodal, seed dependent collapse. Second, we run a data mixing sweep over the recovery derived fraction and find that the effect is non-monotonic: the collapse occupies a narrow instability window rather than growing with the amount of recovery data. Third, we show that validation loss does not rank checkpoints by task success, whereas a single action distribution statistic, the CLICK share, detects the collapse reliably. This is a practical model selection signal for agent fine-tuning. Fourth, we trace the collapse to a late training bifurcation, not an early divergence.

We frame the paper as an empirical characterization rather than a claim of a new phenomenon. Bimodal seed collapse in small model fine-tuning has already been reported in knowledge distillation [Sumit et al., 2026] and vision language action training [2026]; recovery annotated trajectories are deployed at scale by PALADIN [Vuddanti et al., 2025]; and masking the error step is known to help [Chen et al., 2025; SRFT, 2026]. Our contribution is the combination of composition, mixing, and monitoring analysis in the sub-1B browser agent setting, together with the practical recommendation that follows from it.

## 2 Related Work

Recovery and self correction training. PALADIN [Vuddanti et al., 2025] trains on more than fifty thousand recovery annotated trajectories constructed through systematic failure injection and LoRA fine-tuning, and reports substantial gains in recovery rate. STeP [Chen et al., 2025] and Step Rejection Fine-Tuning [SRFT, 2026] mask the error step and train only on the correction, finding that masking helps. These methods operate at a scale and with reasoning channels that we do not use; none of them report seed level instability. We study the small data sub-1B regime in which the effect becomes unreliable.

Bimodal seed collapse. "Below the Noise Floor" [Sumit et al., 2026] documents bimodal seed collapse in small model knowledge distillation, where one in three to two in five seeds falls below 55 percent accuracy while the rest train normally, and distinguishes several collapse modes. Output level regularization [2026] reports the same seed lottery in vision language action fine-tuning. Our work reproduces this phenomenon in a different setting, recovery data supervised fine-tuning for browser tool use, and connects it to a monitoring signal, the action distribution, rather than to a training objective.

SFT, data distribution, and label prior. "SFT Memorizes, RL Generalizes" [2025] argues that supervised fine-tuning imprints the training distribution, and label prior bias is a recognized failure mode of imbalanced fine-tuning [NAACL 2024]. The CLICK only collapse we observe is an instance of this label prior imprinting acting on the action space rather than on a label space.

Small model tool use SFT. A controlled study across data, method, and scale [2026] finds that LoRA supervised fine-tuning is the strongest in distribution method for tool calling from 0.6B to 32B parameters. We hold the pipeline fixed and vary only the composition of the recovery data.

## 3 Method

### 3.1 Benchmark

We use a 51 task browser benchmark with programmatic oracles spanning four categories: navigation (10 tasks), search (9), form filling (19), and multi-step (13). Tasks run against static fixtures through Playwright, and success is determined by the oracle rather than by a language model judge. A separate 10 task held out set on an unseen site serves as a secondary generalization check.

### 3.2 Data

A Qwen3 4B teacher generates successful trajectories, which we convert into single turn supervised fine-tuning examples of the form (observation, action). We collect two pools. The clean pool contains trajectories with no injected failure. The recovery pool contains trajectories in which we injected a stale observation failure, removing an element so that every subsequent reference shifts, and in which the teacher recovered. Each recovery trajectory contributes two kinds of step. Error free steps, which we call clean like, are the actions before the injection and after the recovery and resemble ordinary clean data. Error containing steps, which we call recovery signal, are the steps whose observation carries the injected error.

### 3.3 Composition ablation

From a fixed clean set of 84 examples we build four conditions: A holds 84 clean examples; B adds 34 clean like examples; C adds 21 recovery signal examples; D adds all 55 recovery derived examples.

### 3.4 Data mixing sweep

Holding the recovery composition fixed, we vary the recovery derived fraction across 0, 20, 29, 33, 36, 40, 46, and 50 percent. Fractions above 40 percent are achieved by subsampling the clean set, a confound we flag in Section 5.

### 3.5 Training and evaluation

We fine-tune Qwen3 0.6B with LoRA (rank 8, 16 layers), batch size 4, 200 iterations, learning rate 1e-4, assistant token only loss, and no thinking channel. Each condition uses three to seven seeds. For every run we record task success, per category success, the full action sequence for the CLICK, TYPE, SEARCH, and SUBMIT distribution, validation loss, and the iteration 100 checkpoint.

## 4 Results

### 4.1 Bimodal outcomes in the full recovery condition

Table 1 reports per seed results. Conditions A and B are unimodal and stable. Condition D is bimodal: two seeds collapse to 16 and 27 while five sit between 38 and 47.

| Condition | n | Mean ± std | Range |
|---|---|---|---|
| A (clean) | 7 | 45.7 ± 2.9 | 40 to 49 |
| B (plus clean like) | 6 | 47.2 ± 3.6 | 43 to 51 |
| C (plus recovery) | 7 | 40.9 ± 5.4 | 31 to 46 |
| D (both) | 7 | 36.6 ± 11.2 | 16 to 47 |

When the collapse occurs it takes a consistent form, which we detail in Section 4.3: roughly 94 percent CLICK, zero SEARCH, zero SUBMIT, and a rise in steps per task from 2.8 to 5.1.

### 4.2 Non-monotonic data mixing

Table 2 and Figure 2 show that the instability concentrates in a window near 36 to 40 percent and is not monotonic. Higher recovery fractions do not produce more collapse. The collapse modes also vary across seeds: one seed collapses broadly, another collapses only on search, and the CLICK only mode of Section 4.3 is a third distinct form. The failure is therefore not a single mechanism.

| Recovery derived fraction | n | Mean ± std | Collapse |
|---|---|---|---|
| 0% | 7 | 45.7 ± 2.9 | none |
| 20% | 7 | 40.9 ± 5.4 | none |
| 29% | 6 | 47.2 ± 3.6 | none |
| 33% | 3 | 41.0 ± 2.6 | none |
| 36% | 3 | 41.3 ± 13.3 | 1 of 3 |
| 40% | 7 | 36.6 ± 11.2 | 2 of 7 |
| 46% | 3 | 42.7 ± 4.5 | none |
| 50% | 3 | 43.0 ± 1.7 | none |

### 4.3 The CLICK only attractor and action monitoring

Table 3 and Figure 3 report the action distribution on the clean 51 task run. A high CLICK share alone is not fatal; condition B succeeds at 59 percent CLICK. What distinguishes the collapsed runs is the loss of the other action types. This makes the CLICK share, or equivalently the presence of SEARCH and TYPE, a cheap and interpretable monitor of collapse.

| Model | CLICK | TYPE | SEARCH |
|---|---|---|---|
| A | 29.1% | 59.6% | 3.5% |
| B | 59.3% | 32.1% | 5.7% |
| C | 29.9% | 51.3% | 3.6% |
| D (healthy seed) | 59.7% | 27.6% | 6.0% |
| D (collapsed seed) | 93 to 95% | 0 to 7% | 0% |

### 4.4 Validation loss is blind to the collapse

Figure 4 plots validation loss against task success across all seeds. Validation loss does not rank checkpoints. A collapsed D seed scoring 16 of 51 achieves a validation loss of 0.178, which is lower than the best healthy seed at 47 of 51 with a loss of 0.217. The validation split here contains roughly 14 examples and is dominated by clean data, so it cannot see the action distribution pathology. The CLICK share, by contrast, cleanly separates collapsed from healthy checkpoints at 93, 60, and 30 percent.

### 4.5 A late training bifurcation

Figure 5 compares iteration 100 with final success for the five D seeds. At iteration 100 the seeds destined to collapse look healthier than several that eventually recover: the collapsing seeds score 30 to 32 of 51 with low CLICK, while two healthy seeds are already CLICK dominant at 62 and 100 percent. The collapse therefore develops in the second half of training. The model traverses a transient CLICK phase and either escapes it or is captured by it.

## 5 Discussion

Mechanism. The CLICK only collapse is consistent with label prior imprinting. A recovery set that is 71 percent CLICK biases the policy toward CLICK, and once SEARCH and TYPE are driven out of the action distribution the policy becomes trapped in a loop. We offer this as a hypothesis. The action distribution evidence is consistent with it but does not establish causation.

Relation to prior work. We do not claim to discover bimodal collapse, which Sumit et al. already documented, nor do we claim that recovery data cannot help, which PALADIN shows it can at scale. Our contribution is the controlled composition, mixing, and monitoring analysis in the sub-1B browser setting, and the action distribution signal that detects collapse where validation loss cannot.

Practical recommendations. Practitioners training local sub-1B agents should not assume that recovery data helps. They should validate with multiple seeds rather than one, monitor the action distribution rather than validation loss when selecting a checkpoint, and treat the recovery derived fraction conservatively, since the instability concentrates in a narrow window.

Limitations. The benchmark is synthetic and confined to a single domain. The model family is a single Qwen3 checkpoint. Seed counts are small, between three and seven. The 46 and 50 percent mixing points are confounded by clean subsampling. The mechanism is correlational rather than causal. The held out set is too small to support strong generalization claims.

## 6 Conclusion

Recovery trajectory supervision for sub-1B browser agents has a small and unreliable mean effect but produces bimodal, seed dependent outcomes. A subset of seeds collapse to a CLICK only policy that validation loss fails to detect, while a single action distribution statistic reveals it. The instability is non-monotonic in the recovery derived fraction and reflects a late training bifurcation. For local agent training, the quantity that matters is stability, and it should be assessed with multiple seeds and action distribution monitoring rather than validation loss.
