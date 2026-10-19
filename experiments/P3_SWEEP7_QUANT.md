# P3 Sweep #7 — quantization (POSITIVE)

**Date:** 2026-10-18
**Branch:** `stale-obs`
**Result:** the first clearly positive P3 lever. Quantization preserves the
0.6B agent's capability while cutting memory ~72% and increasing speed ~61%.

## TL;DR

| Metric | FP16 | 8-bit | 4-bit |
|---|---|---|---|
| Normal benchmark (51) | 48/51 | **48/51** | **47/51** |
| Held-out (10) | 9/10 | 9/10 | 9/10 |
| Loaded model size (peak-after-load) | 1137 MB | 604 MB (−47%) | **320 MB (−72%)** |
| Peak eval RAM | 1703 MB | 1110 MB | **851 MB** |
| tokens/sec | 103.7 | 143.6 (+38%) | **167.0 (+61%)** |

8-bit is **free** (identical quality, 0 regressions). 4-bit is **nearly free**
(−1 task on a single search query, −72% memory, +61% speed).

## Method

Fixed the p3-formfix adapter (48/51 clean baseline); changed only the base
model's quantization.

`mlx_lm.load` has **no on-the-fly quantization parameter** — quantization uses
the same pre-conversion flow already used for the teacher:

```bash
python -m mlx_lm convert --model Qwen/Qwen3-0.6B --mlx-path models/Qwen3-0.6B-8bit  -q --q-bits 8
python -m mlx_lm convert --model Qwen/Qwen3-0.6B --mlx-path models/Qwen3-0.6B-4bit  -q --q-bits 4
```

Then eval is unchanged except `--model <converted-path>` (the harness's
`MLXAgent` passes `model_path` straight to `load(..., adapter_path=...)`). No
code change. The LoRA adapter (trained on FP16) applies cleanly on top of the
quantized base.

## Results — normal benchmark (51 tasks)

| Category | FP16 | 8-bit | 4-bit |
|---|---|---|---|
| navigation | 10/10 | 10/10 | 10/10 |
| search | 8/9 | 8/9 | 7/9 |
| form | 18/19 | 18/19 | 18/19 |
| multi_step | 12/13 | 12/13 | 12/13 |
| **Total** | **48/51** | **48/51** | **47/51** |
| tool errors | 2 | 2 | 3 |

Regressions vs FP16: 8-bit **none**; 4-bit only `search-003`.

## Held-out (10 tasks)

All three = **9/10** (identical).

## Memory / size / speed

| Metric | FP16 | 8-bit | 4-bit |
|---|---|---|---|
| Disk size | ~1200 MB | 615 MB | 331 MB |
| Loaded model size (peak-after-load) | 1137 MB | 604 MB | 320 MB |
| Peak eval RAM (`mx.get_peak_memory`) | 1703 MB | 1110 MB | 851 MB |
| Load time | 0.73 s | 0.63 s | 0.75 s |
| tokens/sec | 103.7 | 143.6 | 167.0 |
| avg latency / task | 586 ms | 530 ms | 526 ms |

(On Apple Silicon memory is shared, so both the on-disk size and the resident
`mx` peak memory are reported — the latter is the deployment-relevant number.)

## Key finding: the single 4-bit failure is a string-generation artifact

Diagnosed `search-003` (goal "search for 'tablet'"):

- FP16: `CLICK(ref=4), CLICK(ref=4), SEARCH(query='tablet')` → success.
- 4-bit: `CLICK(ref=4), CLICK(ref=4), SEARCH(query='tableboard'), …` → fail.

The 4-bit model makes the **right decision** (focus the box, then SEARCH) but
generates the **wrong query string** (`tableboard` vs `tablet`). Quantization
preserves action *selection* and only slightly degrades *exact-value*
generation — consistent with the earlier finding that form-value tasks already
sit at the 0.6B capacity ceiling (P3 sweeps #1/#5).

## Conclusion

- **8-bit**: identical quality, −47% loaded memory, +38% speed — free.
- **4-bit**: 47/51 (98% of FP16), −72% loaded memory, +61% speed — recommended
  deployment point for the local/consumer-hardware constraint.
- Quantization is the **first positive P3 lever** (data, loss-weight, observation
  format, tool space, context/memory, and recovery were all negative or
  whack-a-mole). It moves the capability-efficiency frontier without teaching
  new skill.

## Next step

P4: SAN/Needle-style architecture + checkpoint sweep {135M → 600M}.
