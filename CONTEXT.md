# GPU Utilization Simulator — Project Context

## What This Is

A discrete-event simulation project studying GPU utilization in closed queueing networks,
motivated by capacity planning for agentic AI workloads (LLM agents cycling between
tool execution and GPU inference).

**GitHub**: `https://github.com/sroninio/tools-gpu_simulation` (main branch)

---

## Core Insight

The key claim (validated by simulation): **GPU utilization can be predicted using only
the average service time of the tools distribution** — without knowledge of its shape or
variance. This follows from the Law of Large Numbers: when K* is large (many concurrent
agents), the aggregate arrival stream to GPU self-regularizes regardless of individual
distribution shape.

**Formula**: `K* = ceil(T_tools_avg / T_gpu_avg × num_gpus) + num_gpus`

where K* is the number of inflight requests needed to saturate the GPU.

---

## System Model

**Closed 2-station queueing network:**
- **Tools station**: infinite servers, service time distribution `d_tools`
- **GPU station**: `num_gpus` servers, shared queue, service time `d_gpu`
- K requests always in flight, cycling Tools → GPU → Tools → GPU → ...
- **Metric**: GPU utilization = fraction of time GPU(s) are busy

---

## File Structure

```
/Users/rspiegelman/gpu-utilization-sim/
  sim.py          — original closed network simulator
  sim_lru.py      — LRU scheduling variant (see below)
  cmx_calc.py     — Python port of CMX gain calculator
  run_table.py    — table generation across 9 distributions
  save_to_db.py   — persist results to SQLite (~Desktop/cmx_results.db)
  cdfToDistr.py   — real Codex CLI latency CDF data
```

---

## sim.py — Original Simulator

### Key Functions
- `make_constant(value)` — constant distribution
- `make_exponential(mean)` — exponential distribution
- `make_discrete(buckets)` — discrete distribution; buckets = `[(prob, value), ...]`
- `run_simulation(d_tools, d_gpu, K, num_gpus, n_warmup, n_measure, seed)` → utilization float
- `sweep(d_tools, d_gpu, num_gpus, k_multiplier, target_util, n_warmup, n_measure)` — prints table

### Implementation Details
- Min-heap event queue (Python's `heapq`)
- Event tuple: `(time, eid, event_type, req_id, service_time)`
- Event types: `TOOLS_DONE=0`, `GPU_DONE=1`
- Warmup: skip first `n_warmup` GPU completions before measuring
- Service time carried in event tuple (not tracked via timestamps)
- K* uses `math.ceil` (not `int`) to avoid rounding errors

### Standard Parameters Used
- `d_gpu = make_constant(1/3.45)` ≈ 0.29s
- `num_gpus = 16` or `50` (or `1` for small-K* regime)
- `n_warmup = 50_000` (or `500_000` for 50 GPUs / large mean distributions)
- `n_measure = 500_000` (or `10_000_000` for 50 GPUs)
- Request size = **2.5 GB** per inflight request (for capacity calculations)

---

## Key Findings — Original Simulator

### Distribution Shape Doesn't Matter (at large K*)

With `gpu=Constant(1/3.45)`, `num_gpus=50`, `tools_mean≈90s` (K*≈15,000):

| Tools Distribution | CV | Util @ K* | Saturates at |
|---|---|---|---|
| Constant(90) | 0 | 100.0% | 1.00×K* |
| Exponential(90) | 1.00 | 99.6% | 1.03×K* |
| 25%@2, 25%@4, 25%@16, 25%@340 (±Codex CLI) | 1.59 | 99.6% | 1.03×K* |
| 40%@0.5...3%@2410 (big variance) | ~6.5 | 99.5% | 1.03×K* |

**Rule of thumb**: LLN kicks in at K* ≈ 30. Below that, variance matters significantly.

### Capacity Per GPU
`Capacity per GPU = (T_tools_avg / T_gpu_avg + 1) × 2.5 GB`

For Codex CLI [raw] (mean=60.6s): **525 GB/GPU**

---

## Real Workload Distributions

From `cdf_export_for_cmx_tool.md` — inter-turn gaps (tool execution + human think time):

| Distribution | Mean (s) | Variance | CV | Notes |
|---|---|---|---|---|
| codex (local) [raw] | 20.6 | 59,015 | 11.81 | Highest CV |
| codex (local), decode-shifted | 28.9 | 103,270 | 11.12 | Recommended |
| NV-emp subdag (local) [raw] | 39.0 | 184,275 | 11.01 | |
| SemiAnalysis CC 1M ctx [raw] | 296.8 | 9,082,884 | 10.15 | |
| **Codex CLI [raw]** | **60.6** | **322,356** | **9.36** | Main test dist |
| SemiAnalysis CC 256K ctx [raw] | 398.7 | 13,030,726 | 9.05 | Largest mean |
| Multi-agent CC Swarm [raw] | 29.2 | 46,791 | 7.40 | |
| NV-emp subdag, decode-shifted | 69.8 | 190,693 | 6.25 | |
| Claude Code [raw] | 2.7 | 49 | 2.56 | |
| NV Employee Traces [raw] | 39.3 | 3,964 | 1.60 | Lowest CV |

"Decode-shifted" = gap + decode time added (more accurate for CMX idle window modeling).

### Codex CLI [raw] Buckets
```python
[(0.002235,0.182),(0.003352,0.581),(0.262569,2.39),(0.559777,10.3),
 (0.080447,38.2),(0.050279,82.7),(0.029050,179),(0.004470,422),
 (0.001117,971),(0.003352,2280),(0.001117,6320),(0.002235,10800)]
```

### SemiAnalysis CC 256K ctx [raw] Buckets
```python
[(0.006218,0.0215),(0.026206,0.182),(0.134220,0.573),(0.601094,2.1),
 (0.108036,11.4),(0.028988,39.2),(0.020825,85.3),(0.040994,185),
 (0.008411,426),(0.008999,950),(0.006196,2320),(0.009813,36600)]
```

---

## Table Results (16 & 32 GPUs, gpu=Constant(1/3.45))

Run via `run_table.py`. Results in `~/Desktop/cmx_table_results.json` and `~/Desktop/cmx_results.db`.

Key result: **7 of 9 distributions need 0 CMX BW** — HBM (700 GB) alone is sufficient.
Only SemiAnalysis distributions (mean > 296s) need CMX: 0.053 and 0.077 GB/s respectively.

---

## cmx_calc.py — CMX Gain Calculator Port

Python port of `cmx_gain_v2.html` JS logic. Verified to match HTML tool exactly.

### Fixed Constants
```python
SOL    = 3.45      # GPU_SOL_REQ_SEC (req/s)
HBM    = 700       # HBM_DRAM_SIZE_GB
RECOMP = 1e-7      # RECOMPUTE_REQ_SEC
AVG    = 2.5       # AVG_AGENT_CAP_SIZE_IN_GB
```

### Key Functions
- `compute_reqs_sec(hist, tau, tau_frac, cmx_bw)` — hist uses pct in [0,100]
- `best_reqs_at_bw(hist, cmx_bw)` — sweeps τ/frac, returns (best_reqs, tau, frac)
- `find_min_cmx_bw(hist)` — binary search for min BW to reach SOL; returns 0 if HBM sufficient

### CMX Gain Tool
Located at `/Users/rspiegelman/Desktop/cmx-gain/cmx_gain_v2.html`
GitHub: `https://github.com/sroninio/CMX_model`
Added "Find Min CMX BW → SOL" button (purple) using binary search.

---

## sim_lru.py — LRU Scheduling Simulator

### Motivation
Models a KV cache (context storage) with finite capacity K, LRU eviction,
and session lifetimes. Compares to original closed network baseline.

### Model
- **K**: storage capacity (context slots)
- **S**: steps per session (GPU jobs before session dies)
- **d_recompute**: GPU service time for cache-miss jobs (much longer than d_gpu)
- Sessions start at GPU (not Tools) — loop is GPU → Tools → GPU → ... × S times
- New session spawned immediately when any GPU becomes free with empty queue
- Dead sessions (completed S steps) keep storage slot until LRU-evicted
- LRU updated when session returns from Tools (regardless of hit/miss)
- Recomputed sessions re-enter storage AFTER recompute GPU job finishes

### Key Function
```python
run_lru_simulation(d_tools, d_gpu, d_recompute, K, S, num_gpus=1,
                   n_warmup, n_measure, seed, dead_session_fix=False)
→ (good_utilization, avg_alive_sessions)
```

- `good_utilization` = time on good jobs / (num_gpus × wall_time)
- `avg_alive_sessions` = time-weighted average of sessions with steps_remaining > 0
- `dead_session_fix=True` → immediately frees storage slot when session dies

### Implementation: LRU via OrderedDict
```python
from collections import OrderedDict
storage = OrderedDict()
storage.move_to_end(sid)        # promote to MRU
storage.popitem(last=False)     # evict LRU
```

### Key Findings — LRU Simulator

Parameters: Codex CLI [raw], gpu=Constant(1/3.45), S=10

| K/K* | Original | LRU 20× | LRU+slot release on session complete 20× | LRU 200× | LRU+slot release on session complete 200× |
|---|---|---|---|---|---|
| **1 GPU** | | | | | |
| 1× | 95.33% | 86.88% | 98.57% | 42.42% | 93.56% |
| 2× | 100.00% | 88.99% | 100.00% | 50.61% | 100.00% |
| 20× | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| **16 GPUs** | | | | | |
| 1× | 100.00% | 91.84% | 100.00% | 45.98% | 100.00% |
| 5× | 100.00% | 99.10% | 100.00% | 88.73% | 100.00% |
| 15× | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |

**Critical finding**: "Slot release on session complete" (immediately freeing dead session slots)
nearly eliminates eviction penalty — matches original sim at K* for 16 GPUs.

**S=1000**: 100% good utilization everywhere — long sessions amortize eviction cost completely.

### Avg Alive Sessions
At steady state (large K, no evictions), avg alive ≈ K* by Little's Law:
`avg_alive = GPU_throughput × cycle_time = (num_gpus/gpu_mean) × (tools_mean + gpu_mean)`

Note: requires sufficient warmup to converge. Use n_warmup ≥ 1_000_000 for S=1000.

---

## Graphs Saved to Desktop

- `codex_cli_16_32gpu.png` — Codex CLI [raw], 16 & 32 GPUs
- `semianalysis_256k_16_32gpu.png` — SemiAnalysis 256K, 16 & 32 GPUs
- `codex_semianalysis_16_32gpu.png` — combined 2×2 grid
- `codex_local_raw_1gpu.png` — codex (local) [raw], 1 GPU
- `codex_local_shifted70_1gpu.png` — codex (local) + 70s shift (mean≈90), 1 GPU
- Data JSONs: `codex_semianalysis_16_32gpu_data.json`, `semianalysis_256k_utilization_data.json`, etc.

Graph lines:
- **Navy blue** — GPU utilization curve
- **Red dashed** — capacity derived by mean
- **Green solid** — G1+G2 = 700 GB/GPU (HBM)
- **Magenta dash-dot** — 99.9% utilization threshold

---

## Context on "Sessions" in the Real World

- A **session** = one continuous agentic conversation (user ↔ LLM)
- **S steps** = number of turns (GPU forward passes) per session
- **Tools time** = inter-turn gap = tool execution + human think time
- **KV cache** grows as a stack (append-only) within each turn: X1:Y1:Y2:...:YK
- After turn ends: Yi (intermediate tool calls) may be stripped → context becomes X1:X2
- Compression events rebuild entire KV cache (all entries change, even "kept" recent ones)
- In Claude Code: Yi tool call blocks are NOT automatically stripped between turns
- Session IDs in traces may reset on compression → apparent session length shorter than real

---

## TODO / Next Steps
- Run LRU sim with more distributions beyond Codex CLI
- Explore OPT-inspired eviction policy (evict session whose return is furthest in future)
- Model intra-turn Yi token lifecycle across HBM/G2/CMX hierarchy
- Connect LRU results to CMX BW requirements
