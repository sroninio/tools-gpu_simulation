# GPU Utilization Simulator — Project Context

## What This Is

A discrete-event simulation project studying GPU utilization and storage hierarchy
bandwidth in closed queueing networks, motivated by KV cache capacity planning for
agentic AI workloads (LLM agents cycling between tool execution and GPU inference).

**GitHub**: `https://github.com/sroninio/tools-gpu_simulation` (main branch)

---

## Core Insight

GPU utilization can be predicted using only the average tool service time — distribution
shape does not matter at large K* (LLN kicks in). Storage hierarchy bandwidth follows
from Che's approximation applied to agentic tool-time distributions.

**K\* formula**: `K* = ceil(E[T_tools] × rps × num_gpus) + num_gpus`

where rps = GPU throughput (req/sec per GPU job), K* = inflight sessions to saturate GPU.

---

## Storage Hierarchy Model

Two-tier storage per session KV cache:
- **M** (fast tier): capacity M slots, infinite BW — LRU managed
- **CMX** (backing store): infinite capacity, finite BW = CMX_BW (req/sec)

All metrics in **req/sec** (no kv_size units).

**Theoretical minimum CMX_BW** to sustain λ = rps × num_gpus:

Solve for θ* (characteristic time / Che's t_c):
```
M = λ · F(θ*) · E[T | T ≤ θ*]        ← Little's Law on M (M exactly full)
```
Then:
```
CMX_BW* = M · (1 - F(θ*)) / (F(θ*) · E[T | T ≤ θ*])  =  λ · (1 - F(θ*))
```
Total storage throughput:
```
total req/sec = M / E[T | T ≤ θ*]  +  CMX_BW*  =  λ
```

θ* is invariant as num_gpus scales (K/λ = const). CMX_BW* scales linearly with num_gpus.

---

## File Structure

```
/Users/rspiegelman/gpu-utilization-sim/
  sim.py                     — original closed network simulator (no storage)
  sim_lru.py                 — LRU KV cache simulator (kv_size units, recompute on GPU)
  sim_storage.py             — NEW: storage hierarchy simulator (req/sec, CMX queue)
  codex_cli_granular.py      — 2000-bucket Codex CLI CDF for fine-grained θ* solving
  version_2_tool_time_cdfs.py — 15 workload CDFs (P05–P99, 20 buckets each)
  version_3_tool_time_cdfs.py — same but P99 removed, renormalized to P95
  hetero_config.py           — heterogeneous SQ type configuration
  cmx_calc.py                — Python port of CMX gain calculator
  run_table.py               — table generation across 9 distributions
  save_to_db.py              — persist results to SQLite
  cdfToDistr.py              — real Codex CLI latency CDF data
```

---

## sim.py — Original Closed Network Simulator

No storage, no recompute. Pure GPU utilization under closed queueing network.

### Key Functions
- `make_constant(value)`, `make_exponential(mean)`, `make_discrete(buckets)`
- `run_simulation(d_tools, d_gpu, K, num_gpus, n_warmup, n_measure, seed)` → float
- `sweep(...)` — prints utilization table sweeping K

### Implementation
- Min-heap event queue, event types: `TOOLS_DONE=0`, `GPU_DONE=1`
- Warmup: skip first `n_warmup` GPU completions

---

## sim_lru.py — LRU KV Cache Simulator

Models KV cache as LRU with finite capacity K. Misses trigger GPU recompute.
**Units: kv_size (GB), BW in GB/s.** Untouched from original.

### Key Function
```python
run_lru_simulation(d_tools, d_gpu, d_recompute, K, S, num_gpus=1,
                   n_warmup, n_measure, seed,
                   dead_session_fix=False, spawn_on_death=False)
→ (good_util, avg_alive_sessions, eviction_ratio, dead_area/wall)
```

### Parameters
| Param | Description |
|---|---|
| `d_tools` | tool time distribution |
| `d_gpu` | GPU service time (good job) |
| `d_recompute` | GPU service time (cache miss recompute) |
| `K` | LRU cache capacity (slots) |
| `S` | steps per session |
| `num_gpus` | GPU count |
| `dead_session_fix` | if True, free slot immediately on session death |
| `spawn_on_death` | if True, spawn new session on every death (1:1 pairing) |

### LRU via OrderedDict
```python
storage = OrderedDict()          # MRU at back, LRU at front
storage.move_to_end(sid)         # promote to MRU
storage.popitem(last=False)      # evict LRU
```

---

## sim_storage.py — Storage Hierarchy Simulator

**Units: req/sec throughout. No kv_size.**
Models M (fast tier, LRU) + CMX (single-server queue or instant).

### Key Function
```python
run_sim(
    d_tools,            # callable(rng) → tool time (sec)
    rps,                # GPU service rate (req/sec)
    M,                  # fast-tier capacity (slots)
    num_gpus=1,
    S=10000,            # steps per session
    eviction='lru',     # 'lru' | 'oracle' | 'weak_oracle'
    theta_star=None,    # required for oracle / weak_oracle
    miss_mode='cmx',    # 'cmx' | 'recompute'
    cmx_bw=math.inf,    # req/sec; required for miss_mode='cmx'
    recompute_svc=None, # GPU svc time (sec); required for miss_mode='recompute'
    N=None,             # inflight session cap; required for oracle / weak_oracle
    seed=42,
    n_warmup=20_000,
    n_measure=200_000,
) → dict(gpu_util, cmx_util, miss_rate, cmx_real, recomp_rate)
```

### Eviction Modes
| Mode | Behavior |
|---|---|
| `lru` | Standard LRU — always insert at MRU when going to tools |
| `oracle` | Draw T at GPU_DONE; T > θ* → evict immediately from M |
| `weak_oracle` | Draw T at GPU_DONE; T > θ* → schedule EVICT_TIMER at t+θ*; evict at timer only if still in M. **Result: identical to LRU in steady state** |

### Miss Modes
| Mode | Required param | Behavior |
|---|---|---|
| `cmx` | `cmx_bw` (req/sec) | miss → CMX single-server queue, svc = 1/cmx_bw. Use `math.inf` for instant fetch |
| `recompute` | `recompute_svc` (sec) | miss → immediate GPU recompute queue, no CMX |

### Rules
- `eviction='oracle'` or `'weak_oracle'` requires both `theta_star` and `N`
- `N=None` → uncapped (open network); valid only for `eviction='lru'`
- `cmx_bw=math.inf` → no CMX bottleneck, misses served instantly

### Return Values
| Key | Description |
|---|---|
| `gpu_util` | GPU busy fraction |
| `cmx_util` | CMX busy fraction (0 if recompute mode or cmx_bw=inf) |
| `miss_rate` | fraction of tool returns that were misses |
| `cmx_real` | actual CMX fetches/sec (0 in recompute mode) |
| `recomp_rate` | GPU recomputes/sec (0 in cmx mode) |

---

## Key Simulation Results

### Codex CLI granular CDF (mean=27.418s, rps=1.6988, S=10000, M=K*/2, N=K*)

**CMX_BW = CMX_BW* (theoretical minimum), θ* = 102.5s, F(θ*)=0.936**

| GPUs | K* | LRU gpu% | Oracle gpu% |
|------|-----|----------|-------------|
| 1    | 48  | 71%      | 83%         |
| 50   | 2379| 73%      | 95%         |
| 500  | 23790| 86%    | 99%         |

Oracle converges to ~100% at 500 GPUs (LLN). LRU always underperforms oracle because it cannot proactively evict long-T sessions.

**weak_oracle = LRU**: LRU pressure naturally evicts long-staying sessions before the θ* timer fires.

### Theoretical Minimum CMX_BW Derivation

Given M (fast tier slots), λ = rps × num_gpus, and tool time CDF F(t):

**Step 1** — Solve for θ* (the eviction threshold):
```
F(θ*) · E[T | T ≤ θ*] = M / λ
```
Sessions with T ≤ θ* stay in M; sessions with T > θ* go to CMX.

**Step 2** — CMX_BW* is the miss rate at λ:
```
CMX_BW* = λ · (1 - F(θ*))
```

**Step 3** — Verify total storage throughput equals λ:
```
M / E[T | T ≤ θ*]  +  CMX_BW*  =  λ · F(θ*)  +  λ · (1 - F(θ*))  =  λ  ✓
```

To find CMX_BW* numerically: sweep θ* over CDF until Little's Law is satisfied,
then read off CMX_BW* = λ · (1 - F(θ*)).

**Scaling**: θ* is invariant with num_gpus (since M/λ = (K*/2)/(rps×num_gpus) = const).
CMX_BW* scales linearly with num_gpus.

### Che's Approximation Connection
The θ* equation is identical to Che's characteristic time for LRU caches under IRM:
- Che: inter-request time per object → our: tool time T per session
- Che: object request rate → our: session rate λ = rps × num_gpus
- Same math, different physics.

---

## codex_cli_granular.py — Fine-Grained CDF

2000 sub-buckets (20 original × 100 sub-buckets each), linearly spaced within each bucket.
Mean = 27.418s. Used for fine-grained θ* solving.

```python
from codex_cli_granular import codex_cli_granular, cdf_stats, THETAS
F, ET = cdf_stats(theta)   # F(theta), E[T | T <= theta]
```

---

## version_2_tool_time_cdfs.py — 15 Workload CDFs

P05–P99 percentile-based discrete distributions for 15 agentic workloads.

```python
from version_2_tool_time_cdfs import ALL_DISTRIBUTIONS, RAW, PCTS
# ALL_DISTRIBUTIONS = [(name, distribution), ...]
```

Key workloads: Codex CLI (mean=27.4s), SemiAnalysis CC 1M (mean long tail),
Baseten, Multi-agent Swarm CC, NV Employee Traces, Claude Code (tool-call).

---

## DSv3 Reference Numbers (Codex CLI, 1 GPU)

- rps = 1.6988 req/sec
- kv_size = 2.0 GB (legacy, sim_lru only)
- K* = 48 (1 GPU), 762 (16 GPUs), 23790 (500 GPUs)
- θ* ≈ 103s, F(θ*) ≈ 0.937, CMX_BW* ≈ 0.107 req/sec (1 GPU)
