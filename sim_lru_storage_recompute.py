"""
Storage hierarchy simulation — req/sec units throughout, no kv_size.

System model
------------
Sessions cycle: GPU job → tools (time T) → GPU job → ... × S steps, then die.
A new session is spawned whenever a GPU becomes idle and the queue is empty
(open network for eviction='lru'). For eviction='oracle' or 'weak_oracle' the
total number of inflight sessions is capped at N — no new session is spawned if
N sessions are already in the system (closed network).

On return from tools a session either hits M (served instantly, free) or misses
(served according to miss_mode).

Parameters
----------
eviction : 'lru' | 'oracle' | 'weak_oracle'
    'lru'         — standard LRU: always insert session into M at MRU position
                    when it leaves for tools. Evicts LRU slot if M is full.
                    No N required (open network — new session spawned on idle GPU).
    'oracle'      — at GPU_DONE draw T; if T <= theta_star insert into M (MRU),
                    else evict immediately from M. Requires theta_star AND N.
    'weak_oracle' — at GPU_DONE draw T; always insert into M (MRU); if T > theta_star
                    also schedule EVICT_TIMER at t+theta_star — evicts from M at
                    that time only if session is still there.
                    Requires theta_star AND N.
                    Note: in practice weak_oracle == lru (LRU pressure naturally
                    evicts long-staying sessions before the timer fires).

miss_mode : 'cmx' | 'recompute'
    'cmx'       — miss → CMX single-server queue, service time = 1/cmx_bw.
                  Requires cmx_bw (req/sec). Set math.inf for instant (free) fetch.
    'recompute' — miss → immediately queued for GPU recompute job.
                  Requires recompute_svc (GPU service time in sec). No CMX involved.

N : int | None
    Max inflight sessions (closed network cap).
    REQUIRED for eviction='oracle' or 'weak_oracle'. Must be None for open network.

Returns
-------
dict with keys:
    gpu_util    — GPU busy fraction
    cmx_util    — CMX busy fraction (0 when miss_mode='recompute' or cmx_bw=inf)
    miss_rate   — fraction of tool returns that were cache misses
    cmx_real    — CMX fetches per second (0 in recompute mode)
    recomp_rate — GPU recomputes per second (0 in cmx mode)
"""

import heapq
import math
import random
from collections import deque, OrderedDict

_TOOLS_DONE   = 0
_GPU_DONE     = 1
_CMX_DONE     = 2
_EVICT_TIMER  = 3


def run_sim(
    d_tools,           # callable(rng) -> tool time in seconds
    rps,               # GPU service rate (req/sec per GPU job)
    M,                 # fast-tier capacity (slots)
    num_gpus=1,
    S=10000,           # steps per session
    eviction='lru',    # 'lru' | 'oracle'
    theta_star=None,   # required for eviction='oracle'
    miss_mode='cmx',   # 'cmx' | 'recompute'
    cmx_bw=math.inf,   # req/sec; required for miss_mode='cmx'
    recompute_svc=None,# GPU service time (sec); required for miss_mode='recompute'
    N=None,            # inflight session cap; required for eviction='oracle'
    seed=42,
    n_warmup=20_000,
    n_measure=200_000,
):
    # ── validation ────────────────────────────────────────────────────────────
    if eviction in ('oracle', 'weak_oracle'):
        if theta_star is None:
            raise ValueError(f"eviction='{eviction}' requires theta_star")
        if N is None:
            raise ValueError(f"eviction='{eviction}' requires N (inflight cap)")
    if miss_mode == 'cmx' and cmx_bw is None:
        raise ValueError("miss_mode='cmx' requires cmx_bw")
    if miss_mode == 'recompute' and recompute_svc is None:
        raise ValueError("miss_mode='recompute' requires recompute_svc")

    rng = random.Random(seed)
    heap, eid, sid_ = [], -1, 0

    gpu_free  = num_gpus
    gpu_q     = deque()   # (sid, is_recompute)
    cmx_q     = deque()   # sid waiting for CMX read
    cmx_busy  = False

    storage   = OrderedDict()   # LRU: MRU at back, LRU at front
    steps     = {}              # sid -> remaining steps

    # metrics
    measured      = 0
    gpu_busy      = 0.0
    cmx_busy_time = 0.0
    miss_count    = 0
    total_returns = 0
    recomp_count  = 0

    warmup_done    = False
    gpu_done_count = 0
    measure_start  = 0.0
    t              = 0.0
    cmx_start_t    = None   # time CMX last became busy

    # ── helpers ───────────────────────────────────────────────────────────────
    def lru_insert(sid):
        if sid in storage:
            storage.move_to_end(sid)
            return
        if len(storage) >= M:
            storage.popitem(last=False)
        storage[sid] = True

    def spawn():
        nonlocal sid_
        sid = sid_; sid_ += 1
        steps[sid] = S
        lru_insert(sid)
        gpu_q.append((sid, False))

    def try_gpu():
        nonlocal gpu_free, eid
        if not gpu_q or gpu_free == 0:
            return
        sid, is_rc = gpu_q.popleft()
        gpu_free -= 1
        svc = recompute_svc if is_rc else 1.0 / rps
        heapq.heappush(heap, (t + svc, eid, _GPU_DONE, sid, is_rc, svc))
        eid += 1

    def start_cmx(sid):
        nonlocal cmx_busy, eid, cmx_start_t
        cmx_busy   = True
        cmx_start_t = t
        svc = 1.0 / cmx_bw if cmx_bw != math.inf else 0.0
        heapq.heappush(heap, (t + svc, eid, _CMX_DONE, sid, False, svc))
        eid += 1

    def maybe_spawn():
        if N is None:
            spawn(); try_gpu()
        elif len(steps) < N:
            spawn(); try_gpu()

    # ── seed initial sessions ─────────────────────────────────────────────────
    n_seed = min(num_gpus, N) if N is not None else num_gpus
    for _ in range(n_seed):
        spawn()
    for _ in range(n_seed):
        try_gpu()

    # ── event loop ────────────────────────────────────────────────────────────
    while measured < n_measure:
        t, _, et, sid, is_rc, svc = heapq.heappop(heap)

        # ── TOOLS_DONE ────────────────────────────────────────────────────────
        if et == _TOOLS_DONE:
            in_m = sid in storage
            if warmup_done:
                total_returns += 1
                if not in_m:
                    miss_count += 1

            if in_m:
                if eviction == 'lru':
                    storage.move_to_end(sid)   # touch to MRU
                gpu_q.append((sid, False))
                try_gpu()
            else:
                # miss
                if miss_mode == 'cmx':
                    if not cmx_busy:
                        start_cmx(sid)
                    else:
                        cmx_q.append(sid)
                else:  # recompute
                    gpu_q.append((sid, True))
                    try_gpu()

        # ── EVICT_TIMER ───────────────────────────────────────────────────────
        elif et == _EVICT_TIMER:
            if sid in storage:
                del storage[sid]

        # ── CMX_DONE ──────────────────────────────────────────────────────────
        elif et == _CMX_DONE:
            if warmup_done and cmx_start_t is not None:
                cmx_busy_time += t - cmx_start_t
            lru_insert(sid)   # fetched from CMX, now in M
            gpu_q.append((sid, False))
            try_gpu()
            if cmx_q:
                start_cmx(cmx_q.popleft())
            else:
                cmx_busy  = False
                cmx_start_t = None

        # ── GPU_DONE ──────────────────────────────────────────────────────────
        else:
            gpu_free += 1
            gpu_done_count += 1

            if warmup_done:
                measured  += 1
                gpu_busy  += svc
                if is_rc:
                    recomp_count += 1
                    lru_insert(sid)   # re-enter M after recompute
            elif gpu_done_count >= n_warmup:
                warmup_done   = True
                measure_start = t

            if sid in steps:
                steps[sid] -= 1
                if steps[sid] > 0:
                    tool_time = d_tools(rng)
                    if eviction == 'oracle':
                        if tool_time <= theta_star:
                            lru_insert(sid)
                        else:
                            if sid in storage:
                                del storage[sid]
                    elif eviction == 'weak_oracle':
                        lru_insert(sid)
                        if tool_time > theta_star:
                            heapq.heappush(
                                heap,
                                (t + theta_star, eid, _EVICT_TIMER, sid, False, 0.0),
                            )
                            eid += 1
                    else:
                        lru_insert(sid)
                    heapq.heappush(
                        heap,
                        (t + tool_time, eid, _TOOLS_DONE, sid, False, 0.0),
                    )
                    eid += 1
                else:
                    del steps[sid]
                    if sid in storage:
                        del storage[sid]

            if gpu_q:
                try_gpu()
            else:
                maybe_spawn()

    wall = t - measure_start
    gpu_util    = gpu_busy / (num_gpus * wall)
    cmx_util    = cmx_busy_time / wall if miss_mode == 'cmx' and cmx_bw != math.inf else 0.0
    miss_rate   = miss_count / total_returns if total_returns > 0 else 0.0
    cmx_real    = miss_count / wall if miss_mode == 'cmx' else 0.0
    recomp_rate = recomp_count / wall if miss_mode == 'recompute' else 0.0

    return dict(
        gpu_util    = gpu_util,
        cmx_util    = cmx_util,
        miss_rate   = miss_rate,
        cmx_real    = cmx_real,
        recomp_rate = recomp_rate,
    )


if __name__ == '__main__':
    import math
    from codex_cli_granular import codex_cli_granular as d_tools, cdf_stats, THETAS

    rps      = 1.6988
    num_gpus = 1
    S        = 10000
    K_star   = math.ceil(d_tools.mean * rps * num_gpus) + num_gpus
    M        = K_star // 2
    N        = K_star
    theta    = 103.73
    cmx_bw   = 0.107

    print(f"Codex CLI granular  rps={rps}  num_gpus={num_gpus}  K*={K_star}  M={M}  N={N}")
    print(f"θ*={theta}s  CMX_BW={cmx_bw} req/sec\n")

    configs = [
        ('lru',    'cmx',       dict(cmx_bw=cmx_bw,                         N=N)),
        ('oracle', 'cmx',       dict(cmx_bw=cmx_bw, theta_star=theta,       N=N)),
        ('lru',    'recompute', dict(recompute_svc=1/rps,                    N=N)),
        ('oracle', 'recompute', dict(recompute_svc=1/rps, theta_star=theta,  N=N)),
    ]

    print(f"{'eviction':>8} {'miss_mode':>10} {'gpu_util':>10} {'miss%':>8} {'cmx_real':>10} {'recomp/s':>10}")
    print('-' * 65)
    for eviction, miss_mode, extra in configs:
        r = run_sim(
            d_tools, rps, M,
            num_gpus=num_gpus, S=S,
            eviction=eviction, miss_mode=miss_mode,
            **extra
        )
        print(f"{eviction:>8} {miss_mode:>10} {r['gpu_util']*100:>9.2f}% "
              f"{r['miss_rate']*100:>7.2f}% {r['cmx_real']:>10.4f} {r['recomp_rate']:>10.4f}")
