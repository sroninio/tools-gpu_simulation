import heapq
import math
import random
from collections import deque, OrderedDict

# import distribution factories from original sim
from sim import make_constant, make_exponential, make_discrete

TOOLS_DONE = 0
GPU_DONE   = 1


def run_lru_simulation(d_tools, d_gpu, d_recompute, K, S, num_gpus=1,
                       n_warmup=10_000, n_measure=200_000, seed=42,
                       dead_session_fix=False):
    """
    Closed network with LRU context storage.

    K        : storage capacity (number of context slots)
    S        : steps per session (GPU jobs before session dies)
    d_tools  : tools service time distribution
    d_gpu    : good GPU job service time distribution
    d_recompute : recompute GPU job service time distribution (cache miss)

    Returns  : good_utilization = good GPU busy time / (num_gpus * wall_time)
    """
    rng = random.Random(seed)

    def sample_tools():     return d_tools(rng)
    def sample_gpu():       return d_gpu(rng)
    def sample_recompute(): return d_recompute(rng)

    # ── state ──────────────────────────────────────────────────────────────
    heap          = []
    eid           = 0
    session_id    = 0          # monotonically increasing session ID counter

    gpu_free      = num_gpus   # idle GPU server count
    gpu_queue     = deque()    # queue of (session_id, is_recompute)

    # LRU storage: OrderedDict{session_id: True}, LRU at front (last=False)
    storage       = OrderedDict()

    # steps remaining per live session
    steps         = {}         # session_id -> steps_remaining

    # ── metrics ────────────────────────────────────────────────────────────
    good_busy      = 0.0       # accumulated good-job GPU time (post warmup)
    alive_count    = 0         # current number of alive sessions
    alive_area     = 0.0       # time-weighted sum of alive_count (post warmup)
    last_event_t   = 0.0       # time of last event (for time-weighted avg)
    gpu_done_count = 0
    measured       = 0
    warmup_done    = False
    measure_start  = 0.0
    t              = 0.0

    # ── helpers ────────────────────────────────────────────────────────────
    def lru_insert(sid):
        """Insert sid into storage, evicting LRU if needed."""
        if sid in storage:
            storage.move_to_end(sid)
            return
        if len(storage) >= K:
            storage.popitem(last=False)   # evict LRU
        storage[sid] = True

    def lru_touch(sid):
        """Promote sid to MRU (most recently used)."""
        if sid in storage:
            storage.move_to_end(sid)

    def spawn_session():
        """Create a new session and push it directly into the GPU queue."""
        nonlocal session_id, eid, alive_count
        sid = session_id
        session_id += 1
        steps[sid] = S
        alive_count += 1
        lru_insert(sid)           # claim a storage slot
        # go straight to GPU (loop starts at GPU)
        gpu_queue.append((sid, False))   # not a recompute

    def schedule_gpu(sid, is_recompute):
        """Start a GPU job immediately (gpu_free > 0 assumed by caller)."""
        nonlocal gpu_free, eid
        gpu_free -= 1
        svc = sample_recompute() if is_recompute else sample_gpu()
        heapq.heappush(heap, (t + svc, eid, GPU_DONE, sid, is_recompute, svc))
        eid += 1

    def try_dequeue():
        """Pull next job from queue onto a free GPU, or spawn new session if idle."""
        if gpu_queue:
            sid, is_rc = gpu_queue.popleft()
            schedule_gpu(sid, is_rc)
        else:
            # queue empty → this GPU would go idle, spawn new session immediately
            spawn_session()
            sid, is_rc = gpu_queue.popleft()
            schedule_gpu(sid, is_rc)

    # ── initialise: spawn first session ────────────────────────────────────
    spawn_session()
    sid0, rc0 = gpu_queue.popleft()
    schedule_gpu(sid0, rc0)

    # ── event loop ─────────────────────────────────────────────────────────
    while measured < n_measure:
        t, _, etype, sid, is_recompute, svc = heapq.heappop(heap)

        # accumulate time-weighted alive count (post warmup)
        if warmup_done:
            alive_area += alive_count * (t - last_event_t)
        last_event_t = t

        if etype == TOOLS_DONE:
            # session returned from tools; update LRU regardless
            lru_touch(sid)
            in_storage   = sid in storage
            need_recompute = not in_storage
            if need_recompute:
                # will re-enter storage only after recompute finishes
                pass
            if gpu_free > 0:
                schedule_gpu(sid, need_recompute)
            else:
                gpu_queue.append((sid, need_recompute))

        else:  # GPU_DONE
            gpu_done_count += 1
            gpu_free += 1

            if warmup_done:
                if not is_recompute:
                    good_busy += svc
                measured += 1
            elif gpu_done_count >= n_warmup:
                warmup_done   = True
                measure_start = t
                last_event_t  = t

            # post-recompute: session re-enters storage
            if is_recompute:
                lru_insert(sid)

            # decrement steps; send to tools or retire
            if sid in steps:
                steps[sid] -= 1
                if steps[sid] > 0:
                    # send back to tools
                    heapq.heappush(heap, (t + sample_tools(), eid, TOOLS_DONE, sid, False, 0.0))
                    eid += 1
                else:
                    # session dies
                    del steps[sid]
                    alive_count -= 1
                    if dead_session_fix and sid in storage:
                        del storage[sid]   # immediately free the slot

            # free GPU: serve queue or spawn new session
            try_dequeue()

    wall = t - measure_start
    return good_busy / (num_gpus * wall), alive_area / wall


def sweep_lru(d_tools, d_gpu, d_recompute, K, S_list, num_gpus=1,
              n_warmup=10_000, n_measure=200_000, seed=42):
    """Sweep over S (steps per session) values and print good utilization."""
    print(f'\nd_tools={d_tools.name}')
    print(f'd_gpu={d_gpu.name}')
    print(f'd_recompute={d_recompute.name}')
    print(f'K={K}  num_gpus={num_gpus}')
    print()
    print(f"{'S':>6} | {'Good Util':>10}")
    print('-' * 22)
    for S in S_list:
        u = run_lru_simulation(d_tools, d_gpu, d_recompute, K, S, num_gpus,
                               n_warmup=n_warmup, n_measure=n_measure, seed=seed)
        print(f'{S:>6} | {u:>9.2%}')


if __name__ == '__main__':
    # quick sanity: with very large K (no evictions) and S=inf-like (large S),
    # good utilization should approach the original closed-network baseline
    d_tools     = make_exponential(90.0)
    d_gpu       = make_constant(0.3)
    d_recompute = make_constant(3.0)   # 10x longer than good job

    print('=== LRU sim, K=500 (large, few evictions), varying S ===')
    sweep_lru(d_tools, d_gpu, d_recompute, K=500, S_list=[1, 5, 10, 50, 100],
              num_gpus=1, n_warmup=10_000, n_measure=200_000)

    print('\n=== LRU sim, K=10 (tight, many evictions), varying S ===')
    sweep_lru(d_tools, d_gpu, d_recompute, K=10, S_list=[1, 5, 10, 50, 100],
              num_gpus=1, n_warmup=10_000, n_measure=200_000)
