import heapq
import random
from collections import deque

TOOLS_DONE = 0
GPU_DONE   = 1


def make_exponential(mean):
    def sampler(rng): return rng.expovariate(1.0 / mean)
    sampler.mean = mean
    sampler.name = f"Exponential(mean={mean})"
    return sampler

def make_discrete(buckets):
    """buckets: list of (probability, value), probabilities must sum to 1."""
    values = [v for _, v in buckets]
    weights = [p for p, _ in buckets]
    mean = sum(p * v for p, v in buckets)
    def sampler(rng): return rng.choices(values, weights=weights)[0]
    sampler.mean = mean
    sampler.name = "Discrete(" + ", ".join(f"{int(p*100)}%→{v}" for p, v in buckets) + ")"
    return sampler

def make_constant(value):
    def sampler(rng): return value
    sampler.mean = value
    sampler.name = f"Constant({value})"
    return sampler


def run_simulation(d_tools, d_gpu, K, n_warmup=10_000, n_measure=200_000, seed=42):
    rng = random.Random(seed)

    def sample_tools(): return d_tools(rng)
    def sample_gpu():   return d_gpu(rng)

    heap = []
    eid  = 0

    # All K requests start at Tools simultaneously (infinite servers, no wait)
    for i in range(K):
        heapq.heappush(heap, (sample_tools(), eid, TOOLS_DONE, i))
        eid += 1

    gpu_idle       = True
    gpu_queue      = deque()
    gpu_busy_since = 0.0

    gpu_busy_time  = 0.0
    gpu_done_count = 0      # total GPU completions seen
    measured       = 0      # GPU completions counted after warmup
    warmup_done    = False
    measure_start  = 0.0
    t              = 0.0

    while measured < n_measure:
        t, _, etype, req = heapq.heappop(heap)

        if etype == TOOLS_DONE:
            if gpu_idle:
                gpu_idle       = False
                gpu_busy_since = t
                heapq.heappush(heap, (t + sample_gpu(), eid, GPU_DONE, req))
                eid += 1
            else:
                gpu_queue.append(req)

        else:  # GPU_DONE
            gpu_done_count += 1

            if warmup_done:
                gpu_busy_time += t - gpu_busy_since
                measured      += 1
            elif gpu_done_count >= n_warmup:
                # warmup just finished at this event; start measuring from here
                warmup_done   = True
                measure_start = t

            # serve next queued request or go idle
            if gpu_queue:
                next_req       = gpu_queue.popleft()
                gpu_busy_since = t
                heapq.heappush(heap, (t + sample_gpu(), eid, GPU_DONE, next_req))
                eid += 1
            else:
                gpu_idle = True

            # return request to Tools
            heapq.heappush(heap, (t + sample_tools(), eid, TOOLS_DONE, req))
            eid += 1

    wall = t - measure_start
    return gpu_busy_time / wall


def sweep(d_tools, d_gpu, k_multiplier=3, target_util=0.9999,
          n_warmup=10_000, n_measure=200_000, seed=42):
    k_star = int(d_tools.mean / d_gpu.mean) + 1
    k_end  = k_star * k_multiplier
    step   = max(1, k_star // 30)   # ~30 data points regardless of K* size

    print(f"\ntools_avg={d_tools.mean}  gpu_avg={d_gpu.mean}  K*={k_star}")
    print(f"  d_tools = {d_tools.name}")
    print(f"  d_gpu   = {d_gpu.name}")
    print()
    print(f"{'K':>6} | {'K/K*':>6} | {'GPU Util':>9}")
    print("-" * 30)

    K = k_star
    while K <= k_end:
        util   = run_simulation(d_tools, d_gpu, K,
                                n_warmup=n_warmup, n_measure=n_measure, seed=seed)
        marker = "  <-- K*" if K == k_star else ""
        print(f"{K:>6} | {K/k_star:>6.2f} | {util:>8.2%}{marker}")
        if util >= target_util:
            break
        K += step


if __name__ == "__main__":
    gpu = make_constant(0.3)

    d_exp      = make_exponential(90.0)
    d_discrete = make_discrete([(0.25, 2), (0.25, 4), (0.25, 16), (0.25, 350)])

    d_bimodal  = make_discrete([(0.90, 1), (0.10, 900)])
    d_const    = make_constant(90.0)
    gpu_exp    = make_exponential(0.3)

    print("=== gpu=Exponential(0.3) ===")

    print("\n  --- Constant tools ---")
    sweep(d_const, gpu_exp)

    print("\n  --- Exponential tools ---")
    sweep(d_exp, gpu_exp)

    print("\n  --- Discrete tools (25%@2, 25%@4, 25%@16, 25%@350) ---")
    sweep(d_discrete, gpu_exp)

    print("\n  --- Bimodal tools (90%@1, 10%@900) ---")
    sweep(d_bimodal, gpu_exp)
