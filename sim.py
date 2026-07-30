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
    values  = [v for _, v in buckets]
    weights = [p for p, _ in buckets]
    mean    = sum(p * v for p, v in buckets)
    def sampler(rng): return rng.choices(values, weights=weights)[0]
    sampler.mean = mean
    sampler.name = "Discrete(" + ", ".join(f"{int(p*100)}%→{v}" for p, v in buckets) + ")"
    return sampler

def make_constant(value):
    def sampler(rng): return value
    sampler.mean = value
    sampler.name = f"Constant({value})"
    return sampler


def run_simulation(d_tools, d_gpu, K, num_gpus=1,
                   n_warmup=10_000, n_measure=200_000, seed=42):
    rng = random.Random(seed)

    def sample_tools(): return d_tools(rng)
    def sample_gpu():   return d_gpu(rng)

    heap = []
    eid  = 0

    # All K requests start at Tools simultaneously (infinite servers, no wait)
    for i in range(K):
        heapq.heappush(heap, (sample_tools(), eid, TOOLS_DONE, i, 0.0))
        eid += 1

    gpu_free       = num_gpus   # number of idle GPU servers
    gpu_queue      = deque()

    gpu_busy_time  = 0.0        # accumulated service time across all GPUs (post warmup)
    gpu_done_count = 0
    measured       = 0
    warmup_done    = False
    measure_start  = 0.0
    t              = 0.0

    while measured < n_measure:
        t, _, etype, req, svc = heapq.heappop(heap)

        if etype == TOOLS_DONE:
            if gpu_free > 0:
                gpu_free -= 1
                service_time = sample_gpu()
                heapq.heappush(heap, (t + service_time, eid, GPU_DONE, req, service_time))
                eid += 1
            else:
                gpu_queue.append(req)

        else:  # GPU_DONE
            gpu_done_count += 1

            if warmup_done:
                gpu_busy_time += svc   # svc = service time carried in the event
                measured      += 1
            elif gpu_done_count >= n_warmup:
                warmup_done   = True
                measure_start = t

            # assign freed GPU to next queued request, or mark it idle
            if gpu_queue:
                next_req     = gpu_queue.popleft()
                service_time = sample_gpu()
                heapq.heappush(heap, (t + service_time, eid, GPU_DONE, next_req, service_time))
                eid += 1
            else:
                gpu_free += 1

            # return completed request to Tools
            heapq.heappush(heap, (t + sample_tools(), eid, TOOLS_DONE, req, 0.0))
            eid += 1

    wall = t - measure_start
    # utilization = busy GPU-time / total available GPU-time
    return gpu_busy_time / (num_gpus * wall)


def sweep(d_tools, d_gpu, num_gpus=1, k_multiplier=3, target_util=0.9999,
          n_warmup=10_000, n_measure=200_000, seed=42):
    k_star = int(d_tools.mean / d_gpu.mean * num_gpus) + num_gpus
    k_end  = k_star * k_multiplier
    step   = max(1, k_star // 30)

    print(f"\ntools_avg={d_tools.mean}  gpu_avg={d_gpu.mean}  num_gpus={num_gpus}  K*={k_star}")
    print(f"  d_tools = {d_tools.name}")
    print(f"  d_gpu   = {d_gpu.name}")
    print()
    print(f"{'K':>6} | {'K/K*':>6} | {'GPU Util':>9}")
    print("-" * 30)

    K = k_star
    while K <= k_end:
        util   = run_simulation(d_tools, d_gpu, K, num_gpus=num_gpus,
                                n_warmup=n_warmup, n_measure=n_measure, seed=seed)
        marker = "  <-- K*" if K == k_star else ""
        print(f"{K:>6} | {K/k_star:>6.2f} | {util:>8.2%}{marker}")
        if util >= target_util:
            break
        K += step


if __name__ == "__main__":
    d_const    = make_constant(90.0)
    d_exp      = make_exponential(90.0)
    d_discrete = make_discrete([(0.25, 2), (0.25, 4), (0.25, 16), (0.25, 350)])
    d_bimodal  = make_discrete([(0.90, 1), (0.10, 900)])
    gpu        = make_constant(0.3)

    for num_gpus in [1, 4]:
        print(f"\n{'='*50}")
        print(f"  num_gpus = {num_gpus}")
        print(f"{'='*50}")
        for d_tools in [d_const, d_exp, d_discrete, d_bimodal]:
            sweep(d_tools, gpu, num_gpus=num_gpus)
