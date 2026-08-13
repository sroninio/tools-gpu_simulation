import sys, math, json
sys.path.insert(0, '/Users/rspiegelman/gpu-utilization-sim')
import sim
from cmx_calc import find_min_cmx_bw

gpu      = sim.make_constant(1/3.45)
GB       = 2.5
num_gpus = 50

raw_distributions = [
    ('codex (local) [raw]', [
        (13.7616,0.0206),(15.9835,0.179),(23.6868,0.558),(25.5720,2.24),
        (10.5527,11.3),(5.5984,37.8),(2.1434,83.7),(1.4423,185),
        (0.6366,416),(0.5181,860),(0.0846,2130),(0.0200,15800),
    ]),
    ('codex (local), decode-shifted', [
        (0.0031,0.182),(1.8251,0.58),(32.0423,2.36),(47.9211,10.5),
        (11.4398,37.2),(3.6165,82.8),(1.8313,184),(0.6765,415),
        (0.5351,861),(0.0892,2130),(0.0138,5890),(0.0062,37900),
    ]),
    ('NV-emp subdag (local) [raw]', [
        (13.2787,0.0206),(36.8098,0.173),(17.6520,0.557),(14.6750,2.3),
        (6.9050,11.6),(3.0792,39),(2.2578,84.8),(2.6959,186),
        (1.4876,413),(0.8998,878),(0.1789,2200),(0.0803,14400),
    ]),
    ('SemiAnalysis CC 1M ctx [raw]', [
        (0.3802,0.0215),(1.7210,0.182),(8.3614,0.576),(48.6285,2.24),
        (25.1059,11),(4.7204,39),(3.0605,85),(4.8007,183),
        (0.9198,425),(1.0207,939),(0.5805,2300),(0.7004,36100),
    ]),
    ('SemiAnalysis CC 256K ctx [raw]', [
        (0.6218,0.0215),(2.6206,0.182),(13.4220,0.573),(60.1094,2.1),
        (10.8036,11.4),(2.8988,39.2),(2.0825,85.3),(4.0994,185),
        (0.8411,426),(0.8999,950),(0.6196,2320),(0.9813,36600),
    ]),
    ('Multi-agent CC Swarm [raw]', [
        (41.0688,2.32),(48.4083,9.91),(3.9610,38.6),(2.1806,84.6),
        (2.7405,182),(0.7801,417),(0.7600,839),(0.0403,2310),
        (0.0403,5930),(0.0201,10800),
    ]),
    ('NV-emp subdag, decode-shifted', [
        (0.0621,0.0215),(1.0020,0.182),(1.2156,0.581),(15.9783,2.43),
        (49.4442,11),(14.8210,38.2),(7.4452,83.8),(5.0049,186),
        (3.0134,411),(1.7486,846),(0.1844,2200),(0.0803,14400),
    ]),
    ('Claude Code [raw]', [
        (58.6826,0.0163),(7.4851,0.179),(3.5928,0.575),(15.1197,2.27),
        (11.9761,10.1),(2.8443,34),(0.2994,60),
    ]),
    ('NV Employee Traces [raw]', [
        (18,2.42),(54,10.8),(8,39),(8,84.1),(10,172),(2,300),
    ]),
]

all_results = {}

for num_gpus in [16, 32]:
    results = []
    print(f'\n=== {num_gpus} GPUs ===')
    header = '%-40s | %6s | %9s | %8s | %9s | %9s' % (
        'Distribution', 'Mean', 'Cap*/gpu', 'Util@K*', 'Cap@99%', 'MinCMXBW')
    print(header)
    print('-' * len(header))

    for name, buckets in raw_distributions:
        d    = sim.make_discrete([(p/100, v) for p, v in buckets])
        mean = d.mean
        k_star   = math.ceil(mean / gpu.mean * num_gpus) + num_gpus
        cap_star = k_star * GB / num_gpus

        n_warmup  = 200_000 if mean > 50 else 50_000
        n_measure = 5_000_000 if mean > 50 else 500_000

        # 1. util at K*
        u_kstar = sim.run_simulation(d, gpu, k_star, num_gpus=num_gpus,
                                     n_warmup=n_warmup, n_measure=n_measure)

        # 2. capacity for 99% util — binary search on K
        lo_k, hi_k = k_star, k_star * 3
        for _ in range(20):
            mid_k = (lo_k + hi_k) // 2
            u = sim.run_simulation(d, gpu, mid_k, num_gpus=num_gpus,
                                   n_warmup=n_warmup, n_measure=n_measure)
            if u >= 0.99: hi_k = mid_k
            else:         lo_k = mid_k
        cap_99 = hi_k * GB / num_gpus

        # 3. min CMX BW (instant)
        hist_cmx = list(buckets)
        min_bw, tau, frac = find_min_cmx_bw(hist_cmx)
        bw_str = '%.4f' % min_bw if min_bw is not None else 'N/A'

        row = '%-40s | %6.1f | %8.0fGB | %7.1f%% | %8.0fGB | %s GB/s' % (
            name, mean, cap_star, u_kstar*100, cap_99, bw_str)
        print(row, flush=True)

        results.append({
            'name': name, 'num_gpus': num_gpus, 'mean': mean,
            'cap_star_gb': cap_star, 'util_at_kstar': round(u_kstar*100, 2),
            'cap_99pct_gb': cap_99, 'min_cmx_bw_gbs': min_bw,
            'best_tau': tau, 'best_frac': frac,
        })

    all_results[str(num_gpus)] = results

json.dump(all_results, open('/Users/rspiegelman/Desktop/cmx_table_results.json', 'w'), indent=2)
print('\nSaved to ~/Desktop/cmx_table_results.json')
