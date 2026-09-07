"""
Compute and recompute BW for multiple models and sequence configs.
sq format: cached_ISL - new_ISL - OSL

reqs_per_gpu          : compute throughput (req/GPU/sec)
reqs_per_gpu_recompute: recompute throughput (req/GPU/sec)

Usage (same as gpu_req_sec=3.45 in original sim):
    from sim import make_constant
    d_gpu       = make_constant(1 / reqs_per_gpu)
    d_recompute = make_constant(1 / reqs_per_gpu_recompute)
"""

MODELS_BW = [
    # (model,        sq,                    reqs_per_gpu, reqs_per_gpu_recompute)
    ('DSv3',         '131072-2048-1000',    1.6988,       0.0956),
    ('DSv4-1.6T',    '131072-4096-400',     12.9746,      0.5710),
    ('DSv4-1.6T',    '1048576-4096-200',    6.1175,       0.0400),
    ('DSv4-1.6T',    '204800-4096-200',     13.0438,      0.3531),
    ('DSv4-1.6T',    '409600-4096-200',     10.7051,      0.1524),
    ('GPT-OSS-2T',   '131072-2048-1000',    5.0380,       0.4156),
    # ('GPT-OSS-2T', '1048576-4096-200',   None,          None),   # missing data
    ('GPT-OSS-2T',   '204800-4096-200',     4.6365,       0.2129),
    ('GPT-OSS-2T',   '409600-4096-200',     2.4520,       0.0596),
    ('KimiK2',       '131072-2048-1000',    3.1607,       0.3558),
    ('KimiK2',       '1048576-4096-200',    0.6678,       0.0074),
    ('KimiK2',       '204800-4096-200',     3.2222,       0.1711),
    ('Qwen3-235B',   '131072-2048-1000',    2.8728,       0.2268),
    ('Qwen3-235B',   '1048576-4096-200',    0.4385,       0.0040),
    ('Qwen3-235B',   '204800-4096-200',     2.1621,       0.1003),
    ('Qwen3-235B',   '409600-4096-200',     1.1030,       0.0259),
]


if __name__ == '__main__':
    print('%-15s %-22s %10s %13s %16s' % ('Model','sq','reqs/gpu','reqs/gpu_rc','recompute_ratio'))
    print('-' * 80)
    for model, sq, rg, rgr in MODELS_BW:
        ratio = rg / rgr if rgr else 0
        print('%-15s %-22s %10.4f %13.4f %14.1fx' % (model, sq, rg, rgr, ratio))
