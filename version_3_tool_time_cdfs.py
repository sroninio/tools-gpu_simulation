"""
Version 3 tool time CDFs — same as v2 but with P99 removed.
The remaining 19 percentile points (P05–P95) are renormalized to sum to 1.0,
effectively treating P95 as the tail cutoff.
Source: version_2_tool_time_cdfs.py
"""

from sim import make_discrete
from version_2_tool_time_cdfs import RAW, PCTS


PCTS_V3 = PCTS[:-1]  # P05..P95 (drop P99)


def _percentile_to_buckets_v3(values):
    """Convert 19 percentile values (P05–P95) to normalized discrete buckets."""
    vals19 = values[:-1]  # drop P99
    raw_buckets = []
    for i in range(len(PCTS_V3)):
        prob = PCTS_V3[i] - (PCTS_V3[i-1] if i > 0 else 0.0)
        v    = (vals19[i] + (vals19[i-1] if i > 0 else vals19[0])) / 2.0
        raw_buckets.append((prob, v))
    # renormalize: raw probs sum to 0.95
    total = sum(p for p, _ in raw_buckets)
    return [(p / total, v) for p, v in raw_buckets]


ALL_DISTRIBUTIONS = [
    (name, make_discrete(_percentile_to_buckets_v3(vals)))
    for name, vals in RAW.items()
]


if __name__ == '__main__':
    from version_2_tool_time_cdfs import ALL_DISTRIBUTIONS as V2
    v2_means = {name: d.mean for name, d in V2}

    print('Version 3 tool time CDFs (P99 removed, renormalized):')
    print('%-30s %10s %10s %10s %10s' % ('Workload', 'Mean_v2(s)', 'Mean_v3(s)', 'P95(s)', 'Drop%'))
    print('-' * 75)
    for name, d in ALL_DISTRIBUTIONS:
        p95 = RAW[name][18]
        drop = (v2_means[name] - d.mean) / v2_means[name] * 100
        print('%-30s %10.2f %10.2f %10.1f %9.1f%%' % (name, v2_means[name], d.mean, p95, drop))
