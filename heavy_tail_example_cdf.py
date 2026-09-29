"""
Synthetic heavy-tail tool time distribution — example only.

This is a log-normal distribution (mu=1.689, sigma=2.0) with mean ~40s,
inspired by the shape of real agentic tool time CDFs but NOT derived from
any real workload data. Safe to share publicly.

Properties:
    mean  ~  40s
    P50   ~   5.4s
    P95   ~ 145s
    P99   ~ 568s
    CV    ~   3.8  (heavy tail)
"""

from sim import make_discrete
from version_2_tool_time_cdfs import PCTS

# Percentile values (P05..P99) — log-normal shape, scaled to mean=40s
_VALS = [
    0.2759,   # P05
    0.5710,   # P10
    0.9324,   # P15
    1.3770,   # P20
    1.9232,   # P25
    2.5948,   # P30
    3.4278,   # P35
    4.4617,   # P40
    5.7613,   # P45
    7.4062,   # P50
    9.5216,   # P55
   12.2896,   # P60
   16.0058,   # P65
   21.1396,   # P70
   28.5284,   # P75
   39.8355,   # P80
   58.7596,   # P85
   95.8816,   # P90
  198.3914,   # P95
  776.5423,   # P99
]


def _to_buckets(values):
    buckets = []
    for i in range(len(PCTS)):
        prob = PCTS[i] - (PCTS[i-1] if i > 0 else 0.0)
        v    = (values[i] + (values[i-1] if i > 0 else values[0])) / 2.0
        buckets.append((prob, v))
    return buckets


heavy_tail_example = make_discrete(_to_buckets(_VALS))


if __name__ == '__main__':
    print("Synthetic heavy-tail example CDF (log-normal, mean~40s)")
    print(f"  mean : {heavy_tail_example.mean:.2f}s")
    print(f"  P50  : {_VALS[9]:.3f}s")
    print(f"  P95  : {_VALS[18]:.1f}s")
    print(f"  P99  : {_VALS[19]:.1f}s")

    import matplotlib.pyplot as plt
    import numpy as np

    cum, ts, fs = 0.0, [], []
    buckets = _to_buckets(_VALS)
    buckets_sorted = sorted(buckets, key=lambda x: x[1])
    total_p = sum(p for p, _ in buckets_sorted)
    for p, t in buckets_sorted:
        cum += p / total_p
        ts.append(t)
        fs.append(cum)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, log in zip(axes, [False, True]):
        if log:
            ax.semilogx(ts, fs, color='darkorange', lw=2)
            ax.set_xlabel('Tool time T (s) — log scale')
        else:
            ax.plot(ts, fs, color='darkorange', lw=2)
            ax.set_xlabel('Tool time T (s)')
        ax.set_ylabel('F(T)')
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3, which='both')
        for p, label, ls in [(0.5, 'P50', '--'), (0.95, 'P95', ':'), (0.99, 'P99', '-.')]:
            ax.axhline(p, color='gray', ls=ls, lw=0.8, label=label)
        ax.legend()

    plt.suptitle(f'Synthetic heavy-tail CDF  (mean={heavy_tail_example.mean:.1f}s, log-normal)', fontsize=13)
    plt.tight_layout()
    out = '/Users/rspiegelman/Desktop/heavy_tail_example_cdf.png'
    plt.savefig(out, dpi=150)
    print(f"  plot : {out}")
