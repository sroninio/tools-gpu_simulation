"""
Codex CLI tool time CDF — granular version (2000 sub-buckets).

Each of the 20 original percentile buckets is split into 100 equal
sub-buckets with linearly spaced midpoints. This gives much finer
resolution for solving the θ* equation analytically.

Mean = 27.418s  (matches original 20-bucket version exactly)
"""

from sim import make_discrete
from version_2_tool_time_cdfs import RAW, PCTS

_vals = RAW['Codex CLI']
_sub  = 100

_fine_buckets = []
for i in range(len(PCTS)):
    p_bucket = PCTS[i] - (PCTS[i-1] if i > 0 else 0.0)
    lo = _vals[i-1] if i > 0 else _vals[0]
    hi = _vals[i]
    p_sub = p_bucket / _sub
    for j in range(_sub):
        t = lo + (hi - lo) * (j + 0.5) / _sub
        _fine_buckets.append((p_sub, t))

# sorted for formula sweeps
SORTED_BUCKETS = sorted(_fine_buckets, key=lambda x: x[1])
TOTAL_P        = sum(p for p, _ in SORTED_BUCKETS)
THETAS         = sorted(set(t for _, t in SORTED_BUCKETS))

# callable distribution for simulation
codex_cli_granular = make_discrete(_fine_buckets)


def cdf_stats(theta):
    """F(theta) and E[T | T <= theta] for the granular distribution."""
    cum_p, cum_pt = 0.0, 0.0
    for p, t in SORTED_BUCKETS:
        if t <= theta:
            cum_p  += p / TOTAL_P
            cum_pt += p / TOTAL_P * t
    F  = cum_p
    ET = cum_pt / F if F > 0 else 0.0
    return F, ET


if __name__ == '__main__':
    print(f"Codex CLI granular CDF")
    print(f"  buckets : {len(_fine_buckets)}")
    print(f"  mean    : {codex_cli_granular.mean:.3f}s")
    print(f"  min t   : {THETAS[0]:.4f}s")
    print(f"  max t   : {THETAS[-1]:.4f}s")
    F95, ET95 = cdf_stats(_vals[18])
    F99, ET99 = cdf_stats(_vals[19])
    print(f"  F(P95={_vals[18]:.1f}s) = {F95:.4f}   E[T|T<P95] = {ET95:.3f}s")
    print(f"  F(P99={_vals[19]:.1f}s) = {F99:.4f}   E[T|T<P99] = {ET99:.3f}s")
