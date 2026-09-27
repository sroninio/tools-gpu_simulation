"""
All known tool time distributions for agentic workloads.

Usage
-----
    from distributions import ALL, get

    d = get('Codex CLI')           # standard 20-bucket version
    d = get('Codex CLI granular')  # 2000-bucket fine-grained version

    for name, d in ALL:
        print(name, d.mean)

Source: version_2_tool_time_cdfs.py (P05–P99 percentile buckets, 20 points each)
        codex_cli_granular.py      (Codex CLI split into 2000 sub-buckets)
"""

from version_2_tool_time_cdfs import ALL_DISTRIBUTIONS as _V2
from codex_cli_granular import codex_cli_granular as _codex_granular

# ── catalogue ─────────────────────────────────────────────────────────────────
#
# Name                       Mean (s)   P50 (s)   P99 (s)   Notes
# ─────────────────────────────────────────────────────────────────────────────
# Codex CLI                  27.4       8.3       410.4     Main test dist; heavy tail
# Codex CLI granular         27.4       —         410.4     Same dist, 2000 buckets
# SemiAnalysis CC (1M)       46.4       2.3       1393.2    Very heavy tail
# SemiAnalysis CC (256K)     24.4       1.2       736.1     Heavy tail
# Baseten (gap+decode)       38.1       9.3       652.0
# Multi-agent Swarm CC       12.0       3.7       178.6
# NV Employee Traces          7.7       1.2       160.2
# Cursor - SOLBench           9.6       5.6        75.1
# Together AI                 5.3       1.8        72.0
# SWE Coding agent            1.7       0.4        29.1
# Claude Code (tool-call)     3.4       1.0        47.7
# AC: SWE                     1.7       0.4        29.1     Same as SWE Coding agent
# AC: SampledWorkflow         2.4       2.0         9.9     Narrow distribution
# AC: AgentGuidance           1.3       0.6        12.9
# AC: Code Review             0.6       0.4         3.5     Fastest / narrowest
# AA-AgentPerf-v2             1.2       1.0         5.0

_v2_dict = dict(_V2)

ALL = list(_V2) + [('Codex CLI granular', _codex_granular)]


def get(name):
    """Return distribution by name. Raises KeyError if not found."""
    if name == 'Codex CLI granular':
        return _codex_granular
    if name in _v2_dict:
        return _v2_dict[name]
    raise KeyError(f"Unknown distribution: {name!r}. Available: {[n for n, _ in ALL]}")


if __name__ == '__main__':
    print(f"{'Name':<30} {'Mean(s)':>10} {'P50(s)':>10} {'P99(s)':>10}")
    print('-' * 65)
    from version_2_tool_time_cdfs import RAW
    for name, d in _V2:
        vals = RAW[name]
        print(f"{name:<30} {d.mean:>10.3f} {vals[9]:>10.3f} {vals[19]:>10.1f}")
    print(f"{'Codex CLI granular':<30} {_codex_granular.mean:>10.3f} {'—':>10} {'410.4':>10}")
