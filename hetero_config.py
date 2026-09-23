"""
Heterogeneous LRU simulation — GPT-OSS-2T mixed SQ types
Capacity-constrained storage (GB), slot release on session death.

Results (S=10, dead_session_fix=True, top-4 combined CDF right-edge, 1 GPU):

  K_mult    K     C(GB)    util    miss%
  0.50     113    339.7   61.5%    3.1%
  0.75     170    511.0   79.9%    1.2%
  1.00     226    679.3   96.9%    0.2%   <- K*
  1.25     282    847.7  100.0%    0.0%
  1.50     339   1019.0  100.0%    0.0%

99% utilization at K=239, C=718.4GB (1.058 x K*)
"""

# SQ mixture: (prob, rps_normal, rps_recompute, kv_gb)
SQ_TYPES = [
    (0.750, 5.0380, 0.4156, 131072 * 18 / 1048576),  # 131K-2K-1K  2.25GB  12.1x recompute
    (0.125, 4.6365, 0.2129, 204800 * 18 / 1048576),  # 204K-4K-200 3.52GB  21.8x recompute
    (0.125, 2.4520, 0.0596, 409600 * 18 / 1048576),  # 409K-4K-200 7.03GB  41.1x recompute
]

S            = 10       # steps per session
NUM_GPUS     = 1
DEAD_SESSION_FIX = True  # slot released immediately on session death

# Derived
# reqs_eff  = 4.409 req/s  (weighted harmonic mean)
# mean_kv   = 3.006 GB     (weighted average KV size)
# K_star    = 226          (Little's Law: ceil(mean_tools * reqs_eff) + 1)
# C_at_Kstar = 679.3 GB    (K* x mean_kv)
# mean_tools = 50.8s       (top-4 combined CDF, right-edge)

# Tools CDF: top-4 combined equal-weight right-edge
TOP4_NAMES = [
    'SemiAnalysis CC (1M)',
    'SemiAnalysis CC (256K)',
    'Codex CLI',
    'Baseten (gap+decode)',
]
