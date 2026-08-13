"""
Python port of cmx_gain_v2.html calculator logic.
Functions: compute_reqs_sec, best_reqs_at_bw, auto_optimize, find_min_cmx_bw
"""

# Fixed constants per user spec
SOL   = 3.45        # GPU_SOL_REQ_SEC (req/s)
HBM   = 700         # HBM_DRAM_SIZE_GB
RECOMP = 1e-7       # RECOMPUTE_REQ_SEC
AVG   = 2.5         # AVG_AGENT_CAP_SIZE_IN_GB


def compute_reqs_sec(hist, tau, tau_frac, cmx_bw):
    """
    Exact port of JS computeReqsSec.
    hist: list of (pct, lat) tuples, pct in [0,100]
    Returns req/s or -1 if invalid.
    """
    total_pct = sum(p for p, _ in hist)
    if abs(total_pct - 100) > 0.01:
        return -1
    T = sum(p * l for p, l in hist) / 100
    if T <= 0:
        return -1

    cmx_rows   = [(p, l) for p, l in hist if l >  tau]
    at_tau_rows = [(p, l) for p, l in hist if l == tau]
    hbm_rows   = [(p, l) for p, l in hist if l <  tau]

    cmx_pct = (sum(p for p, _ in cmx_rows)
               + sum(p * (1 - tau_frac) for p, _ in at_tau_rows))
    hbm_pct = 100 - cmx_pct
    cmx_ratio = cmx_pct / 100

    if cmx_ratio <= 0 or cmx_ratio >= 1:
        return -1

    cmx_ttools = (
        sum(p * l for p, l in cmx_rows)
        + sum(p * (1 - tau_frac) * l for p, l in at_tau_rows)
    ) / (cmx_ratio * 100)

    hbm_ttools = (
        sum(p * l for p, l in hbm_rows)
        + sum(p * tau_frac * l for p, l in at_tau_rows)
    ) / hbm_pct if hbm_pct > 0 else 0

    slope           = 1.0 / (AVG * T)
    cmx_cap_littles = cmx_bw * cmx_ttools
    cmx_cap_ratio   = (HBM * (cmx_ratio / (1 - cmx_ratio)) * (cmx_ttools / hbm_ttools)
                       if hbm_ttools > 0
                       else HBM * cmx_ratio / (1 - cmx_ratio))

    if cmx_cap_ratio > cmx_cap_littles:
        return -1   # not valid: ratio bound exceeds BW bound

    cmx_t_gb_eff = min(cmx_cap_littles, cmx_cap_ratio)
    x_c_gb       = HBM + cmx_t_gb_eff
    return min(x_c_gb * slope, SOL)


def best_reqs_at_bw(hist, cmx_bw):
    """Sweep all tau/frac combos, return best req/s achievable at this BW."""
    tau_candidates = sorted(set(l for _, l in hist))
    best = -1
    best_tau = None
    best_frac = None
    for tau in tau_candidates:
        for fi in range(101):
            frac = fi / 100
            r = compute_reqs_sec(hist, tau, frac, cmx_bw)
            if r > best:
                best = r
                best_tau = tau
                best_frac = frac
    return best, best_tau, best_frac


def find_min_cmx_bw(hist, tol=1e-6):
    """
    Binary search: find minimum CMX BW to reach SOL.
    Returns (min_bw, tau, frac) or (None, None, None) if unreachable.
    Returns (0, None, None) if HBM alone already reaches SOL.
    """
    T = sum(p * l for p, l in hist) / 100
    if T > 0 and (1.0 / (AVG * T)) * HBM >= SOL - tol:
        return 0.0, None, None  # HBM alone sufficient, no CMX BW needed

    max_req, _, _ = best_reqs_at_bw(hist, 1e9)
    if max_req < SOL - tol:
        return None, None, None

    lo, hi = 0.0, 1e6
    for _ in range(60):
        mid = (lo + hi) / 2
        r, _, _ = best_reqs_at_bw(hist, mid)
        if r >= SOL - tol:
            hi = mid
        else:
            lo = mid

    _, best_tau, best_frac = best_reqs_at_bw(hist, hi)
    return hi, best_tau, best_frac


if __name__ == '__main__':
    # Verification test: default histogram from CONTEXT.md (50%@0, 50%@120)
    # with CMX_BW=4, tau=0, tau_frac=0.15
    hist_default = [(50, 0), (50, 120)]
    r = compute_reqs_sec(hist_default, tau=0, tau_frac=0.15, cmx_bw=4)
    print(f'Default hist, tau=0, frac=0.15, BW=4 -> {r:.6f} req/s')

    # Verification test: 25%@2, 25%@8, 25%@12, 25%@340 — the "±Codex CLI" proxy
    hist_proxy = [(25, 2), (25, 8), (25, 12), (25, 340)]
    r2, t2, f2 = best_reqs_at_bw(hist_proxy, cmx_bw=4)
    print(f'Proxy hist, BW=4, best tau={t2}, frac={f2:.2f} -> {r2:.6f} req/s')

    bw, tau, frac = find_min_cmx_bw(hist_proxy)
    print(f'Proxy hist, min CMX BW to SOL: {bw:.4f} GB/s  tau={tau}  frac={frac:.2f}')
