"""
Version 1 tool time CDFs — from cdf_export_for_cmx_tool.md
Inter-turn gap distributions (tool execution + human think time).
All buckets in format: (probability, latency_seconds)
"""

from sim import make_discrete

# 1. codex (local) [raw]  — mean=20.6s, CV=11.81
codex_local_raw = make_discrete([
    (13.7616/100, 0.0206), (15.9835/100, 0.179), (23.6868/100, 0.558),
    (25.5720/100, 2.24),  (10.5527/100, 11.3),  (5.5984/100, 37.8),
    (2.1434/100,  83.7),  (1.4423/100,  185),   (0.6366/100, 416),
    (0.5181/100,  860),   (0.0846/100,  2130),  (0.0200/100, 15800),
])

# 2. codex (local), decode-shifted  — mean=28.9s, CV=11.12
codex_local_decoded = make_discrete([
    (0.0031/100, 0.182),  (1.8251/100, 0.58),   (32.0423/100, 2.36),
    (47.9211/100, 10.5),  (11.4398/100, 37.2),  (3.6165/100, 82.8),
    (1.8313/100,  184),   (0.6765/100, 415),    (0.5351/100, 861),
    (0.0892/100,  2130),  (0.0138/100, 5890),   (0.0062/100, 37900),
])

# 3. NV-emp subdag (local) [raw]  — mean=39.0s, CV=11.01
nv_subdag_local_raw = make_discrete([
    (13.2787/100, 0.0206), (36.8098/100, 0.173), (17.6520/100, 0.557),
    (14.6750/100, 2.3),    (6.9050/100,  11.6),  (3.0792/100, 39),
    (2.2578/100,  84.8),   (2.6959/100,  186),   (1.4876/100, 413),
    (0.8998/100,  878),    (0.1789/100,  2200),  (0.0803/100, 14400),
])

# 4. SemiAnalysis CC 1M ctx [raw]  — mean=296.8s, CV=10.15
semianalysis_1m_raw = make_discrete([
    (0.3802/100, 0.0215), (1.7210/100, 0.182), (8.3614/100, 0.576),
    (48.6285/100, 2.24),  (25.1059/100, 11),   (4.7204/100, 39),
    (3.0605/100,  85),    (4.8007/100,  183),  (0.9198/100, 425),
    (1.0207/100,  939),   (0.5805/100,  2300), (0.7004/100, 36100),
])

# 5. Codex CLI [raw]  — mean=60.6s, CV=9.36
codex_cli_raw = make_discrete([
    (0.2235/100, 0.182),  (0.3352/100, 0.581),  (26.2569/100, 2.39),
    (55.9777/100, 10.3),  (8.0447/100, 38.2),   (5.0279/100, 82.7),
    (2.9050/100,  179),   (0.4470/100, 422),    (0.1117/100, 971),
    (0.3352/100,  2280),  (0.1117/100, 6320),   (0.2235/100, 10800),
])

# 6. SemiAnalysis CC 256K ctx [raw]  — mean=398.7s, CV=9.05
semianalysis_256k_raw = make_discrete([
    (0.6218/100, 0.0215), (2.6206/100, 0.182),  (13.4220/100, 0.573),
    (60.1094/100, 2.1),   (10.8036/100, 11.4),  (2.8988/100, 39.2),
    (2.0825/100,  85.3),  (4.0994/100,  185),   (0.8411/100, 426),
    (0.8999/100,  950),   (0.6196/100,  2320),  (0.9813/100, 36600),
])

# 7. Multi-agent CC Swarm [raw]  — mean=29.2s, CV=7.40
multiagent_swarm_raw = make_discrete([
    (41.0688/100, 2.32),  (48.4083/100, 9.91),  (3.9610/100, 38.6),
    (2.1806/100,  84.6),  (2.7405/100,  182),   (0.7801/100, 417),
    (0.7600/100,  839),   (0.0403/100,  2310),  (0.0403/100, 5930),
    (0.0201/100,  10800),
])

# 8. NV-emp subdag, decode-shifted  — mean=69.8s, CV=6.25
nv_subdag_decoded = make_discrete([
    (0.0621/100, 0.0215), (1.0020/100, 0.182),  (1.2156/100, 0.581),
    (15.9783/100, 2.43),  (49.4442/100, 11),    (14.8210/100, 38.2),
    (7.4452/100,  83.8),  (5.0049/100,  186),   (3.0134/100, 411),
    (1.7486/100,  846),   (0.1844/100,  2200),  (0.0803/100, 14400),
])

# 9. Claude Code [raw]  — mean=2.7s, CV=2.56
claude_code_raw = make_discrete([
    (58.6826/100, 0.0163), (7.4851/100, 0.179), (3.5928/100, 0.575),
    (15.1197/100, 2.27),   (11.9761/100, 10.1), (2.8443/100, 34),
    (0.2994/100,  60),
])

# 10. NV Employee Traces [raw]  — mean=39.3s, CV=1.60
nv_employee_traces_raw = make_discrete([
    (18/100, 2.42), (54/100, 10.8), (8/100, 39),
    (8/100, 84.1),  (10/100, 172),  (2/100, 300),
])

# ── Convenience list ──
ALL_DISTRIBUTIONS = [
    ('codex (local) [raw]',              codex_local_raw),
    ('codex (local), decode-shifted',    codex_local_decoded),
    ('NV-emp subdag (local) [raw]',      nv_subdag_local_raw),
    ('SemiAnalysis CC 1M ctx [raw]',     semianalysis_1m_raw),
    ('Codex CLI [raw]',                  codex_cli_raw),
    ('SemiAnalysis CC 256K ctx [raw]',   semianalysis_256k_raw),
    ('Multi-agent CC Swarm [raw]',       multiagent_swarm_raw),
    ('NV-emp subdag, decode-shifted',    nv_subdag_decoded),
    ('Claude Code [raw]',                claude_code_raw),
    ('NV Employee Traces [raw]',         nv_employee_traces_raw),
]


if __name__ == '__main__':
    print('Version 1 tool time CDFs:')
    for name, d in ALL_DISTRIBUTIONS:
        import math
        var = sum(p*v**2 for p,v in zip(
            [b[0] for b in d.__self__[0]] if hasattr(d,'__self__') else [],[]
        )) if False else 0
        print(f'  {name:<40} mean={d.mean:.1f}s')
