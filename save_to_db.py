"""
Saves CMX table results to SQLite for later analysis.
Run after run_table.py completes.
"""
import sqlite3, json

DB_PATH   = '/Users/rspiegelman/Desktop/cmx_results.db'
JSON_PATH = '/Users/rspiegelman/Desktop/cmx_table_results.json'

conn = sqlite3.connect(DB_PATH)
c    = conn.cursor()

c.execute('''
    CREATE TABLE IF NOT EXISTS cmx_results (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        run_tag         TEXT,
        distribution    TEXT,
        num_gpus        INTEGER,
        gpu_latency_s   REAL,
        mean_s          REAL,
        k_star_per_gpu  INTEGER,
        cap_star_gb     REAL,
        util_at_kstar   REAL,
        cap_99pct_gb    REAL,
        min_cmx_bw_gbs  REAL,
        best_tau        REAL,
        best_frac       REAL,
        hbm_gb          REAL,
        avg_agent_gb    REAL,
        sol_reqs        REAL
    )
''')

results  = json.load(open(JSON_PATH))
run_tag  = 'run_2026-08-02'
num_gpus = 50
gpu_lat  = round(1/3.45, 6)
hbm      = 700
avg      = 2.5
sol      = 3.45

for r in results:
    c.execute('''
        INSERT INTO cmx_results
        (run_tag, distribution, num_gpus, gpu_latency_s, mean_s,
         k_star_per_gpu, cap_star_gb, util_at_kstar, cap_99pct_gb,
         min_cmx_bw_gbs, best_tau, best_frac, hbm_gb, avg_agent_gb, sol_reqs)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', (run_tag, r['name'], num_gpus, gpu_lat, r['mean'],
          r['k_star_per_gpu'], r['cap_star_gb'], r['util_at_kstar'],
          r['cap_99pct_gb'], r['min_cmx_bw_gbs'], r['best_tau'],
          r['best_frac'], hbm, avg, sol))

conn.commit()
conn.close()

print(f'Saved {len(results)} rows to {DB_PATH}')
print('\nQuery example:')
print('  sqlite3 ~/Desktop/cmx_results.db "SELECT distribution, util_at_kstar, cap_99pct_gb, min_cmx_bw_gbs FROM cmx_results"')
