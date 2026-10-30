#!/usr/bin/env python3
"""Gather training time + hardware usage of all batch runs into one table (for the paper).

Reads sb3/logs/*/run_info/{job.json,gpu_usage.csv} written by train_vec.sbatch, adds Slurm per-job accounting
(sacct on the .batch step: elapsed, total CPU time, MaxRSS, GPU util/memory) and the achieved steps from progress.csv.
Run on the HOST, not in the container (the container has no sacct).
CPU/RAM come from Slurm only: cpu_load.csv (top over all of the user's processes) counts other jobs on a shared node.
usage: python3 infra/collect_run_stats.py [--filter SUBSTR ...] [--out results/<race2|race_auv>/run_stats]
Writes <out>.csv (one row per run), <out>.md (per-run + per vehicle/algorithm/config mean +- std).
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p = argparse.ArgumentParser()
p.add_argument('--filter', nargs='*', default=[], help='keep runs whose dir contains ALL of these substrings')
p.add_argument('--out', default=os.path.join(ROOT, 'results', 'race2', 'run_stats'))
args = p.parse_args()


def seconds(t):
    """Slurm duration [D-]HH:MM:SS[.ms] or MM:SS[.ms] -> s."""
    if not t:
        return None
    d, _, t = t.rpartition('-')
    parts = [float(x) for x in t.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    return (int(d) if d else 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def size_gb(s):
    m = re.match(r'([\d.]+)([KMGT]?)', s or '')
    return float(m[1]) * {'': 1 / 1024 ** 3, 'K': 1 / 1024 ** 2, 'M': 1 / 1024, 'G': 1, 'T': 1024}[m[2]] if m else None


def sacct(job_id):
    try:
        out = subprocess.run(['sacct', '-j', str(job_id), '-n', '-P',
                              '--format=JobID,Elapsed,TotalCPU,MaxRSS,State,TRESUsageInTot'],
                             capture_output=True, text=True).stdout.strip().splitlines()
    except FileNotFoundError:
        raise SystemExit('sacct not found: run this on the host, not inside the container')
    res = {}
    for line in out:
        jid, el, cpu, rss, state, tres = (line.split('|') + [''] * 6)[:6]
        if jid == str(job_id):
            res.update(elapsed_s=seconds(el), state=state.split()[0] if state else '')
        if jid == f'{job_id}.batch':   # the training runs in the batch step
            res.update(cpu_s=seconds(cpu), max_rss_gb=size_gb(rss))
            t = dict(kv.split('=', 1) for kv in tres.split(',') if '=' in kv)
            res.update(gpu_util_pct=float(t['gres/gpuutil']) if 'gres/gpuutil' in t else None,
                       gpu_mem_gb=size_gb(t.get('gres/gpumem')))
    return res


def read_csv(path):
    try:
        rows = list(csv.DictReader(open(path)))
        return {k: np.array([float(r[k]) for r in rows if r.get(k) not in (None, '')]) for k in rows[0]} if rows else {}
    except (OSError, ValueError):
        return {}


def rnd(x, n=1):
    return None if x is None else round(float(x), n)


rows = []
for jf in sorted(glob.glob(os.path.join(ROOT, 'sb3', 'logs', '*', 'run_info', 'job.json'))):
    run_dir = os.path.dirname(os.path.dirname(jf))
    if not all(f in run_dir for f in args.filter):
        continue
    j = json.load(open(jf))
    prog = read_csv(os.path.join(run_dir, 'progress.csv'))
    steps = int(np.nanmax(prog['time/total_timesteps'])) if 'time/total_timesteps' in prog else None
    # a run resumed over several jobs (dreamer --run_id): later parts' info is nested in run_info/job_<id>_*/
    part_dirs = [os.path.dirname(jf)] + sorted(glob.glob(os.path.join(os.path.dirname(jf), 'job_*')))
    parts = [(json.load(open(os.path.join(d, 'job.json'))), d) for d in part_dirs if os.path.exists(os.path.join(d, 'job.json'))]
    accts = [sacct(pj['job_id']) for pj, _ in parts]
    gs = [read_csv(os.path.join(d, 'gpu_usage.csv')) for _, d in parts]
    g = {k: np.concatenate([x[k] for x in gs if k in x]) for k in ('gpu_util_pct', 'gpu_mem_mib', 'gpu_power_w') if any(k in x for x in gs)}
    walls = [a.get('elapsed_s') or pj.get('wall_time_s') for a, (pj, _) in zip(accts, parts)]
    wall = sum(w for w in walls if w) or None
    cpu = sum(a['cpu_s'] for a in accts if a.get('cpu_s')) or None
    util = [(a['gpu_util_pct'], w) for a, w in zip(accts, walls) if a.get('gpu_util_pct') is not None and w]
    rss = [a['max_rss_gb'] for a in accts if a.get('max_rss_gb')]
    s = {'gpu_util_pct': sum(u * w for u, w in util) / sum(w for _, w in util) if util else None,
         'max_rss_gb': max(rss) if rss else None, 'gpu_mem_gb': max((a.get('gpu_mem_gb') or 0) for a in accts) or None,
         'state': '+'.join(a.get('state') or str(pj.get('exit_status')) for a, (pj, _) in zip(accts, parts))}
    st = j['settings']
    rows.append({
        'run': os.path.basename(run_dir), 'job_id': '+'.join(str(pj['job_id']) for pj, _ in parts), 'job_name': j['job_name'],
        'jobs': len(parts),
        'vehicle': st.get('RACE2_VEHICLE', 'race2'), 'trainer': j['trainer'],
        'config': 'articulated' if st.get('RACE2_SERVOS', '0') != '0' else 'fixed',
        'seed': (re.search(r'--seed (\d+)', j.get('extra_args', '')) or [None, None])[1],
        'settings': ' '.join(f"{k[6:].lower()}={v}" for k, v in sorted(st.items())),
        'node': j['node'], 'gpu': j['gpu'].split(',')[0], 'cpu_model': j.get('cpu_model', ''),
        'cpus_alloc': j['cpus_allocated'], 'mem_alloc_gb': rnd(int(j['mem_allocated_mb']) / 1024, 0),
        'sims': j['sims'], 'steps': steps,
        'wall_h': rnd(wall / 3600 if wall else None, 2),
        'steps_per_s': rnd(steps / wall if steps and wall else None),
        'cpu_cores_mean': rnd(cpu / wall if cpu and wall else None),
        'cpu_core_h': rnd(cpu / 3600 if cpu else None),
        'gpu_h': rnd(wall / 3600 if wall else None, 2),   # 1 GPU per job
        'ram_max_gb': rnd(s.get('max_rss_gb')),
        'gpu_util_pct': rnd(s.get('gpu_util_pct') if s.get('gpu_util_pct') is not None
                            else (np.mean(g['gpu_util_pct']) if 'gpu_util_pct' in g else None)),
        'gpu_mem_max_gb': rnd(np.max(g['gpu_mem_mib']) / 1024 if 'gpu_mem_mib' in g else s.get('gpu_mem_gb'), 2),
        'gpu_power_mean_w': rnd(np.mean(g['gpu_power_w']) if 'gpu_power_w' in g else None),
        'state': s['state'], 'git': json.dumps(j.get('git', {})),
    })

if not rows:
    raise SystemExit('no run_info/job.json found')
os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out + '.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

hw = sorted({(r['cpu_model'], r['gpu']) for r in rows})
cols = ['job_name', 'vehicle', 'trainer', 'config', 'seed', 'steps', 'wall_h', 'steps_per_s', 'sims', 'cpus_alloc',
        'cpu_cores_mean', 'cpu_core_h', 'ram_max_gb', 'gpu_util_pct', 'gpu_mem_max_gb', 'gpu_power_mean_w', 'state']
stat = ['steps', 'wall_h', 'steps_per_s', 'cpu_cores_mean', 'cpu_core_h', 'ram_max_gb', 'gpu_util_pct',
        'gpu_mem_max_gb', 'gpu_power_mean_w']
with open(args.out + '.md', 'w') as f:
    f.write('Hardware: ' + '; '.join(f'{c} + {g}' for c, g in hw) + '. One GPU per job; CPU, RAM and GPU util '
            'from Slurm per-job accounting (sacct, batch step); GPU memory/power from nvidia-smi every 30 s.\n\n')
    f.write('## Per algorithm (mean ± std over seeds)\n\n| vehicle | algorithm | config | runs | ' + ' | '.join(stat)
            + ' | total GPU-h | total CPU core-h |\n|' + '---|' * (len(stat) + 6) + '\n')
    groups = {}
    for r in rows:
        groups.setdefault((r['vehicle'], r['trainer'], r['config']), []).append(r)
    for (v, t, c), rs in sorted(groups.items()):
        def ms(k):
            x = [r[k] for r in rs if r[k] is not None]
            return '' if not x else f'{np.mean(x):.4g}' if len(x) == 1 else f'{np.mean(x):.4g} ± {np.std(x):.2g}'
        tot = lambda k: f"{sum(r[k] or 0 for r in rs):.1f}"
        n_to = sum(str(r['state']).split('+')[-1] not in ('COMPLETED', '0') for r in rs)
        f.write(f"| {v} | {t} | {c} | {len(rs)}{f' ({n_to} not completed)' if n_to else ''} | "
                + ' | '.join(ms(k) for k in stat) + f" | {tot('gpu_h')} | {tot('cpu_core_h')} |\n")
    f.write(f"\nAll runs: {sum(r['gpu_h'] or 0 for r in rows):.1f} GPU-h, "
            f"{sum(r['cpu_core_h'] or 0 for r in rows):.0f} CPU core-h, {sum(r['steps'] or 0 for r in rows) / 1e6:.1f}M steps.\n")
    f.write('\n## Per run\n\n| ' + ' | '.join(cols) + ' |\n|' + '---|' * len(cols) + '\n')
    for r in rows:
        f.write('| ' + ' | '.join(str(r[c]) for c in cols) + ' |\n')
print(open(args.out + '.md').read())
