"""Seed-averaged Fig-4 table: mean +- std per variant over all runs matching the given groups.

usage: python3 summarize_fig4.py --out DIR  LABEL=GLOB [LABEL=GLOB ...]  [--tags fig4_v015,fig4_v05]
  e.g. v2_rate='logs/*action-rate*depth_inv-1*seed-*'
Steady state = the last `--ss-seconds` (default 50 s) of each block (block length from the json), i.e. after the transient.
Writes DIR/summary_<tag>.csv (per run) and DIR/summary_<tag>.md (mean +- std per group).
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
p.add_argument('--tags', default='fig4_v015,fig4_v05')
p.add_argument('--ss-seconds', type=float, default=50.0)
p.add_argument('groups', nargs='+')
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

METRICS = [  # (key, header, format)
    ('depth', 'depth RMSE [m]', '.3f'), ('depth_ss', 'depth ss [m]', '.3f'),
    ('yaw', 'yaw RMSE [deg]', '.1f'), ('yaw_ss', 'yaw ss [deg]', '.2f'),
    ('roll', 'roll RMSE [deg]', '.1f'), ('pitch', 'pitch RMSE [deg]', '.1f'),
    ('surge', 'surge RMSE [m/s]', '.3f'), ('surge_ss', 'surge ss [m/s]', '.4f'),
    ('roll_max', 'max roll [deg]', '.1f'), ('pitch_max', 'max pitch [deg]', '.1f'),
    ('energy', 'energy sum thr^2 dt', '.0f'), ('servo_travel', 'servo travel [rad]', '.1f'),
    ('servo_max', 'max servo [deg]', '.1f'),
]


def run_metrics(path):
    d = json.load(open(path))
    r = d['rows']
    col = lambda k: np.array([float(x[k]) if x.get(k, '') != '' else np.nan for x in r])
    t, blk = col('t'), col('block')
    t_in = t - np.array([t[blk == b][0] for b in blk])            # time since the block started
    ss = t_in >= (float(d.get('block_seconds') or 100.0) - args.ss_seconds)   # fig4: 100 s blocks, att90: 60 s
    rms = lambda x: float(np.sqrt(np.nanmean(x ** 2)))
    e_d, e_y, e_u = col('err_depth'), np.degrees(col('err_yaw')), col('err_surge_cmd')
    thr = np.stack([col(f'thr{i}') for i in range(len(d.get('thruster_names', range(4))))], 1)   # race2 runs: 4
    n_serv = int(d.get('num_servos', 0))
    servo = np.stack([col(f'servo{i}') for i in range(n_serv)], 1) if n_serv else np.zeros((len(t), 0))
    return {
        'depth': rms(e_d), 'depth_ss': rms(e_d[ss]), 'yaw': rms(e_y), 'yaw_ss': rms(e_y[ss]),
        'surge': rms(e_u), 'surge_ss': rms(e_u[ss]),
        'roll': rms(np.degrees(col('err_roll'))), 'pitch': rms(np.degrees(col('err_pitch'))),   # rot rep: rotation-vector components
        'roll_max': float(np.nanmax(np.abs(np.degrees(col('roll'))))),
        'pitch_max': float(np.nanmax(np.abs(np.degrees(col('pitch'))))),
        'energy': float(np.nansum(thr ** 2) * 0.1),
        'servo_travel': float(np.nansum(np.abs(np.diff(servo, axis=0)))) if n_serv else 0.0,
        'servo_max': float(np.degrees(np.nanmax(np.abs(servo)))) if n_serv else 0.0,
        'failed': d.get('failed') or '',
    }


for tag in args.tags.split(','):
    per_run, md = [], []
    md.append(f"## {tag} (mean ± std over seeds; steady state = last {args.ss_seconds:.0f} s of each block)\n")
    md.append('| variant | n | ' + ' | '.join(h for _, h, _ in METRICS) + ' |')
    md.append('|---|---|' + '---|' * len(METRICS))
    for spec in args.groups:
        label, pattern = spec.split('=', 1)
        runs = sorted(glob.glob(pattern))
        vals = []
        for run in runs:
            f = os.path.join(run, tag, 'fig4_steps.json')
            if not os.path.exists(f):
                continue
            m = run_metrics(f)
            vals.append(m)
            per_run.append({'variant': label, 'run': os.path.basename(run), **m})
        if not vals:
            continue
        cells = []
        for k, _, fmt in METRICS:
            v = np.array([m[k] for m in vals])
            cells.append(f"{v.mean():{fmt}} ± {v.std():{fmt}}")
        fails = sum(bool(m['failed']) for m in vals)
        md.append(f"| {label} | {len(vals)}{f' ({fails} failed)' if fails else ''} | " + ' | '.join(cells) + ' |')
    with open(os.path.join(args.out, f'summary_{tag}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per_run[0].keys()))
        w.writeheader()
        w.writerows(per_run)
    open(os.path.join(args.out, f'summary_{tag}.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md) + '\n')
