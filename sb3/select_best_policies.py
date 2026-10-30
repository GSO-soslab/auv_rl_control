"""Keep one policy per vehicle x thruster config x algorithm: the seed whose FINAL model has the highest evaluation return
(IQM over its eval_checkpoints.py episodes). Copies it with its settings to <out>/policies/ and writes the selection table.

usage: python3 select_best_policies.py --out results/race2/2026-10-28_v4 --filter ref_all-1 integral-1
       python3 select_best_policies.py --out results/race_auv/<date> --filter vehicle-race_auv
Paper wording: statistics are over all seeds; the deployed policy is the best seed by evaluation return.
Copies only; deleting the other seeds is a separate, explicit step.
"""
import argparse
import csv
import glob
import json
import os
import re
import shutil

import numpy as np

ALGOS = ['ppo', 'td3', 'dreamer']
p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
p.add_argument('--filter', nargs='*', default=[])
p.add_argument('--logs', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'))
args = p.parse_args()


def iqm(x):
    x = np.sort(np.asarray(x, float))
    return x[int(np.floor(0.25 * len(x))):int(np.ceil(0.75 * len(x)))].mean()


cands = {}
for run in sorted(glob.glob(os.path.join(args.logs, '*'))):
    name = os.path.basename(run)
    algo = name.split('_')[0]
    f = os.path.join(run, 'eval_checkpoints.csv')
    if algo not in ALGOS or not all(s in name for s in args.filter) or not os.path.exists(f):
        continue
    if 'vehicle-' in name and not any('vehicle-' in s for s in args.filter):
        continue
    rows = [r for r in csv.DictReader(open(f)) if r['checkpoint'].startswith('final_model')]
    if not rows:
        print(f'skip {name}: final model not evaluated')
        continue
    vehicle = (re.search(r'vehicle-([a-z0-9_]+?)_stage', name) or [None, 'race2'])[1]
    config = 'artic' if 'servos-2' in name else 'fixed'
    seed = (re.search(r'seed-(\d+)', name) or [None, '?'])[1]
    m = {k: np.array([float(r[k]) for r in rows]) for k in ('reward', 'energy', 'terminated', 'rmse_depth', 'rmse_yaw', 'rmse_surge')}
    cands.setdefault((vehicle, config, algo), []).append({'run': run, 'seed': seed, 'iqm': iqm(m['reward']), 'm': m})

os.makedirs(os.path.join(args.out, 'policies'), exist_ok=True)
lines = ['| vehicle | config | algorithm | kept seed | eval return IQM per seed | energy | terminated | run |', '|' + '---|' * 8]
for (vehicle, config, algo), cs in sorted(cands.items()):
    best = max(cs, key=lambda c: c['iqm'])
    model = next(f for f in (os.path.join(best['run'], 'final_model.zip'), os.path.join(best['run'], 'final_model.pt'))
                 if os.path.exists(f))
    dst = os.path.join(args.out, 'policies', f'{algo}_{config}{os.path.splitext(model)[1]}')
    shutil.copy2(model, dst)
    info = os.path.join(best['run'], 'run_info', 'job.json')
    meta = {'source_run': os.path.basename(best['run']), 'seed': best['seed'], 'algo': algo, 'vehicle': vehicle, 'config': config,
            'selection': 'highest IQM evaluation return of the final model over the eval_checkpoints.py episodes',
            'eval_return_iqm': {c['seed']: round(float(c['iqm']), 2) for c in cs},
            'settings': json.load(open(info))['settings'] if os.path.exists(info) else None}
    json.dump(meta, open(os.path.splitext(dst)[0] + '.json', 'w'), indent=1)
    per_seed = ', '.join(f"s{c['seed']} {c['iqm']:.0f}" for c in sorted(cs, key=lambda c: c['seed']))
    lines.append(f"| {vehicle} | {config} | {algo} | s{best['seed']} | {per_seed} | {np.mean(best['m']['energy']):.1f} | "
                 f"{np.mean(best['m']['terminated']):.2f} | {os.path.basename(best['run'])[:40]}... |")
    print(f'{vehicle}/{config}/{algo}: keep seed {best["seed"]} ({per_seed}) -> {dst}')
open(os.path.join(args.out, 'policies', 'SELECTION.md'), 'w').write(
    'One policy per vehicle x config x algorithm: the seed with the highest evaluation return (IQM) of its final model.\n'
    'Statistics in the paper are over all seeds. Each policy has a .json with its run, settings and the per-seed scores.\n'
    'Pilot/evaluate: the settings in the .json are the env vars the policy needs (infra/pilot.sh takes the source run).\n\n'
    + '\n'.join(lines) + '\n')
