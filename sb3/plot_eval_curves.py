"""Evaluation-return learning curves and final-policy table from eval_checkpoints.py outputs.

Per run: the mean over the evaluation episodes at each checkpoint; per algorithm: mean over seeds with a 95% bootstrap
confidence interval (resampling seeds). Final table: interquartile mean (IQM) over all final-model episodes of all seeds,
with a 95% stratified bootstrap CI (episodes resampled within each seed), as recommended by Agarwal et al. (NeurIPS 2021).

usage: python3 plot_eval_curves.py --out results/race2/2026-10-28_v4/eval_curves --filter ref_all-1 integral-1
Writes <out>.png, <out>.pdf (paper width), <out>_final.md/.csv. Runs are selected as in plot_training_curves.py.
"""
import argparse
import csv
import glob
import os
import re
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# same fixed categorical order as plot_fig4_compare.py / plot_training_curves.py: ppo, td3, dreamer
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
ALGOS = ['ppo', 'td3', 'dreamer']
LABELS = {'ppo': 'PPO', 'td3': 'TD3', 'dreamer': 'DreamerV3'}
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
PANELS = [('reward', 'evaluation return\n(max ≈ 600)'), ('rmse_depth', 'depth RMSE [m]'), ('rmse_yaw', 'yaw RMSE [rad]'),
          ('energy', 'energy $\\sum_k u_k^2\\,\\Delta t$')]
FINAL = ['reward', 'rmse_depth', 'rmse_yaw', 'rmse_surge', 'rmse_pitch', 'rmse_roll', 'energy', 'terminated']
rng = np.random.default_rng(0)

p = argparse.ArgumentParser()
p.add_argument('--out', required=True, help='output path without extension')
p.add_argument('--filter', nargs='*', default=[])
p.add_argument('--every', type=float, default=1e5, help='checkpoint spacing used by eval_checkpoints.py')
p.add_argument('--logs', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'))
p.add_argument('--boot', type=int, default=2000)
args = p.parse_args()


def load(path):
    """{x: {metric: per-episode array}}; x = checkpoint steps rounded to the eval spacing (final model wins ties)."""
    by = {}
    for r in csv.DictReader(open(path)):
        by.setdefault((int(r['steps']), r['checkpoint']), []).append(r)
    out = {}
    for (steps, ck), rows in sorted(by.items()):
        x = round(steps / args.every) * args.every
        if x in out and not ck.startswith('final'):
            continue
        out[x] = {m: np.array([float(r[m]) for r in rows]) for m in FINAL}
    return out


def boot_ci(per_seed):
    """95% CI of the mean over seeds (bootstrap over seeds)."""
    per_seed = np.asarray(per_seed)
    if len(per_seed) < 2:
        return per_seed.mean(), per_seed.mean()
    b = rng.choice(per_seed, (args.boot, len(per_seed))).mean(axis=1)
    return np.percentile(b, 2.5), np.percentile(b, 97.5)


def iqm(x):
    x = np.sort(np.asarray(x).ravel())
    n = len(x)
    return x[int(np.floor(0.25 * n)):int(np.ceil(0.75 * n))].mean()


def iqm_ci(per_seed_eps):
    """IQM over all episodes; 95% stratified bootstrap CI (episodes resampled within each seed)."""
    b = [iqm(np.concatenate([rng.choice(e, len(e)) for e in per_seed_eps])) for _ in range(args.boot)]
    return iqm(np.concatenate(per_seed_eps)), np.percentile(b, 2.5), np.percentile(b, 97.5)


groups = {}
for run in sorted(glob.glob(os.path.join(args.logs, '*'))):
    name = os.path.basename(run)
    algo = name.split('_')[0]
    f = os.path.join(run, 'eval_checkpoints.csv')
    if algo not in ALGOS or not all(s in name for s in args.filter) or not os.path.exists(f):
        continue
    if 'vehicle-' in name and not any('vehicle-' in s for s in args.filter):
        continue
    vehicle = (re.search(r'vehicle-([a-z0-9_]+?)_stage', name) or [None, 'race2'])[1]
    config = 'articulated' if 'servos-2' in name else 'fixed thrusters'
    groups.setdefault((vehicle, config), {}).setdefault(algo, []).append(load(f))
if not groups:
    raise SystemExit('no eval_checkpoints.csv matched')

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'legend.fontsize': 7})
configs = sorted(groups, key=lambda vc: (vc[0], vc[1] != 'fixed thrusters'))
fig, axs = plt.subplots(len(PANELS), len(configs), figsize=(3.5 * len(configs), 1.6 * len(PANELS) + 0.4),
                        squeeze=False, sharex=True, facecolor=SURF)
final_rows = []
for c, vc in enumerate(configs):
    means = {}
    for algo in ALGOS:
        runs = groups[vc].get(algo)
        if not runs:
            continue
        color = SERIES[ALGOS.index(algo)]
        xs = sorted(set.intersection(*[set(r) for r in runs]))   # checkpoints every seed has
        for m, (key, lab) in enumerate(PANELS):
            per = np.array([[r[x][key].mean() for r in runs] for x in xs])   # [checkpoint, seed]
            lo, hi = np.array([boot_ci(row) for row in per]).T
            ax = axs[m, c]
            ax.fill_between(np.array(xs) / 1e6, lo, hi, color=color, alpha=0.18, lw=0)
            ax.plot(np.array(xs) / 1e6, per.mean(axis=1), color=color, lw=1.6, label=f'{LABELS[algo]} (n={len(runs)})')
            means.setdefault(m, []).extend(per.mean(axis=1))
        last = xs[-1]
        row = {'vehicle': vc[0], 'config': vc[1], 'algo': LABELS[algo], 'seeds': len(runs),
               'episodes': sum(len(r[last]['reward']) for r in runs), 'steps': int(last)}
        for key in FINAL:
            v, l, h = iqm_ci([r[last][key] for r in runs]) if key != 'terminated' else \
                (np.mean(np.concatenate([r[last][key] for r in runs])),) * 3
            row[key] = (round(float(v), 4), round(float(l), 4), round(float(h), 4))
        final_rows.append(row)
    for m, (_, lab) in enumerate(PANELS):
        ax = axs[m, c]
        # y range from the mean curves without their extreme 2% (an early checkpoint or a failed seed's CI band
        # runs off the axes instead of flattening everything else)
        lo, hi = np.percentile(means[m], [2, 98])
        ax.set_ylim(lo - 0.15 * (hi - lo), hi + 0.15 * (hi - lo))
        ax.set_facecolor(SURF)
        ax.grid(color=GRID, lw=0.6)
        ax.tick_params(colors=MUTED, labelsize=7)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color('#c9c8c2')
        ax.set_ylabel(lab, color=MUTED)
    axs[0, c].set_title(f'{vc[0]}: {vc[1]}', color=INK, loc='left')
    axs[-1, c].set_xlabel('environment steps [M]', color=MUTED)
    axs[0, c].legend(frameon=False, loc='lower right', labelcolor=INK)
note = textwrap.fill('return = sum over a 60 s episode (600 steps) of the per-step reward ≤ 1; max ≈ 600. Deterministic '
                     'policy, same seeded setpoint episodes at every checkpoint; mean over seeds, 95% bootstrap CI (bands and outlying '
                     'early checkpoints may extend past the axes). RMSE is to the raw step targets, so it includes the transients',
                     int(fig.get_figwidth() * 21))
fig.text(0.01, 0.003, note, fontsize=6.5, color=MUTED, va='bottom')
fig.tight_layout(rect=(0, 0.11 * (note.count('\n') + 1) / fig.get_figheight(), 1, 1))
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
for ext in ('png', 'pdf'):
    fig.savefig(f'{args.out}.{ext}', dpi=200, facecolor=SURF)

with open(f'{args.out}_final.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['vehicle', 'config', 'algo', 'seeds', 'episodes', 'steps'] + [f'{k}_{s}' for k in FINAL for s in ('iqm', 'lo', 'hi')])
    for r in final_rows:
        w.writerow([r[k] for k in ('vehicle', 'config', 'algo', 'seeds', 'episodes', 'steps')] + [v for k in FINAL for v in r[k]])
with open(f'{args.out}_final.md', 'w') as f:
    f.write('Final model, deterministic: IQM over all evaluation episodes [95% stratified bootstrap CI]; '
            'terminated = fraction of episodes ended early.\n\n')
    f.write('| vehicle | config | algorithm | seeds x episodes | ' + ' | '.join(FINAL) + ' |\n|' + '---|' * (len(FINAL) + 4) + '\n')
    for r in final_rows:
        cells = [f'{r[k][0]:.3g} [{r[k][1]:.3g}, {r[k][2]:.3g}]' if k != 'terminated' else f'{r[k][0]:.2f}' for k in FINAL]
        f.write(f"| {r['vehicle']} | {r['config']} | {r['algo']} | {r['seeds']} x {r['episodes'] // r['seeds']} | "
                + ' | '.join(cells) + ' |\n')
print(open(f'{args.out}_final.md').read())
