"""Training curves (episode reward and episode length vs env steps) from the runs' progress.csv,
mean +- std over seeds per algorithm, one column per thruster config.

usage: python3 plot_training_curves.py --out results/race2/2026-10-28_v4/training_curves.png --filter ref_all-1 integral-1
       python3 plot_training_curves.py --out results/race_auv/<run>/training_curves.png --filter vehicle-race_auv
Runs are the dirs in sb3/logs/ whose name contains ALL --filter substrings (and not 'vehicle-' unless filtered for);
their algorithm is the dir prefix (ppo/td3/dreamer), the config is articulated if the name has servos-2.
Also writes <out>.csv with the per-group curves.
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

# same fixed categorical order as plot_fig4_compare.py (validated palette, light mode): ppo, td3, dreamer
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
ALGOS = ['ppo', 'td3', 'dreamer']
LABELS = {'ppo': 'PPO', 'td3': 'TD3', 'dreamer': 'DreamerV3'}
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
METRICS = [('rollout/ep_rew_mean', 'episode return\n(max ≈ 600)'), ('rollout/ep_len_mean', 'episode length [steps]')]

p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
p.add_argument('--filter', nargs='*', default=[])
p.add_argument('--exclude', nargs='*', default=[], help='skip runs whose dir contains any of these')
p.add_argument('--return-only', action='store_true', help='only the return row (e.g. for the README)')
p.add_argument('--logs', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'))
args = p.parse_args()
if args.return_only:
    METRICS = METRICS[:1]


def load(run):
    rows = list(csv.DictReader(open(os.path.join(run, 'progress.csv'))))
    keys = ['time/total_timesteps'] + [k for k, _ in METRICS]
    rows = [r for r in rows if all(r.get(k) not in (None, '') for k in keys)]
    d = {k: np.array([float(r[k]) for r in rows]) for k in keys}
    ok = np.all([np.isfinite(v) for v in d.values()], axis=0)   # Dreamer logs nan before its first finished episode
    return {k: v[ok] for k, v in d.items()}


def binned(x, y, edges):
    """Mean of y within each step bin (no aliasing of single spikes); empty bins interpolated."""
    i = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    keep = x <= edges[-1]
    cnt = np.bincount(i[keep], minlength=len(edges) - 1)
    tot = np.bincount(i[keep], weights=y[keep], minlength=len(edges) - 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    ok = cnt > 0
    return np.interp(mid, mid[ok], tot[ok] / cnt[ok])


groups = {}
for run in sorted(glob.glob(os.path.join(args.logs, '*'))):
    name = os.path.basename(run)
    algo = name.split('_')[0]
    if algo not in ALGOS or not all(f in name for f in args.filter) or any(x in name for x in args.exclude) or not os.path.exists(os.path.join(run, 'progress.csv')):
        continue
    if 'vehicle-' in name and not any('vehicle-' in f for f in args.filter):
        continue
    vehicle = (re.search(r'vehicle-([a-z0-9_]+?)_stage', name) or [None, 'race2'])[1]
    config = 'articulated' if 'servos-2' in name else 'fixed thrusters'
    d = load(run)
    if len(d['time/total_timesteps']) > 1:
        groups.setdefault((vehicle, config), {}).setdefault(algo, []).append(d)
if not groups:
    raise SystemExit('no runs matched')

configs = sorted(groups, key=lambda vc: (vc[0], vc[1] != 'fixed thrusters'))
fig, axs = plt.subplots(len(METRICS), len(configs), figsize=(6.5 * len(configs), 3.8 * len(METRICS)), squeeze=False, facecolor=SURF,
                        sharex=True)
out_rows = []
for c, vc in enumerate(configs):
    ends, lims = {}, {}
    for algo in ALGOS:
        runs = groups[vc].get(algo)
        if not runs:
            continue
        color = SERIES[ALGOS.index(algo)]
        # runs at < half the longest seed's steps are still starting (restarted job): left out, n says so
        longest = max(r['time/total_timesteps'].max() for r in runs)
        runs = [r for r in runs if r['time/total_timesteps'].max() >= 0.5 * longest]
        # common support: stop at the shortest seed, so the mean never switches to fewer seeds (timed-out runs)
        edges = np.linspace(0, min(r['time/total_timesteps'].max() for r in runs), 201)
        grid = 0.5 * (edges[1:] + edges[:-1])
        for m, (key, lab) in enumerate(METRICS):
            Y = np.array([binned(r['time/total_timesteps'], r[key], edges) for r in runs])
            n = np.full(len(grid), len(runs))
            mean, std = Y.mean(axis=0), Y.std(axis=0)
            lims.setdefault(m, []).extend([mean.min(), mean.max()])
            ax = axs[m, c]
            ax.fill_between(grid / 1e6, mean - std, mean + std, color=color, alpha=0.18, lw=0)
            ax.plot(grid / 1e6, mean, color=color, lw=2, label=f'{LABELS[algo]} (n={len(runs)})')
            ends.setdefault(m, []).append((mean[-1], grid[-1] / 1e6, LABELS[algo]))
            if m == 0:
                out_rows += [{'vehicle': vc[0], 'config': vc[1], 'algo': algo, 'steps': int(s), 'n_seeds': int(k),
                              'ep_rew_mean': round(float(a), 3), 'ep_rew_std': round(float(b), 3)}
                             for s, k, a, b in zip(grid, n, mean, std)]
    for m, (_, lab) in enumerate(METRICS):
        ax = axs[m, c]
        # y range from the means (a diverged seed's band may run off the plot instead of flattening the curves)
        lo, hi = min(lims[m]), max(lims[m])
        lo, hi = lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo)
        ax.set_ylim(lo, hi)
        # direct labels at the line ends, pushed apart vertically so they never overlap
        gap, pos = 0.06 * (hi - lo), []
        for y, x, text in sorted(ends.get(m, [])):
            pos.append([max(y, pos[-1][0] + gap) if pos else y, x, text])
        shift = max(0.0, pos[-1][0] - (hi - gap / 2)) if pos else 0.0   # keep the stack inside the axes
        for y_lab, x, text in pos:
            ax.annotate(text, (x, y_lab - shift), xytext=(x + 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0]), y_lab - shift),
                        va='center', fontsize=8, color=INK)
        ax.set_facecolor(SURF)
        ax.grid(color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color('#c9c8c2')
        ax.set_ylabel(lab, color=MUTED, fontsize=9)
        ax.margins(x=0.08)
    axs[0, c].set_title(f'{vc[0]}: {vc[1]}', color=INK, fontsize=11, loc='left')
    axs[-1, c].set_xlabel('environment steps [M]', color=MUTED)
    axs[0, c].legend(frameon=False, fontsize=8, loc='lower right', labelcolor=INK)
note = textwrap.fill('episode return = sum over a 60 s episode (600 steps) of the per-step reward ≤ 1 (tracking score in '
                     '[0, 1] minus penalties); max ≈ 600. Training rollouts (with exploration). Mean ± std over seeds (band '
                     'may extend past the axes), up to the shortest seed', int(fig.get_figwidth() * 17))
fig.text(0.01, 0.005, note, fontsize=8, color=MUTED, va='bottom')
fig.tight_layout(rect=(0, 0.13 * (note.count('\n') + 1) / fig.get_figheight(), 1, 1))
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
fig.savefig(args.out, dpi=150, facecolor=SURF)
with open(os.path.splitext(args.out)[0] + '.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
    w.writeheader()
    w.writerows(out_rows)
print('wrote', args.out, {f'{v}/{c}': {a: len(r) for a, r in g.items()} for (v, c), g in groups.items()})
