"""Compare several controllers on the IROS 2025 Fig. 4 protocol (outputs of eval_fig4.py).

usage: python3 plot_fig4_compare.py --out DIR  label1=path/to/fig4_steps.json  label2=...  [...]
Writes:
  compare_timeseries.png   desired vs each controller: depth, yaw, surge, pitch, roll (paper Figs. 5-7 style)
  compare_blocks.png       per-block RMSE depth / yaw / surge
  compare_summary.csv      overall + per-block metrics per controller
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# fixed categorical order (validated palette, light mode); desired value in neutral ink
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
BLOCK_NAMES = ['Dive', 'Fwd', 'Fwd', 'Fwd', 'Bwd', 'Bwd', 'Bwd', 'Hover', 'Hover']


def continuous_yaw(t, yaw, sp_yaw, block):
    """Unwrapped actual yaw, and the setpoint shifted by k*2pi per block to the equivalent angle the
    vehicle settles at, so 180 deg turns show as a continuous ramp in the direction actually taken."""
    y = np.unwrap(yaw)
    sp = np.array(sp_yaw, dtype=float).copy()
    for b in np.unique(block):
        m = block == b
        end = y[m][-1]
        sp[m] += 2 * np.pi * np.round((end - sp[m][-1]) / (2 * np.pi))
    return y, sp

p = argparse.ArgumentParser()
p.add_argument('--out', required=True)
p.add_argument('runs', nargs='+', help='label=path/to/fig4_steps.json')
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

runs = []
for spec in args.runs:
    label, path = spec.split('=', 1)
    d = json.load(open(path))
    runs.append((label, d['rows'], d.get('failed')))
    names = d.get('block_names', BLOCK_NAMES)   # protocol of the (last) run; all runs must share it
nb = len(names)


def col(rows, k):
    return np.array([float(r[k]) if r.get(k, '') != '' else np.nan for r in rows])


def rms(x):
    x = np.asarray(x, float)
    return float(np.sqrt(np.nanmean(x ** 2)))


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#c9c8c2')


# ---- time series
panels = [('Depth [m]', 'sp_depth', 'depth', 1.0), ('Yaw [deg]', 'sp_yaw', 'yaw', 180 / np.pi),
          ('Surge [m/s]', 'sp_surge', 'surge', 1.0), ('Pitch [deg]', 'sp_pitch', 'pitch', 180 / np.pi),
          ('Roll [deg]', 'sp_roll', 'roll', 180 / np.pi)]
fig, axs = plt.subplots(len(panels), 1, figsize=(12, 11), sharex=True, facecolor=SURF)
ref_rows = runs[0][1]
t_ref = col(ref_rows, 't')
block_len = (t_ref[-1] + (t_ref[1] - t_ref[0])) / nb if len(t_ref) > 1 else 100
for ax, (lab, des, act, k) in zip(axs, panels):
    style(ax)
    if des:
        ax.plot(t_ref, col(ref_rows, des) * k, color=INK, lw=1.6, ls='--', label='desired', zorder=5)
    for i, (label, rows, failed) in enumerate(runs):
        y = col(rows, act)
        if act == 'yaw':   # wrapped to [-180, 180): break the line at the wrap instead of drawing a vertical jump
            y = (y + np.pi) % (2 * np.pi) - np.pi
            y[1:][np.abs(np.diff(y)) > np.pi] = np.nan
        ax.plot(col(rows, 't'), y * k, color=SERIES[i % len(SERIES)], lw=1.4, label=label)
    if act == 'yaw':
        ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_ylabel(lab, color=MUTED, fontsize=9)
    for b in range(1, nb):
        ax.axvline(b * block_len, color='#c9c8c2', lw=0.6, zorder=0)
for b, name in enumerate(names):
    axs[0].text((b + 0.5) * block_len, 1.02, name, transform=axs[0].get_xaxis_transform(),
                ha='center', va='bottom', fontsize=8, color=MUTED)
axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.32), ncol=min(len(runs) + 1, 6), fontsize=8, frameon=False)
axs[-1].set_xlabel('time [s]', color=MUTED)
fig.tight_layout()
fig.savefig(os.path.join(args.out, 'compare_timeseries.png'), dpi=140)

# ---- per-block bars + summary csv
metrics = [('RMSE depth [m]', 'err_depth'), ('RMSE yaw [rad]', 'err_yaw'), ('RMSE surge [m/s]', 'err_surge_cmd'),
           ('RMSE roll [rad]', 'err_roll'), ('RMSE pitch [rad]', 'err_pitch')]
summary = []
for label, rows, failed in runs:
    blocks = col(rows, 'block')
    row = {'controller': label, 'failed': failed or ''}
    for _, key in metrics:
        k = key if key in rows[0] else ('err_surge' if key == 'err_surge_cmd' else key)
        v = col(rows, k)
        row[f'{key}_all'] = rms(v)
        for b in range(nb):
            row[f'{key}_b{b}'] = rms(v[blocks == b]) if np.any(blocks == b) else np.nan
    thr = np.stack([col(rows, f'thr{i}') for i in range(sum(k.startswith('thr') for k in rows[0]))], 1)
    row['energy_sum_thr2_dt'] = float(np.nansum(thr ** 2) * 0.1)
    row['max_abs_roll_deg'] = float(np.degrees(np.nanmax(np.abs(col(rows, 'roll')))))
    row['max_abs_pitch_deg'] = float(np.degrees(np.nanmax(np.abs(col(rows, 'pitch')))))
    summary.append(row)
with open(os.path.join(args.out, 'compare_summary.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader()
    w.writerows(summary)

fig, axs = plt.subplots(len(metrics), 1, figsize=(12, 11), facecolor=SURF)
x = np.arange(nb)
wbar = 0.8 / len(runs)
for ax, (lab, key) in zip(axs, metrics):
    style(ax)
    for i, row in enumerate(summary):
        vals = [row[f'{key}_b{b}'] for b in range(nb)]
        ax.bar(x - 0.4 + wbar * (i + 0.5), vals, width=wbar * 0.92, color=SERIES[i % len(SERIES)],
               label=row['controller'])
    ax.set_xticks(x)
    ax.set_xticklabels([f'{b} {n}' for b, n in enumerate(names)], fontsize=8, color=MUTED)
    ax.set_ylabel(lab, color=MUTED, fontsize=9)
axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.28), ncol=min(len(runs), 6), fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(args.out, 'compare_blocks.png'), dpi=140)

print(f"{'controller':22s} {'depth':>7s} {'yaw':>7s} {'surge':>7s} {'energy':>8s} {'roll':>6s} {'pitch':>6s}")
for r in summary:
    print(f"{r['controller']:22s} {r['err_depth_all']:7.3f} {r['err_yaw_all']:7.3f} {r['err_surge_cmd_all']:7.3f} "
          f"{r['energy_sum_thr2_dt']:8.1f} {r['max_abs_roll_deg']:6.1f} {r['max_abs_pitch_deg']:6.1f}"
          + (f"  FAILED: {r['failed']}" if r['failed'] else ''))
print('wrote', args.out)
