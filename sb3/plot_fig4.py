"""Plot one Fig-4 evaluation (paper Figs. 5-7 layout) from eval_fig4.py's fig4_steps.json.

usage: python3 plot_fig4.py <fig4_steps.json> [--label NAME] [--out fig4.png]
Left: desired (dashed) vs actual for roll, pitch, yaw, depth, surge. Right: actuator commands.
Grey lines mark setpoint changes (block boundaries); transients right after them are expected coupling
(e.g. surge dips during 180 deg turns because the surge thrusters also produce the yaw torque).
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, GRID, SURF, RULE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb', '#c9c8c2'
BLUE, ORANGE, GREEN = '#2a78d6', '#eb6834', '#1baf7a'
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


def plot(path, label=None, out=None):
    d = json.load(open(path))
    rows = d['rows']
    label = label or os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(path))))
    out = out or os.path.join(os.path.dirname(path), 'fig4.png')
    col = lambda k: np.array([float(r[k]) if r.get(k, '') != '' else np.nan for r in rows])
    t = col('t')
    names = d.get('block_names', BLOCK_NAMES)
    block_len = d.get('block_seconds') or (t[-1] + (t[1] - t[0])) / len(names)
    n_serv = int(d.get('num_servos', 0))

    fig = plt.figure(figsize=(13, 10.5), facecolor=SURF)
    gs = fig.add_gridspec(5, 2, width_ratios=[1.35, 1], hspace=0.35, wspace=0.18)
    left = [fig.add_subplot(gs[i, 0]) for i in range(5)]
    right = [fig.add_subplot(gs[0:2, 1]), fig.add_subplot(gs[2:4, 1]), fig.add_subplot(gs[4:5, 1])]
    for a in left[1:]:
        a.sharex(left[0])
    for a in right[1:]:
        a.sharex(right[0])

    def style(a):
        a.set_facecolor(SURF)
        a.grid(color=GRID, lw=0.8)
        a.set_axisbelow(True)
        a.tick_params(colors=MUTED, labelsize=8)
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            a.spines[s].set_color(RULE)
        for b in range(1, len(names)):
            a.axvline(b * block_len, color=RULE, lw=0.7, zorder=0)

    sp_att = lambda k: np.degrees(col(k)) if k in rows[0] else None   # commanded roll/pitch (artic protocol), else 0
    states = [('Roll [deg]', sp_att('sp_roll'), np.degrees(col('roll'))), ('Pitch [deg]', sp_att('sp_pitch'), np.degrees(col('pitch'))),
              ('Yaw [deg]', *[np.degrees(v) for v in continuous_yaw(t, col('yaw'), col('sp_yaw'), col('block'))[::-1]]),
              ('Depth [m]', col('sp_depth'), col('depth')), ('Surge [m/s]', col('sp_surge'), col('surge'))]
    for a, (lab, des, act) in zip(left, states):
        style(a)
        if des is None:
            des = np.zeros_like(act)   # roll/pitch are regulated to 0
        a.plot(t, des, color=INK, lw=1.4, ls='--', zorder=4)
        a.plot(t, act, color=BLUE, lw=1.3, zorder=3)
        a.set_ylabel(lab, color=MUTED, fontsize=9)
    for b, name in enumerate(names):
        left[0].text((b + 0.5) * block_len, 1.04, name, transform=left[0].get_xaxis_transform(),
                     ha='center', va='bottom', fontsize=8, color=MUTED)
    left[-1].set_xlabel('time [s]', color=MUTED)
    for a in left[:-1]:
        plt.setp(a.get_xticklabels(), visible=False)

    names_thr = d.get('thruster_names', ['heave_bow', 'heave_stern', 'surge_port', 'surge_starboard'])   # race2 runs predate the field
    groups = [[(f'thr{i}', n) for i, n in enumerate(names_thr) if n.startswith('heave')],
              [(f'thr{i}', n) for i, n in enumerate(names_thr) if not n.startswith('heave')]]
    short = lambda n: n.split('_', 1)[1] if '_' in n and n.startswith(('heave', 'surge')) else n
    pairs = [('Heave thrusters [-]', [(k, short(n)) for k, n in groups[0]]),
             ('Surge / sway thrusters [-]' if len(groups[1]) > 2 else 'Surge thrusters [-]', [(k, short(n)) for k, n in groups[1]])]
    for a, (lab, keys) in zip(right[:2], pairs):
        style(a)
        for (k, name), c in zip(keys, (BLUE, ORANGE, GREEN)):
            a.plot(t, col(k), color=c, lw=1.1, label=name)
        a.set_ylim(-1.05, 1.05)
        a.set_ylabel(lab, color=MUTED, fontsize=9)
        a.legend(loc='upper right', fontsize=8, frameon=False, ncol=len(keys))
    a = right[2]
    style(a)
    if n_serv:
        for i, (name, c) in enumerate(zip(('port', 'starboard'), (BLUE, ORANGE))):
            a.plot(t, np.degrees(col(f'servo{i}')), color=c, lw=1.1, label=name)
        a.legend(loc='upper right', fontsize=8, frameon=False, ncol=2)
    else:
        a.set_yticks([])
        a.text(0.5, 0.5, 'fixed thrusters (no servos)', transform=a.transAxes,
               ha='center', va='center', fontsize=9, color=MUTED)
    a.set_ylabel('Servo angle [deg]', color=MUTED, fontsize=9)
    a.set_xlabel('time [s]', color=MUTED)
    for a in right[:2]:
        plt.setp(a.get_xticklabels(), visible=False)

    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=INK, ls='--', lw=1.4), Line2D([], [], color=BLUE, lw=1.3)],
               ['desired (setpoint)', 'actual'], loc='upper left', bbox_to_anchor=(0.06, 0.94),
               ncol=2, fontsize=9, frameon=False)
    sub = ((f"IROS 2025 Fig. 4 protocol, surge ±{d.get('surge', '?')} m/s | " if d.get('protocol', 'fig4') == 'fig4'
            else "articulation protocol (roll/pitch/depth/surge combos) | ")
           + (f"articulated ({d.get('action_mode', '?')} action)" if n_serv else 'fixed thrusters')
           + (f" | surge reward: {d['surge_mode']}" if d.get('surge_mode') else ''))
    import textwrap   # long run-dir labels: wrap to the figure width (else bbox 'tight' widens the canvas)
    fig.suptitle(textwrap.fill(label, 150) + f"\n{sub}" + (f" | {d['failed']}" if d.get('failed') else ''),
                 color=INK, fontsize=10, x=0.06, y=0.995, ha='left')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('json')
    p.add_argument('--label', default=None)
    p.add_argument('--out', default=None)
    a = p.parse_args()
    print('wrote', plot(a.json, a.label, a.out))
