"""Evaluate a policy on the setpoint sequence of Zhou, Naderi, Yuan (IROS 2025), Fig. 4.

Nine blocks (default 100 s each), vehicle spawned at 0.5 m heading North (default sim pose):
  Dive | Move forward x3 | Move backward x3 | Hover x2
  depth  5    5    5    2.5  2.5  2.5  5    5    5   m
  yaw    W    W    E    E    E    W    W    W    E   (W = -pi/2, E = +pi/2, NED)
  surge  0    +v   +v   +v   -v   -v   -v   0    0   (v = 0.15 m/s in the paper's base model)

--protocol artic: combo maneuvers that need vertical thrust at the stern, i.e. articulation (9 x 60 s, see ARTIC).
Fixed thrusters are expected to fail the roll blocks (the heave thrusters are on the centerline).

usage (with `race2 sim --rl` running; the RACE2_* env vars must match the model, see infra/run_fig4_runs.sh):
  python3 eval_fig4.py <model.zip> --algo ppo [--protocol fig4|artic] [--surge 0.15] [--block-seconds 100] [--out DIR]
Writes <out>/fig4_steps.json/.csv, fig4_metrics.csv and fig4.png.
"""
import argparse
import csv
import os

import numpy as np
from types import SimpleNamespace

from stable_baselines3 import DDPG, PPO, SAC, TD3

from AUVEnv import AUVEnv

ALGOS = {'ddpg': DDPG, 'ppo': PPO, 'sac': SAC, 'td3': TD3,   # dreamer: imported on use (needs ~/dreamerv3-torch)
         'dreamer': SimpleNamespace(load=lambda p, device='cpu': __import__('dreamer_agent').DreamerPolicy.load(p, device))}
from fig4_protocols import PROTOCOLS   # shared with eval_mvp.py

p = argparse.ArgumentParser()
p.add_argument('model')
p.add_argument('--algo', default='ppo', choices=ALGOS)
p.add_argument('--protocol', default='fig4', choices=PROTOCOLS)
p.add_argument('--surge', type=float, default=0.15, help='fig4 only: surge magnitude of the move blocks')
p.add_argument('--block-seconds', type=float, default=None, help='default 100 (fig4) / 60 (artic)')
p.add_argument('--out', default=None)
args = p.parse_args()
BLOCKS, default_len = PROTOCOLS[args.protocol]
block_seconds = args.block_seconds or default_len
out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.model)), f'{args.protocol}_eval')
os.makedirs(out, exist_ok=True)

import torch
torch.set_num_threads(1)   # evals share a node with real-time sims: torch's default (one thread per core in every
                           # process) starved them on 2026-10-28 (frozen sim state in all Dreamer evals)
model = ALGOS[args.algo].load(args.model, device='cpu')
env = AUVEnv()
env.tracking_cfg['setpoint_change_seconds'] = 0   # the protocol sets all setpoints (no random step commands)
n, dt = env.node, env.tracking_cfg.get('control_dt', 0.1)
obs, _ = env.reset(options={'spawn_depth': 0.5, 'spawn_yaw': 0.0})   # default sim spawn (same as the mvp baseline)
env.max_steps = 10 ** 9  # one continuous run; no truncation
sp = n.setpoint_manager.current_setpoint
steps_per_block = int(round(block_seconds / dt))

rows, failed = [], None
for b, (mode, depth, yaw, surge, roll_sp, pitch_sp) in enumerate(BLOCKS):
    roll_sp, pitch_sp = float(roll_sp), float(pitch_sp)   # ROS float fields reject ints
    sp.position.z, sp.orientation.x, sp.orientation.y, sp.orientation.z = float(depth), roll_sp, pitch_sp, float(yaw)
    sp.velocity.x = float(surge * args.surge if args.protocol == 'fig4' else surge)
    for k in range(steps_per_block):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _, info = env.step(action)
        r, pch, y = n.orientation_state
        rows.append({
            't': (b * steps_per_block + k) * dt, 'block': b, 'mode': mode,
            'sp_depth': depth, 'sp_yaw': yaw, 'sp_surge': sp.velocity.x, 'sp_roll': roll_sp, 'sp_pitch': pitch_sp,
            'depth': n.position_state[2], 'roll': r, 'pitch': pch, 'yaw': y,
            'surge': n.v_state[0], 'sway': n.v_state[1],
            **{f'thr{i}': float(v) for i, v in enumerate(env.thruster_action)},
            **{f'servo{i}': float(v) for i, v in enumerate(n.servo_measured[:env.num_servos])},
            'reward': reward,
            **{f'err_{d}': v for d, v in info['abs_err'].items()},
        })
        if terminated:
            failed = f"terminated in block {b} ({mode}) at t={rows[-1]['t']:.1f}s"
            break
    rb = [x for x in rows if x['block'] == b]
    print(f"block {b} {mode:13s} depth {depth:3.1f} yaw {np.degrees(yaw):+4.0f} surge {sp.velocity.x:+.2f} "
          f"roll {np.degrees(roll_sp):+3.0f} pitch {np.degrees(pitch_sp):+3.0f}: depth err {np.mean([x['err_depth'] for x in rb]):.3f} m, "
          f"roll err {np.degrees(np.mean([x['err_roll'] for x in rb])):.1f} deg, pitch err {np.degrees(np.mean([x['err_pitch'] for x in rb])):.1f} deg",
          flush=True)
    if failed:
        print(failed)
        break

import json
with open(os.path.join(out, 'fig4_steps.json'), 'w') as f:   # raw data first, so post-processing bugs can't lose a run
    json.dump({'rows': rows, 'failed': failed, 'model': args.model, 'surge': args.surge, 'protocol': args.protocol,
               'block_names': [blk[0] for blk in BLOCKS], 'block_seconds': block_seconds,
               'action_mode': env.action_mode, 'num_servos': env.num_servos, 'surge_mode': env.surge_mode,
               'vehicle': env.vehicle_name, 'thruster_names': list(n.thruster_names)}, f, default=float)
fields = list(dict.fromkeys(k for r in rows for k in r))   # union (some keys, e.g. err_surge_steady, start later)
with open(os.path.join(out, 'fig4_steps.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields, restval='')
    w.writeheader()
    w.writerows(rows)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def metrics(rs):
    thr = np.array([[x[f'thr{i}'] for i in range(env.num_thrusters)] for x in rs])
    m = {'rmse_depth_m': rms([x['err_depth'] for x in rs]), 'rmse_yaw_rad': rms([x['err_yaw'] for x in rs]),
         'rmse_surge_ms': rms([x['err_surge'] for x in rs]),
         'rmse_roll_deg': float(np.degrees(rms([x['err_roll'] for x in rs]))),
         'rmse_pitch_deg': float(np.degrees(rms([x['err_pitch'] for x in rs]))),
         'max_abs_roll_deg': float(np.degrees(np.max(np.abs([x['roll'] for x in rs])))),
         'max_abs_pitch_deg': float(np.degrees(np.max(np.abs([x['pitch'] for x in rs])))),
         'energy_sum_thr2_dt': float(np.sum(thr ** 2) * dt)}
    if env.num_servos:
        sv = np.array([[x[f'servo{i}'] for i in range(env.num_servos)] for x in rs])
        m['servo_travel_rad'] = float(np.sum(np.abs(np.diff(sv, axis=0))))
    return m


with open(os.path.join(out, 'fig4_metrics.csv'), 'w', newline='') as f:
    groups = [('all', rows)] + [(f'{b}_{BLOCKS[b][0]}', [x for x in rows if x['block'] == b])
                                 for b in sorted({x['block'] for x in rows})]
    keys = list(metrics(rows).keys())
    w = csv.writer(f)
    w.writerow(['segment'] + keys)
    for name, rs in groups:
        mm = metrics(rs)
        w.writerow([name] + [f'{mm[k]:.4f}' for k in keys])
print("overall:", {k: round(v, 3) for k, v in metrics(rows).items()}, "| failed:", failed)

# Figure in the layout of the paper's Figs. 5-7 (re-render any time with plot_fig4.py <fig4_steps.json>)
from plot_fig4 import plot
print('wrote', plot(os.path.join(out, 'fig4_steps.json'), label=os.path.basename(os.path.dirname(os.path.abspath(args.model)))))
