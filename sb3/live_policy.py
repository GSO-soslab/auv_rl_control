"""Run a trained policy continuously in the sim, with setpoints set live from a JSON control file.

usage (with `race2 sim --rl` running; env vars must match the model, e.g. RACE2_SURGE_REWARD=integral_obs):
  python3 live_policy.py <model.zip> --algo ppo --control ~/ros2_ws/runs/live/setpoint.json
Control file (re-read every step; edit it to change targets):
  {"depth": 2.0, "surge": 0.0, "yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}   # optional "respawn": <any new value> respawns
Prints a status line every second.
"""
import argparse
import json
import os
import time

import numpy as np
from types import SimpleNamespace

from stable_baselines3 import DDPG, PPO, SAC, TD3

from AUVEnv import AUVEnv

ALGOS = {'ddpg': DDPG, 'ppo': PPO, 'sac': SAC, 'td3': TD3,   # dreamer: imported on use (needs ~/dreamerv3-torch)
         'dreamer': SimpleNamespace(load=lambda p, device='cpu': __import__('dreamer_agent').DreamerPolicy.load(p, device))}
p = argparse.ArgumentParser()
p.add_argument('model')
p.add_argument('--algo', default='ppo', choices=ALGOS)
p.add_argument('--control', default=os.path.expanduser('~/ros2_ws/runs/live/setpoint.json'))
args = p.parse_args()

import torch
torch.set_num_threads(1)   # evals share a node with real-time sims: torch's default (one thread per core in every
                           # process) starved them on 2026-10-28 (frozen sim state in all Dreamer evals)
model = ALGOS[args.algo].load(args.model, device='cpu')
env = AUVEnv()
env.tracking_cfg['setpoint_change_seconds'] = 0          # setpoints come only from the control file
n = env.node


def read_control():
    try:
        with open(args.control) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


ctl = read_control() or {'depth': 2.0, 'surge': 0.0, 'yaw_deg': 0.0}
obs, _ = env.reset(options={'spawn_depth': float(ctl.get('depth', 2.0)), 'spawn_yaw': np.radians(ctl.get('yaw_deg', 0.0))})
env.max_steps = 10 ** 9
sp = n.setpoint_manager.current_setpoint
last_respawn, t0, k = ctl.get('respawn'), time.time(), 0
print(f"live: {os.path.basename(args.model)} | control file {args.control}", flush=True)
while True:
    c = read_control()
    if c:
        if c.get('respawn') != last_respawn:
            last_respawn = c.get('respawn')
            obs, _ = env.reset(options={'spawn_depth': float(c.get('depth', 2.0)), 'spawn_yaw': np.radians(c.get('yaw_deg', 0.0))})
            getattr(model, 'reset', lambda: None)()   # recurrent policies (dreamer) start a fresh latent
            env.max_steps = 10 ** 9
            sp = n.setpoint_manager.current_setpoint
            print("respawned", flush=True)
        sp.position.z = float(c.get('depth', sp.position.z))
        sp.velocity.x = float(c.get('surge', sp.velocity.x))
        sp.velocity.y = float(c.get('sway', 0.0))           # not commanded in training (always 0)
        sp.orientation.z = float(np.radians(c.get('yaw_deg', np.degrees(sp.orientation.z))))
        sp.orientation.x = float(np.radians(c.get("roll_deg", 0.0)))
        sp.orientation.y = float(np.radians(c.get("pitch_deg", 0.0)))
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, _, info = env.step(action)
    k += 1
    if k % 10 == 0:
        r, pch, y = np.degrees(n.orientation_state)
        print(f"t={time.time() - t0:6.1f}s | depth {n.position_state[2]:5.2f}/{sp.position.z:4.1f} m | "
              f"surge {n.v_state[0]:+5.2f}/{sp.velocity.x:+4.2f} sway {n.v_state[1]:+5.2f}/{sp.velocity.y:+4.2f} m/s | yaw {y:+6.1f}/{np.degrees(sp.orientation.z):+6.1f} deg | "
              f"roll {r:+5.1f} pitch {pch:+5.1f} | thr {np.round(env.thruster_action, 2)}"
              + (f" servo_deg {np.round(np.degrees(n.servo_measured[:env.num_servos]), 1)}" if env.num_servos else "")
              + (" | TERMINATED (limits)" if terminated else ""), flush=True)
