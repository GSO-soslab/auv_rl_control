"""Run a trained SB3 policy deterministically in the (running) sim and record tracking errors.

usage: python3 eval_policy.py <model.zip> [--algo ddpg|ppo|sac] [--episodes 2] [--steps 300]
Writes <model>_eval.csv (per step: setpoint, state, errors, reward, action) and prints a summary.
Needs `race2 sim --rl` running on the same ROS_DOMAIN_ID.
"""
import argparse
import csv
import os

import numpy as np
from stable_baselines3 import DDPG, PPO, SAC, TD3

from AUVEnv import AUVEnv

ALGOS = {'ddpg': DDPG, 'ppo': PPO, 'sac': SAC, 'td3': TD3}

p = argparse.ArgumentParser()
p.add_argument('model')
p.add_argument('--algo', default='ddpg', choices=ALGOS)
p.add_argument('--episodes', type=int, default=2)
p.add_argument('--steps', type=int, default=300)
args = p.parse_args()

model = ALGOS[args.algo].load(args.model, device='cpu')
env = AUVEnv()
out = os.path.splitext(args.model)[0] + '_eval.csv'
dofs = ['depth', 'surge', 'pitch', 'yaw', 'roll']

with open(out, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['episode', 'step', 'sp_depth', 'depth', 'sp_surge', 'surge', 'sp_yaw', 'yaw', 'reward']
               + [f'err_{d}' for d in dofs] + [f'a{i}' for i in range(env.action_space.shape[0])])
    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep)
        sp = env.node.setpoint_manager.current_setpoint
        errs = []
        for t in range(args.steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            e, n = info['abs_err'], env.node
            errs.append([e[d] for d in dofs])
            w.writerow([ep, t, sp.position.z, n.position_state[2], sp.velocity.x, n.v_state[0],
                        sp.orientation.z, n.orientation_state[2], reward]
                       + [e[d] for d in dofs] + list(np.round(action, 3)))
            if t % 25 == 0:
                print(f"ep {ep} t {t:3d}  depth {n.position_state[2]:6.2f}/{sp.position.z:5.2f}  "
                      f"surge {n.v_state[0]:6.3f}/{sp.velocity.x:6.3f}  yaw err {e['yaw']:5.2f}  r {reward:6.3f}",
                      flush=True)
            if terminated or truncated:
                break
        last = np.array(errs[-50:])
        print(f"episode {ep} | mean |err| over last 50 steps: "
              + ", ".join(f"{d} {v:.3f}" for d, v in zip(dofs, last.mean(axis=0))), flush=True)
print(f"wrote {out}")
