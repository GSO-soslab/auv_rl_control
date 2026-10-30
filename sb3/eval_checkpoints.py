"""Deterministic evaluation of saved training checkpoints (evaluation-return learning curves).

Each checkpoint (one every --every env steps, plus the final model) runs one episode per sim with the deterministic
policy. Env i resets with seed --seed + i, so every checkpoint, run and algorithm sees the same setpoint sequences
(the env draws its setpoints, respawn pose and mid-episode setpoint changes from that seed).
The env settings (RACE2_*) of each run are taken from its run_info/job.json, so one job can evaluate runs with
different settings of the SAME vehicle (the sims are the job's RACE2_VEHICLE).

Run through the vec launcher (it starts the sims and adds --n_envs/--domain_base):
  RACE2_VEHICLE=race2 race2 vec 20 evalckpt -- --runs logs/ppo_... logs/td3_...
Writes <run>/eval_checkpoints[TAG].csv (one row per checkpoint x episode); already evaluated checkpoints are skipped.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vehicles import VEHICLES  # noqa: E402

COLS = ['steps', 'checkpoint', 'env', 'seed', 'reward', 'length', 'terminated',
        'rmse_depth', 'rmse_yaw', 'rmse_surge', 'rmse_pitch', 'rmse_roll', 'energy']
ERR_KEYS = {'rmse_depth': 'depth', 'rmse_yaw': 'yaw', 'rmse_surge': 'surge_cmd', 'rmse_pitch': 'pitch', 'rmse_roll': 'roll'}

DT = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'config_sac.yaml'))
                    )['tracking_reward'].get('control_dt', 0.1)


def checkpoints(run, every):
    """[(steps, path)]: the checkpoint nearest to each multiple of `every`, plus the final model."""
    found = sorted((int(m[1]), f) for f in glob.glob(os.path.join(run, 'checkpoints', '*_steps.*'))
                   if (m := re.search(r'_(\d+)_steps\.(zip|pt)$', f)))
    if not found:
        return []
    steps = np.array([s for s, _ in found])
    picks = {}
    for k in range(1, int(steps.max() // every) + 1):
        i = int(np.argmin(abs(steps - k * every)))
        picks[found[i][0]] = found[i][1]
    final = [f for f in (os.path.join(run, 'final_model.zip'), os.path.join(run, 'final_model.pt')) if os.path.exists(f)]
    if final:
        picks[int(steps.max()) + 1] = final[0]   # final model: just after the last checkpoint
    return sorted(picks.items())


def load_policy(algo, path):
    if algo == 'dreamer':
        from dreamer_agent import DreamerPolicy
        return DreamerPolicy.load(path, 'cuda' if torch.cuda.is_available() else 'cpu')
    from stable_baselines3 import PPO, TD3
    return {'ppo': PPO, 'td3': TD3}[algo].load(path, device='cpu')


def run_episodes(env, policy, n, seed, n_thr):
    env.seed(seed)
    obs = env.reset()
    getattr(policy, 'reset', lambda: None)()
    acc = {k: np.zeros(n) for k in ('reward', 'length', 'terminated', 'energy', *ERR_KEYS)}
    done = np.zeros(n, bool)
    for _ in range(args.max_steps):
        action, _ = policy.predict(obs, deterministic=True)
        obs, rew, dones, infos = env.step(action)
        for i in np.flatnonzero(~done):
            info = infos[i]
            acc['reward'][i] += rew[i]
            acc['length'][i] += 1
            acc['energy'][i] += info['reward_terms']['effort'] * n_thr * DT   # sum_k u_k^2 dt
            for col, key in ERR_KEYS.items():
                acc[col][i] += info['abs_err'][key] ** 2
            if dones[i]:
                done[i] = True
                acc['terminated'][i] = info['reward_terms'].get('terminated', 0.0)
        if done.all():
            break
    for col in ERR_KEYS:
        acc[col] = np.sqrt(acc[col] / np.maximum(acc['length'], 1))
    return acc


def main():
    global args
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', default=[], help='run dirs (sb3/logs/<run>)')
    p.add_argument('--runs-from-jobs', default='', help='comma-separated job-name prefixes: every sb3/logs run whose '
                   'run_info/job.json has such a job name and exit status 0 (resolved now, so it can be queued before training)')
    p.add_argument('--every', type=int, default=100000, help='evaluate the checkpoint nearest to every k*EVERY steps')
    p.add_argument('--seed', type=int, default=1000)
    p.add_argument('--max-steps', type=int, default=700, help='safety cap per episode (episodes are 600 steps)')
    p.add_argument('--n_envs', type=int, required=True)
    p.add_argument('--domain_base', type=int, default=None)
    p.add_argument('--tag', default='', help='output eval_checkpoints<TAG>.csv (e.g. _test for a trial run)')
    p.add_argument('--timesteps', default=None, help='ignored (passed by train_vec.sbatch)')
    args = p.parse_args()
    assert args.n_envs >= 2, 'needs the vec env (n_envs >= 2)'
    if args.runs_from_jobs:
        prefixes = tuple(args.runs_from_jobs.split(','))
        for jf in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', '*', 'run_info', 'job.json'))):
            j = json.load(open(jf))
            if (j.get('job_name') or '').startswith(prefixes) and j.get('exit_status') == 0:
                args.runs.append(os.path.dirname(os.path.dirname(jf)))
        print(f'runs from jobs {args.runs_from_jobs}: {len(args.runs)}', flush=True)
    assert args.runs, 'no runs (--runs or --runs-from-jobs)'
    torch.set_num_threads(1)   # torch's default thread count starves the real-time sims
    from vec_env_utils import make_env

    job_vehicle = os.environ.get('RACE2_VEHICLE', 'race2')
    # fallback for runs without run_info/job.json (written at job end, so missing after a TIMEOUT):
    # the RACE2_* settings of THIS eval job, which must then be the run's training settings
    job_settings = {k: v for k, v in os.environ.items() if k.startswith('RACE2_')}
    for run in args.runs:
        run = os.path.abspath(run)
        jf = os.path.join(run, 'run_info', 'job.json')
        if os.path.exists(jf):
            info = json.load(open(jf))
            algo, settings = info['trainer'], info['settings']
        else:
            algo, settings = os.path.basename(run).split('_')[0], job_settings
            print(f'{os.path.basename(run)}: no run_info/job.json (timed out?), using this job\'s RACE2_* settings', flush=True)
        vehicle = settings.get('RACE2_VEHICLE', 'race2')
        if vehicle != job_vehicle:
            print(f'skip {os.path.basename(run)}: vehicle {vehicle} but the sims are {job_vehicle}', flush=True)
            continue
        out = os.path.join(run, f'eval_checkpoints{args.tag}.csv')
        done_steps = {int(r['steps']) for r in csv.DictReader(open(out))} if os.path.exists(out) else set()
        todo = [(s, f) for s, f in checkpoints(run, args.every) if s not in done_steps]
        print(f'{os.path.basename(run)}: {algo}, {len(todo)} checkpoints to evaluate', flush=True)
        if not todo:
            continue
        # the env subprocesses read RACE2_* at construction: this run's settings only
        for k in [k for k in os.environ if k.startswith('RACE2_')]:
            del os.environ[k]
        os.environ.update(settings)
        env = make_env(args.n_envs, args.domain_base)
        n_thr = len(VEHICLES[vehicle]['thrusters'])
        new = not os.path.exists(out)
        with open(out, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            if new:
                w.writeheader()
            for steps, path in todo:
                acc = run_episodes(env, load_policy(algo, path), args.n_envs, args.seed, n_thr)
                for i in range(args.n_envs):
                    w.writerow({'steps': steps, 'checkpoint': os.path.basename(path), 'env': i, 'seed': args.seed + i,
                                **{k: round(float(v[i]), 5) for k, v in acc.items()}})
                f.flush()
                print(f'  {steps:>8d}: reward {acc["reward"].mean():7.1f} +- {acc["reward"].std():5.1f}, '
                      f'terminated {int(acc["terminated"].sum())}/{args.n_envs}', flush=True)
        env.close()
    os.environ['RACE2_VEHICLE'] = job_vehicle
    print('eval_checkpoints done', flush=True)


if __name__ == '__main__':   # the env subprocesses are spawned: they re-import this module
    main()
