#!/usr/bin/env python3
"""Online DreamerV3 on N Stonefish sims (same env, reward and logging as ppo_sb3.py / td3_sb3.py).

The sims run in real time at the env's 10 Hz control period, so collection and training are decoupled
(upstream dreamerv3-torch trains between env steps, which would stretch the control period here):
  * main thread: steps the vectorized env with an acting copy of the agent and writes the episode replay
  * learner thread: samples replay and trains continuously, capped at --train_ratio replayed steps per env
    step, and copies its weights into the acting copy every --sync_every updates
Run through `race2 vec N dreamer -- --timesteps ...`. Writes progress.csv (SB3 column names where they exist),
TensorBoard events, checkpoints/dreamer_auv_model_<steps>_steps.pt and final_model.pt (load with
dreamer_agent.DreamerPolicy).

Long runs over several Slurm jobs (8 h limit): give every job the same --run_id and --max_hours below the limit. A job
trains until --timesteps or --max_hours, saves resume_state.pt (weights, optimizers, replay, counters, logs) and exits;
the next job with the same run id continues from it (a finished run exits at once). resume_state.pt is also refreshed
every --resume_every steps, so a job killed at the limit loses at most that many steps.
"""
import argparse
import collections
import copy
import csv
import os
import pathlib
import threading
import time
from datetime import datetime

import numpy as np
import torch

from dreamer_agent import DREAMERV3_TORCH_COMMIT, load_config, make_agent, policy_obs, tools, OBS_KEY
from td3_sb3 import RUN_TAG_KEYS
from vec_env_utils import checkpoint_dir, make_env


def main():
    parser = argparse.ArgumentParser(description='Train DreamerV3 for AUV control')
    parser.add_argument('--config', type=str, default='config/config_dreamer.yaml', help='Path to config file')
    parser.add_argument('--timesteps', type=int, required=True, help='Total env steps')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (default: config seed)')
    parser.add_argument('--n_envs', type=int, default=1, help='Parallel sims (each on its own ROS domain)')
    parser.add_argument('--domain_base', type=int, default=None, help='ROS_DOMAIN_ID of sim 0 (default: $ROS_DOMAIN_ID)')
    parser.add_argument('--train_ratio', type=float, default=None, help='Max replayed steps per env step (overrides config)')
    parser.add_argument('--sync_every', type=int, default=20, help='Learner updates between weight copies to the acting agent')
    parser.add_argument('--checkpoint_every', type=int, default=100000, help='Env steps between checkpoints')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Fixed run dir id instead of the start time; resumes from its resume_state.pt if present')
    parser.add_argument('--max_hours', type=float, default=None, help='Save resume state and exit after this wall time')
    parser.add_argument('--resume_every', type=int, default=500000, help='Env steps between resume-state saves (with --run_id)')
    args = parser.parse_args()

    overrides = {k: v for k, v in (('seed', args.seed), ('train_ratio', args.train_ratio)) if v is not None}
    config = load_config(os.path.join(os.path.dirname(__file__), args.config), overrides)
    print(f"Random seed: {config.seed}")
    tools.set_seed_everywhere(config.seed)

    env = make_env(args.n_envs, args.domain_base)
    if not hasattr(env, 'num_envs'):
        from stable_baselines3.common.vec_env import DummyVecEnv
        env = DummyVecEnv([lambda e=env: e])
    n, obs_dim, n_act = env.num_envs, env.observation_space.shape[0], env.action_space.shape[0]
    run_tag = "".join(f"_{k.split('_', 1)[1].lower()}-{os.environ[k]}" for k in RUN_TAG_KEYS if k in os.environ)
    log_dir = pathlib.Path("logs") / f"dreamer_{args.run_id or datetime.now().strftime('%Y%m%d_%H%M%S')}{run_tag}_seed-{config.seed}"
    if (log_dir / 'final_model.pt').exists():
        print(f"{log_dir} already finished (final_model.pt): nothing to do")
        return
    resume = torch.load(log_dir / 'resume_state.pt', map_location=config.device, weights_only=False) \
        if (log_dir / 'resume_state.pt').exists() else None
    checkpoint_dir(log_dir)   # on /scratch (vec_env_utils)
    print(f"Logging to {log_dir}")
    print(f"DreamerV3 (dreamerv3-torch {DREAMERV3_TORCH_COMMIT}): {args.timesteps} steps, {n} envs, obs {obs_dim}, "
          f"actions {n_act}, train_ratio {config.train_ratio}, batch {config.batch_size}x{config.batch_length}")

    # Replay: episodes keyed in collection order (erase_over_episodes drops the oldest first)
    cache, data_lock, sync_lock = collections.OrderedDict(), threading.Lock(), threading.Lock()
    dataset = tools.from_generator(tools.sample_episodes(cache, config.batch_length, config.seed), config.batch_size)
    logger = tools.Logger(log_dir, resume['steps'] if resume else 0)
    agent = make_agent(config, obs_dim, n_act, logger, dataset)
    opts = {'model': agent._wm._model_opt._opt, 'actor': agent._task_behavior._actor_opt._opt,
            'value': agent._task_behavior._value_opt._opt}
    if resume:
        agent.load_state_dict(resume['agent_full'])
        for k, o in opts.items():
            o.load_state_dict(resume['optimizers'][k])
        cache.update(resume['cache'])
        print(f"Resumed {log_dir} at {resume['steps']} steps, {resume['updates']} updates, {len(cache)} replay episodes")
    actor = make_agent(argparse.Namespace(**{**vars(config), 'compile': False}), obs_dim, n_act)   # acting copy

    def weights():   # learner weights without torch.compile's '_orig_mod.' prefix (loadable uncompiled)
        return {k.replace('_orig_mod.', ''): v for k, v in agent.state_dict().items()}

    actor.load_state_dict(weights())
    print(f"Agent parameters: {sum(p.numel() for p in agent.parameters()) / 1e6:.1f}M")

    resumed = resume is not None
    shared = {'steps': resume['steps'] if resume else 0, 'updates': resume['updates'] if resume else 0,
              'stop': False, 'error': None}
    train_lock = threading.Lock()   # resume-state saves take a consistent snapshot between updates
    seq_steps = config.batch_size * config.batch_length

    def learner():
        try:
            pretrain = 0 if resumed else config.pretrain   # updates on the prefill data first (upstream), then rate-capped
            while not shared['stop']:
                if pretrain > 0:
                    pretrain -= 1
                elif shared['updates'] * seq_steps >= shared['steps'] * config.train_ratio:
                    time.sleep(0.005)
                    continue
                with data_lock:
                    batch = next(dataset)
                with train_lock:
                    agent._train(batch)
                    shared['updates'] += 1
                if shared['updates'] % args.sync_every == 0:
                    with sync_lock:
                        actor.load_state_dict(weights())
        except Exception as e:   # surfaced by the main loop
            shared['error'] = e

    def save(path):
        torch.save({'agent_state_dict': weights(), 'config': {**vars(config), 'compile': False}, 'obs_dim': obs_dim,
                    'n_actions': n_act, 'steps': shared['steps'], 'updates': shared['updates'],
                    'dreamerv3_torch_commit': DREAMERV3_TORCH_COMMIT}, path)

    def save_resume():
        """Full training state; in-progress episodes are dropped (the next job starts fresh episodes).
        Finished episodes are stacked into arrays in place (they never change again; the sampler accepts both), so
        the locks are held only briefly and the file is written outside them."""
        with data_lock:
            for k, ep in cache.items():
                if k not in episode_ids and isinstance(next(iter(ep.values())), list):
                    cache[k] = {kk: np.stack(vv) for kk, vv in ep.items()}
            replay = {k: v for k, v in cache.items() if k not in episode_ids}
        with train_lock:
            agent_full = {k: v.detach().clone() for k, v in agent.state_dict().items()}
            optimizers = {k: copy.deepcopy(o.state_dict()) for k, o in opts.items()}
            counters = {'steps': shared['steps'], 'updates': shared['updates']}
        torch.save({'agent_full': agent_full, 'optimizers': optimizers, **counters, 'next_episode': next_episode,
                    'next_ckpt': next_ckpt, 'rows': list(rows), 'ep_rew': list(ep_rew), 'ep_len': list(ep_len),
                    'cache': replay}, log_dir / 'resume_state.pt.tmp')
        os.replace(log_dir / 'resume_state.pt.tmp', log_dir / 'resume_state.pt')

    next_episode = resume['next_episode'] if resume else n
    episode_ids = [f"{next_episode + k:06d}-env{k:02d}" for k in range(n)]
    next_episode += n
    def start_episode(i, o):
        tools.add_to_cache(cache, episode_ids[i], {OBS_KEY: o, 'reward': 0.0, 'discount': 1.0,
                                                  'is_first': True, 'is_last': False, 'is_terminal': False})

    obs = env.reset()
    with data_lock:
        for i in range(n):
            start_episode(i, obs[i])
    is_first, state = np.ones(n, bool), None
    rng = np.random.default_rng(config.seed)
    learner_thread = threading.Thread(target=learner, daemon=True)
    rows, sums, counts, ep_rew, ep_len = (resume['rows'] if resume else []), collections.defaultdict(float), \
        collections.defaultdict(int), collections.deque(resume['ep_rew'] if resume else [], maxlen=100), \
        collections.deque(resume['ep_len'] if resume else [], maxlen=100)
    t0 = t_log = time.time()
    t0_run = t0 - (rows[-1]['time/time_elapsed'] if rows else 0)   # time_elapsed continues across jobs
    steps_log, updates_log = shared['steps'], shared['updates']
    next_ckpt = resume['next_ckpt'] if resume else args.checkpoint_every
    next_resume = shared['steps'] + args.resume_every
    resume = None   # free the loaded copy (cache/rows now live in the objects above)
    out_of_time = False

    try:
        while shared['steps'] < args.timesteps:
            if shared['error'] is not None:
                raise shared['error']
            if args.max_hours and time.time() - t0 > args.max_hours * 3600:
                out_of_time = True
                break
            if shared['steps'] < config.prefill:
                action = rng.uniform(-1.0, 1.0, (n, n_act)).astype(np.float32)
            else:
                with sync_lock, torch.no_grad():
                    out, state = actor._policy(policy_obs(obs, is_first), state, training=True)
                action = np.clip(out['action'].cpu().numpy(), -1.0, 1.0)
            new_obs, reward, done, infos = env.step(action)
            with data_lock:
                for i in range(n):
                    terminal = bool(done[i]) and not infos[i].get('TimeLimit.truncated', False)
                    o = infos[i]['terminal_observation'] if done[i] else new_obs[i]
                    tools.add_to_cache(cache, episode_ids[i], {
                        OBS_KEY: o, 'action': action[i], 'reward': float(reward[i]), 'discount': 0.0 if terminal else 1.0,
                        'is_first': False, 'is_last': bool(done[i]), 'is_terminal': terminal})
                    if done[i]:
                        episode_ids[i] = f"{next_episode:06d}-env{i:02d}"
                        next_episode += 1
                        start_episode(i, new_obs[i])
                if done.any():
                    tools.erase_over_episodes(cache, config.dataset_size)
            for info in infos:
                for group, prefix in (('abs_err', 'track/abs_err_'), ('reward_terms', 'reward/')):
                    for k, v in info.get(group, {}).items():
                        sums[prefix + k] += v
                        counts[prefix + k] += 1
                if 'episode' in info:
                    ep_rew.append(info['episode']['r'])
                    ep_len.append(info['episode']['l'])
            obs, is_first = new_obs, done.copy()
            shared['steps'] += n
            if shared['steps'] >= config.prefill and not learner_thread.is_alive():
                print(f"Prefill done ({shared['steps']} steps): learner starts")
                learner_thread.start()

            if shared['steps'] - steps_log >= config.log_every:
                now = time.time()
                metrics, agent._metrics = agent._metrics, {}
                row = {'time/total_timesteps': shared['steps'], 'time/time_elapsed': int(now - t0_run),
                       'time/fps': (shared['steps'] - steps_log) / (now - t_log),
                       'train/updates': shared['updates'],
                       'train/updates_per_s': (shared['updates'] - updates_log) / (now - t_log),
                       'train/replay_ratio': shared['updates'] * seq_steps / shared['steps'],
                       'rollout/ep_rew_mean': float(np.mean(ep_rew)) if ep_rew else float('nan'),
                       'rollout/ep_len_mean': float(np.mean(ep_len)) if ep_len else float('nan'),
                       **{k: sums[k] / counts[k] for k in sums},
                       **{f'train/{k}': float(np.mean([float(x) for x in v])) for k, v in metrics.items() if len(v)}}
                rows.append(row)
                fields = list(dict.fromkeys(k for r in rows for k in r))
                with open(log_dir / 'progress.csv', 'w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=fields, restval='')
                    w.writeheader()
                    w.writerows(rows)
                for k, v in row.items():
                    logger.scalar(k, v)
                logger.write(step=shared['steps'])
                sums.clear()
                counts.clear()
                steps_log, updates_log, t_log = shared['steps'], shared['updates'], now
            if shared['steps'] >= next_ckpt:
                save(log_dir / 'checkpoints' / f"dreamer_auv_model_{shared['steps']}_steps.pt")
                next_ckpt += args.checkpoint_every
                if args.run_id and shared['steps'] >= next_resume:
                    save_resume()
                    next_resume += args.resume_every
        shared['stop'] = True
        learner_thread.join(timeout=60)
        if out_of_time:
            save_resume()
            print(f"--max_hours reached at {shared['steps']} steps: resume state saved to {log_dir}/resume_state.pt")
        else:
            save(log_dir / 'final_model.pt')
            (log_dir / 'resume_state.pt').unlink(missing_ok=True)   # ~0.6-1.3 GB of replay; only needed mid-run
            print(f"Training completed. Final model saved to {log_dir}/final_model.pt")
    except KeyboardInterrupt:
        shared['stop'] = True
        save(log_dir / 'interrupted_model.pt')
        print(f"Interrupted model saved to {log_dir}/interrupted_model.pt")


if __name__ == "__main__":
    import multiprocessing, sys, traceback
    code = 0
    try:
        main()
    except BaseException:   # e.g. a full disk on save (2026-10-30): exit now with the error, don't hang
        traceback.print_exc()
        code = 1
    # ROS spin threads inside the SubprocVecEnv workers keep the interpreter alive after training (and after an
    # exception), so stop the workers explicitly and exit (otherwise a batch job holds its node until the time limit).
    sys.stdout.flush()
    sys.stderr.flush()
    for p in multiprocessing.active_children():
        p.terminate()
    os._exit(code)
