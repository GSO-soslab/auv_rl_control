"""Run one SB3 learner on N Stonefish sims in parallel.

Each sim runs on its own ROS_DOMAIN_ID (domain_base, domain_base+1, ...). Each env lives in
its own subprocess so it can set ROS_DOMAIN_ID before rclpy.init(), giving every AUVEnv
its own isolated ROS graph. SB3 seeds env i with seed+i, so every sim gets different setpoints.
"""
import os
from functools import partial

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv


def checkpoint_dir(log_dir):
    """<log_dir>/checkpoints, created as a symlink into /scratch/$USER/race2_logs when scratch exists: checkpoints are
    most of a run's size and home is capped (~50 GB; it filled up on 2026-10-30 and blocked logins)."""
    path = os.path.join(str(log_dir), 'checkpoints')
    scratch = os.path.join('/scratch', os.environ.get('USER', ''))
    if not os.path.lexists(path) and os.path.isdir(scratch):
        target = os.path.join(scratch, 'race2_logs', os.path.basename(os.path.normpath(str(log_dir))), 'checkpoints')
        os.makedirs(target, exist_ok=True)
        os.makedirs(str(log_dir), exist_ok=True)
        os.symlink(target, path)
    os.makedirs(path, exist_ok=True)
    return path


def domain_for(domain_base, i):
    return 1 + (domain_base - 1 + i) % 101


def _make_auv_env(domain_id):
    os.environ['ROS_DOMAIN_ID'] = str(domain_id)
    from AUVEnv import AUVEnv  # imported after ROS_DOMAIN_ID is set
    return Monitor(AUVEnv())


def make_env(n_envs=1, domain_base=None):
    """n_envs == 1: a plain AUVEnv (old behaviour). n_envs > 1: SubprocVecEnv over N sims."""
    if n_envs <= 1:
        from AUVEnv import AUVEnv
        return AUVEnv()
    if domain_base is None:
        domain_base = int(os.environ.get('ROS_DOMAIN_ID', 0))
    domains = [domain_for(domain_base, i) for i in range(n_envs)]
    print(f"Vectorized env: {n_envs} sims on ROS domains {domains}")
    return SubprocVecEnv([partial(_make_auv_env, d) for d in domains], start_method='spawn')
