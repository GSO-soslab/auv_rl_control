"""DreamerV3 (Hafner et al. 2023; PyTorch port NM512/dreamerv3-torch) for the RACE2 tracking env.

World model = RSSM (GRU deterministic state + discrete stochastic latents) with MLP encoder/decoder/heads;
actor and critic are MLPs on the latent features. No image inputs: the env's vector observation is the
single `state` key. The port is used unchanged except WorldModel.preprocess, which assumes an image.

Upstream checkout: $DREAMERV3_TORCH (default ~/dreamerv3-torch), pinned to commit DREAMERV3_TORCH_COMMIT:
  git clone https://github.com/NM512/dreamerv3-torch ~/dreamerv3-torch && git -C ~/dreamerv3-torch checkout 6ef8646
Config: upstream `defaults` + `dmc_proprio` preset + sb3/config/config_dreamer.yaml `dreamer:` overrides.
"""
import argparse
import os
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import torch
from gymnasium import spaces

DREAMERV3_TORCH = pathlib.Path(os.environ.get('DREAMERV3_TORCH', '~/dreamerv3-torch')).expanduser()
DREAMERV3_TORCH_COMMIT = '6ef8646'
# _deps: ruamel.yaml, the one upstream dependency the container lacks
# (pip install --target ~/dreamerv3-torch/_deps 'ruamel.yaml<0.19'; the container's Python is PEP 668 managed)
sys.path[:0] = [str(DREAMERV3_TORCH), str(DREAMERV3_TORCH / '_deps')]
from ruamel.yaml import YAML  # noqa: E402
import models  # noqa: E402  (dreamerv3-torch)
import tools   # noqa: E402
from dreamer import Dreamer  # noqa: E402

OBS_KEY = 'state'


def _preprocess(self, obs):
    """models.WorldModel.preprocess without the image scaling (vector observations only)."""
    obs = {k: torch.as_tensor(np.asarray(v), device=self._config.device, dtype=torch.float32) for k, v in obs.items()}
    if 'discount' in obs:
        obs['discount'] = (obs['discount'] * self._config.discount).unsqueeze(-1)
    obs['cont'] = (1.0 - obs['is_terminal']).unsqueeze(-1)
    return obs


models.WorldModel.preprocess = _preprocess


def load_config(config_path, overrides=None):
    """Upstream defaults + dmc_proprio, then our yaml, then CLI overrides (dict). Returns a namespace."""
    load = YAML(typ='safe').load   # YAML 1.2 (as upstream): 1e-4 is a float
    upstream = load((DREAMERV3_TORCH / 'configs.yaml').read_text())
    ours = load(open(config_path).read())

    def update(base, new):
        for k, v in new.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                update(base[k], v)
            else:
                base[k] = v

    cfg = {}
    for part in (upstream['defaults'], upstream['dmc_proprio'], ours['dreamer'], overrides or {}):
        update(cfg, part)
    for k in ('steps', 'eval_every', 'log_every', 'dataset_size', 'prefill', 'pretrain', 'expl_until', 'reset_every'):
        cfg[k] = int(float(cfg[k]))
    return argparse.Namespace(**cfg)


def spaces_for(obs_dim, n_actions):
    obs_space = spaces.Dict({OBS_KEY: spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)})
    act_space = spaces.Box(-1.0, 1.0, (n_actions,), np.float32)
    return obs_space, act_space


def make_agent(config, obs_dim, n_actions, logger=None, dataset=None):
    config.num_actions = n_actions
    obs_space, act_space = spaces_for(obs_dim, n_actions)
    agent = Dreamer(obs_space, act_space, config, logger or SimpleNamespace(step=0), dataset).to(config.device)
    agent.requires_grad_(requires_grad=False)   # as upstream: the optimizers enable grads inside their own scopes
    return agent


def policy_obs(obs, is_first):
    """Batch of env observations -> the dict the agent's policy takes."""
    obs = np.asarray(obs, dtype=np.float32)
    n = len(obs)
    return {OBS_KEY: obs, 'is_first': np.asarray(is_first, dtype=bool),
            'is_last': np.zeros(n, bool), 'is_terminal': np.zeros(n, bool)}


class DreamerPolicy:
    """SB3-like wrapper for evaluation: DreamerPolicy.load(path).predict(obs, deterministic=True).
    Keeps the recurrent latent between calls; call reset() at an env reset (the first call starts fresh)."""

    def __init__(self, agent):
        self.agent, self.state, self.first = agent, None, True

    @classmethod
    def load(cls, path, device='cpu'):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        config = argparse.Namespace(**ckpt['config'])
        config.device, config.compile = device, False
        agent = make_agent(config, ckpt['obs_dim'], ckpt['n_actions'])
        agent.load_state_dict(ckpt['agent_state_dict'])
        agent.eval()
        return cls(agent)

    def reset(self):
        self.state, self.first = None, True

    def predict(self, obs, deterministic=True):
        single = np.ndim(obs) == 1
        o = policy_obs(np.atleast_2d(obs), [self.first] * (1 if single else len(obs)))
        with torch.no_grad():
            out, self.state = self.agent._policy(o, self.state, training=not deterministic)
        self.first = False
        action = out['action'].cpu().numpy()
        return (action[0] if single else action), None
