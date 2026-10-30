"""Log per-DOF tracking errors and reward terms (from AUVEnv tracking mode `info`) to TensorBoard.

Values are averaged over all envs and steps between logger dumps:
  track/abs_err_<dof>   absolute setpoint error (m, m/s, rad)
  reward/<term>         reward components (per-DOF scores in [0, 1], rate, effort, oob)
"""
from stable_baselines3.common.callbacks import BaseCallback


class TrackingLogger(BaseCallback):
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for k, v in info.get("abs_err", {}).items():
                self.logger.record_mean(f"track/abs_err_{k}", v)
            for k, v in info.get("reward_terms", {}).items():
                self.logger.record_mean(f"reward/{k}", v)
        return True
