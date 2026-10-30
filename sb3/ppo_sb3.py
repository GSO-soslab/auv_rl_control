#!/usr/bin/env python3

import os
import numpy as np
import yaml
import time
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import copy

# Stable Baselines 3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv
import torch as th
import torch.nn as nn
import torch.optim as optim

# For environment compatibility
import gym

# Import custom environment
from AUVEnv import AUVEnv
from vec_env_utils import checkpoint_dir, make_env
from tracking_logger import TrackingLogger

class GradientMonitoringCallback(BaseCallback):
    """
    Custom callback for monitoring gradients and network weight updates (PPO version)
    """
    def __init__(self, monitor_interval=100, verbose=1):
        super(GradientMonitoringCallback, self).__init__(verbose)
        self.monitor_interval = monitor_interval
        self.step_count = 0
        self.previous_weights = {}
        self.gradient_norms = {'actor': [], 'critic': []}
        self.weight_changes = {'actor': [], 'critic': []}
        self.update_counts = {'actor': 0, 'critic': 0}
        
    def _on_training_start(self) -> None:
        """Initialize weight tracking"""
        if hasattr(self.model, 'policy'):
            # Store initial weights
            self._store_current_weights()
            print("Gradient and weight monitoring initialized.")
            print(f"Network architecture:")
            self._print_network_info()

    def _store_current_weights(self):
        """Store current network weights for comparison"""
        if hasattr(self.model.policy, 'action_net'):
            self.previous_weights['actor'] = self._get_network_weights(self.model.policy.action_net)
        if hasattr(self.model.policy, 'value_net'):
            self.previous_weights['critic'] = self._get_network_weights(self.model.policy.value_net)

    def _get_network_weights(self, network):
        """Extract weights from a network"""
        weights = {}
        for name, param in network.named_parameters():
            weights[name] = param.data.clone().detach()
        return weights

    def _calculate_weight_changes(self):
        """Calculate the magnitude of weight changes"""
        changes = {}
        
        # Actor network
        if 'actor' in self.previous_weights and hasattr(self.model.policy, 'action_net'):
            current_actor_weights = self._get_network_weights(self.model.policy.action_net)
            actor_change = 0
            for name, current_weight in current_actor_weights.items():
                if name in self.previous_weights['actor']:
                    diff = current_weight - self.previous_weights['actor'][name]
                    actor_change += th.norm(diff).item()
            changes['actor'] = actor_change
            
        # Critic network
        if 'critic' in self.previous_weights and hasattr(self.model.policy, 'value_net'):
            current_critic_weights = self._get_network_weights(self.model.policy.value_net)
            critic_change = 0
            for name, current_weight in current_critic_weights.items():
                if name in self.previous_weights['critic']:
                    diff = current_weight - self.previous_weights['critic'][name]
                    critic_change += th.norm(diff).item()
            changes['critic'] = critic_change
                    
        return changes

    def _get_gradient_norms(self):
        """Calculate gradient norms for each network"""
        grad_norms = {}
        
        # Actor gradients
        if hasattr(self.model.policy, 'action_net'):
            actor_grad_norm = 0
            for param in self.model.policy.action_net.parameters():
                if param.grad is not None:
                    actor_grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norms['actor'] = actor_grad_norm ** 0.5
            
        # Critic gradients
        if hasattr(self.model.policy, 'value_net'):
            critic_grad_norm = 0
            for param in self.model.policy.value_net.parameters():
                if param.grad is not None:
                    critic_grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norms['critic'] = critic_grad_norm ** 0.5
                
        return grad_norms

    def _print_network_info(self):
        """Print network architecture information"""
        if hasattr(self.model.policy, 'action_net'):
            print("Actor Network:")
            for name, module in self.model.policy.action_net.named_modules():
                if isinstance(module, nn.Linear):
                    print(f"  {name}: {module.in_features} -> {module.out_features}")
                    
        if hasattr(self.model.policy, 'value_net'):
            print("Critic Network:")
            for name, module in self.model.policy.value_net.named_modules():
                if isinstance(module, nn.Linear):
                    print(f"  {name}: {module.in_features} -> {module.out_features}")

    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Monitor at specified intervals
        if self.step_count % self.monitor_interval == 0:
            # Calculate gradient norms
            grad_norms = self._get_gradient_norms()
            
            # Calculate weight changes
            weight_changes = self._calculate_weight_changes()
            
            # Store gradient norms
            for network, norm in grad_norms.items():
                self.gradient_norms[network].append(norm)
                
            # Store weight changes and update counts
            for network, change in weight_changes.items():
                self.weight_changes[network].append(change)
                if change > 1e-8:  # Threshold for considering a weight update
                    self.update_counts[network] += 1
                    
            # Log information
            if self.verbose > 0:
                print(f"\n--- Step {self.step_count} Monitoring ---")
                print("Gradient Norms:")
                for network, norm in grad_norms.items():
                    print(f"  {network}: {norm:.6f}")
                    
                print("Weight Changes (L2 norm):")
                for network, change in weight_changes.items():
                    print(f"  {network}: {change:.6f}")
                    
                print("Update Counts:")
                for network, count in self.update_counts.items():
                    print(f"  {network}: {count} updates")
                    
            # Log to tensorboard
            if self.model.logger:
                for network, norm in grad_norms.items():
                    self.model.logger.record(f"gradients/{network}_norm", norm)
                for network, change in weight_changes.items():
                    self.model.logger.record(f"weights/{network}_change", change)
                for network, count in self.update_counts.items():
                    self.model.logger.record(f"updates/{network}_count", count)
                    
            # Update stored weights for next comparison
            self._store_current_weights()
            
            # Save gradient plots
            if self.model.logger and self.model.logger.dir:
                self._save_gradient_plots()
                
        return True
        
    def _save_gradient_plots(self):
        """Save gradient and weight change plots"""
        if not self.gradient_norms['actor']:  # No data yet
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Gradient norms plot
        axes[0, 0].set_title('Gradient Norms Over Time')
        for network, norms in self.gradient_norms.items():
            if norms:
                axes[0, 0].plot(norms, label=network)
        axes[0, 0].set_xlabel('Monitoring Steps')
        axes[0, 0].set_ylabel('Gradient Norm')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 0].set_yscale('log')
        
        # Weight changes plot
        axes[0, 1].set_title('Weight Changes Over Time')
        for network, changes in self.weight_changes.items():
            if changes:
                axes[0, 1].plot(changes, label=network)
        axes[0, 1].set_xlabel('Monitoring Steps')
        axes[0, 1].set_ylabel('Weight Change (L2 Norm)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 1].set_yscale('log')
        
        # Update counts bar plot
        networks = list(self.update_counts.keys())
        counts = list(self.update_counts.values())
        axes[1, 0].bar(networks, counts)
        axes[1, 0].set_title('Total Weight Updates by Network')
        axes[1, 0].set_ylabel('Update Count')
        axes[1, 0].grid(True, axis='y')
        
        # Recent gradient norms (last 20 monitoring steps)
        axes[1, 1].set_title('Recent Gradient Norms (Last 20 Steps)')
        for network, norms in self.gradient_norms.items():
            if norms:
                recent_norms = norms[-20:] if len(norms) > 20 else norms
                axes[1, 1].plot(recent_norms, label=network, marker='o')
        axes[1, 1].set_xlabel('Recent Monitoring Steps')
        axes[1, 1].set_ylabel('Gradient Norm')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        plot_path = os.path.join(self.model.logger.dir, "gradient_monitoring.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

class RewardPlottingCallback(BaseCallback):
    """
    Custom callback for plotting rewards during training
    """
    def __init__(self, plot_interval=10, verbose=0):
        super(RewardPlottingCallback, self).__init__(verbose)
        self.plot_interval = plot_interval
        self.rewards = []
        self.moving_avg_rewards = []
        self.episodes = []
        self.episode_count = 0
        self.episode_reward = 0
        self.window_size = 20  # For moving average
        self.initial_episode_offset = 0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        if self.model.num_timesteps > 0 and self.model.logger:
            try:
                print(f"Resuming training. Plot episode count will restart from 1 for this session.")
            except Exception:
                pass

    def _on_step(self) -> bool:
        # Accumulate reward
        self.episode_reward += self.locals["rewards"][0]

        # Check if episode is done
        if self.locals["dones"][0]:
            self.episode_count += 1
            self.rewards.append(self.episode_reward)
            self.episodes.append(self.initial_episode_offset + self.episode_count)
            self.episode_reward = 0

            # Calculate moving average
            if len(self.rewards) >= self.window_size:
                avg = np.mean(self.rewards[-self.window_size:])
            else:
                avg = np.mean(self.rewards)
            self.moving_avg_rewards.append(avg)

            # Plot at specified intervals
            if self.episode_count % self.plot_interval == 0:
                plt.figure(figsize=(10, 6))
                plt.plot(self.episodes, self.rewards, 'b-', alpha=0.3, label='Episode Reward')
                plt.plot(self.episodes, self.moving_avg_rewards, 'r-', label=f'Moving Avg ({self.window_size} episodes)')
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.title('Training Rewards (Current Session)')
                plt.legend()
                plt.grid(True)
                if self.model.logger and self.model.logger.dir:
                    os.makedirs(self.model.logger.dir, exist_ok=True)
                    plt.savefig(f"{self.model.logger.dir}/reward_plot_session.png")
                plt.close()

                # Log to stable-baselines logger
                if self.model.logger:
                    self.model.logger.record("reward/episode_reward", self.rewards[-1])
                    self.model.logger.record("reward/moving_avg", self.moving_avg_rewards[-1])
                    self.model.logger.record("reward/episode_count_session", self.episode_count)

                if self.verbose > 0:
                    print(f"Episode {self.episode_count} (Session), Reward: {self.rewards[-1]:.2f}, Moving Avg: {self.moving_avg_rewards[-1]:.2f}")
        return True

def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration from {config_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Train PPO agent with SGD optimizer for AUV control')
    parser.add_argument('--config', type=str, default='config/config_ppo.yaml', help='Path to config file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='Training or testing mode')
    parser.add_argument('--model', type=str, default=None, help='Path to model file for testing OR initial model for transfer learning')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Path to a .zip model file to resume training from')
    parser.add_argument('--timesteps', type=int, default=None, help='Total timesteps for training (overall target)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (default: config others.random_seed; random if that is null)')
    parser.add_argument('--n_envs', type=int, default=1, help='Parallel sims feeding one learner (each on its own ROS domain)')
    parser.add_argument('--domain_base', type=int, default=None, help='ROS_DOMAIN_ID of sim 0 (default: $ROS_DOMAIN_ID)')
    parser.add_argument('--gradient_monitor_interval', type=int, default=100, help='Interval for gradient monitoring (default: 100)')
    args = parser.parse_args()

    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config)
    config = load_config(config_path)

    # Set random seed
    random_seed = args.seed if args.seed is not None else config['others'].get('random_seed')
    if random_seed is None:
        random_seed = int.from_bytes(os.urandom(4), 'little') % 100000
    print(f"Random seed: {random_seed}")
    np.random.seed(random_seed)
    th.manual_seed(random_seed)

    # Create environment
    env = make_env(args.n_envs if args.mode == 'train' else 1, args.domain_base)

    # Set up logging directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume_from_checkpoint:
        log_dir_base_name = os.path.splitext(os.path.basename(args.resume_from_checkpoint))[0]
        log_dir = os.path.join("logs", f"ppo_resumed_{log_dir_base_name}_{timestamp}")
    else:
        run_tag = "".join(f"_{k.split('_', 1)[1].lower()}-{os.environ[k]}" for k in
                          ("RACE2_VEHICLE", "RACE2_STAGE", "RACE2_SURGE_REWARD", "RACE2_ACTION", "RACE2_SERVOS", "RACE2_SERVO_LIMIT_DEG",
                           "RACE2_PAPER_TERMS", "RACE2_DEPTH_INV", "RACE2_ATT_SP", "RACE2_K_SLEW", "RACE2_REF_ALL", "RACE2_ATT_REP", "RACE2_INTEGRAL", "RACE2_ROLL_MAX_DEG",
                           "RACE2_PITCH_MAX_DEG", "RACE2_DEPTH_MAX") if k in os.environ) + f"_seed-{random_seed}"
        log_dir = os.path.join("logs", f"ppo_{timestamp}{run_tag}")
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to: {log_dir}")

    if args.mode == 'train':
        # Configure PPO hyperparameters from config
        learning_rate = config['agent']['learning_rate']
        n_steps = config['agent']['n_steps']
        if args.n_envs > 1:  # keep the rollout size (n_steps * n_envs) of the single-env config
            n_steps = max(64, n_steps // args.n_envs)
            print(f"n_envs={args.n_envs}: n_steps per env -> {n_steps} (rollout {n_steps * args.n_envs})")
        batch_size = config['agent']['batch_size']
        n_epochs = config['agent']['n_epochs']
        gamma = config['agent']['gamma']
        gae_lambda = config['agent']['gae_lambda']
        clip_range = config['agent']['clip_range']
        clip_range_vf = config['agent'].get('clip_range_vf', None)
        ent_coef = config['agent']['ent_coef']
        vf_coef = config['agent']['vf_coef']
        max_grad_norm = config['agent']['max_grad_norm']
        
        # Configure SGD optimizer
        optimizer_config = config['agent'].get('optimizer', {})
        momentum = optimizer_config.get('momentum', 0.9)
        weight_decay = optimizer_config.get('weight_decay', 0.0)
        nesterov = optimizer_config.get('nesterov', False)
        
        policy_kwargs = {
            "net_arch": config['qnetwork']['actor_hidden_layers'],
            "optimizer_class": optim.SGD,
            "optimizer_kwargs": {
                "momentum": momentum,
                "weight_decay": weight_decay,
                "nesterov": nesterov
            }
        }

        if args.resume_from_checkpoint:
            print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
            # Note: When loading from checkpoint, optimizer settings are preserved from the saved model
            model = PPO.load(
                args.resume_from_checkpoint,
                env=env,
                tensorboard_log=log_dir,
            )
            print(f"Model loaded. Current timesteps: {model.num_timesteps}. Training will continue up to the target total_timesteps.")
            print(f"Original learning rate (from loaded model): {model.learning_rate}")
            print("Note: Optimizer settings (SGD, momentum, etc.) are preserved from the checkpoint.")

        else:
            print("Starting new training session.")
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                clip_range_vf=clip_range_vf,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                seed=random_seed,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=log_dir
            )
            print(f"Learning rate: {learning_rate}, n_steps: {n_steps}, batch_size: {batch_size}")
            print(f"n_epochs: {n_epochs}, clip_range: {clip_range}")
            print(f"Optimizer: SGD (momentum: {momentum}, weight_decay: {weight_decay}, nesterov: {nesterov})")
            print(f"Actor network: {config['qnetwork']['actor_hidden_layers']}")

        # Configure custom logger
        new_logger = configure(log_dir, ["stdout", "csv", "tensorboard"])
        model.set_logger(new_logger)

        # Set up callbacks
        max_episode_steps = config['training']['max_t']
        checkpoint_callback = CheckpointCallback(
            save_freq=max(config['training']['save_freq_timesteps'] // max(args.n_envs, 1), 1),  # in vec-env calls
            save_path=checkpoint_dir(log_dir),   # on /scratch (vec_env_utils)
            name_prefix="ppo_auv_model"
        )

        plot_callback = RewardPlottingCallback(
            plot_interval=config['plotting']['plot_interval'],
            verbose=1
        )
        
        # Add gradient monitoring callback
        gradient_callback = GradientMonitoringCallback(
            monitor_interval=args.gradient_monitor_interval,
            verbose=1
        )

        # Determine total timesteps
        if args.timesteps:
            total_timesteps = args.timesteps
        else:
            total_timesteps = config['training']['max_episodes'] * max_episode_steps

        remaining_timesteps = total_timesteps - model.num_timesteps
        if remaining_timesteps <= 0:
            print(f"Model already trained for {model.num_timesteps} timesteps. Target total_timesteps {total_timesteps} already met or exceeded.")
            print("If you want to train further, increase --timesteps or config['training']['max_episodes'].")
        else:
            print(f"Starting PPO training. Current timesteps: {model.num_timesteps}. Target timesteps: {total_timesteps}. Remaining: {remaining_timesteps}")
            print(f"Gradient monitoring interval: {args.gradient_monitor_interval} steps")
            try:
                model.learn(
                    total_timesteps=total_timesteps,
                    callback=[TrackingLogger(), checkpoint_callback, plot_callback, gradient_callback],
                    log_interval=1,
                    reset_num_timesteps=False
                )

                final_model_path = os.path.join(log_dir, "final_model")
                model.save(final_model_path)
                print(f"Training completed. Final model saved to {final_model_path}. Total timesteps: {model.num_timesteps}")

                # Print final gradient monitoring summary
                print(f"\n--- Final Gradient Monitoring Summary ---")
                print("Total Updates by Network:")
                for network, count in gradient_callback.update_counts.items():
                    print(f"  {network}: {count} updates")

            except KeyboardInterrupt:
                print("Training interrupted. Saving current model...")
                interrupted_model_path = os.path.join(log_dir, "interrupted_model")
                model.save(interrupted_model_path)
                print(f"Interrupted model saved to {interrupted_model_path}. Total timesteps: {model.num_timesteps}")

    elif args.mode == 'test':
        if args.model is None:
            if args.resume_from_checkpoint:
                potential_log_dir = os.path.dirname(os.path.dirname(args.resume_from_checkpoint))
                final_model_path = os.path.join(potential_log_dir, "final_model.zip")
                interrupted_model_path = os.path.join(potential_log_dir, "interrupted_model.zip")
                if os.path.exists(final_model_path):
                    model_path_to_test = final_model_path
                elif os.path.exists(interrupted_model_path):
                    model_path_to_test = interrupted_model_path
                else:
                    model_path_to_test = args.resume_from_checkpoint
                print(f"Testing with model: {model_path_to_test} (derived from resume_from_checkpoint)")
            else:
                raise ValueError("Model path (--model) must be provided for testing mode if not resuming.")
        else:
            model_path_to_test = args.model

        print(f"Loading model from {model_path_to_test}")
        model = PPO.load(model_path_to_test, env=env)

        # Test the model
        print("Starting evaluation...")
        test_episodes = config['evaluation']['num_episodes']
        episode_rewards = []
        episode_lengths = []
        max_episode_steps = config['training']['max_t']

        for episode in range(test_episodes):
            obs, info = env.reset()
            episode_reward = 0
            terminated = False
            truncated = False
            step = 0

            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                step += 1
                obs = next_obs

                if step % 50 == 0 and config['evaluation']['verbose']:
                    print(f"Episode {episode+1}/{test_episodes}, Step {step}, Action: {action}, Reward: {reward:.4f}")
                if step >= max_episode_steps:
                    print(f"Warning: Episode {episode+1} reached max_episode_steps ({max_episode_steps}) set in test script.")
                    truncated = True

            episode_rewards.append(episode_reward)
            episode_lengths.append(step)
            print(f"Episode {episode+1} completed. Reward: {episode_reward:.4f}, Length: {step}")

        # Plot evaluation results
        eval_plot_path = os.path.join(log_dir, "evaluation_results.png")
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, test_episodes+1), episode_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(f'Evaluation Results ({os.path.basename(model_path_to_test)})')
        plt.grid(True, axis='y')
        plt.savefig(eval_plot_path)
        print(f"Evaluation plot saved to {eval_plot_path}")
        plt.close()

        avg_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        avg_length = np.mean(episode_lengths)
        print(f"\nEvaluation Summary ({os.path.basename(model_path_to_test)}):")
        print(f"Number of episodes: {test_episodes}")
        print(f"Average Reward: {avg_reward:.4f} ± {std_reward:.4f}")
        print(f"Average Length: {avg_length:.2f} steps")

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