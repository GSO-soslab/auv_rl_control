#!/bin/bash
# N parallel (headless GPU-rendered sim + trainer) pairs on ONE B200, as a batch job.
#   sbatch ~/ros2_ws/train_batch.sbatch                                   # 6 pairs ddpg,ppo, 250k steps
#   PAIRS=8 TRAINERS=ddpg STEPS=250000 sbatch ~/ros2_ws/train_batch.sbatch
#   PAIRS=1 TRAINERS=ppo  STEPS=100000 sbatch ~/ros2_ws/train_batch.sbatch
# Sizing (measured): per pair ~1.6 GB GPU memory, ~1.5 CPU cores, ~3.1 env steps/s
# (250k steps ~= 22 h). Keep --cpus-per-task >= 2 * PAIRS.
#SBATCH --job-name=race2_rl
#SBATCH --partition=b200-batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out

PAIRS=${PAIRS:-6}
TRAINERS=${TRAINERS:-ddpg,ppo}
STEPS=${STEPS:-250000}

echo "job $SLURM_JOB_ID on $(hostname): $PAIRS pairs [$TRAINERS] x $STEPS steps"
nvidia-smi -L
exec "$HOME/ros2_ws/race2.sh" multi "$PAIRS" "$TRAINERS" -- --timesteps "$STEPS"
