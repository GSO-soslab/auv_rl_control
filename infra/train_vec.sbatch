#!/bin/bash
# ONE learner (on the GPU) fed by many non-rendered Stonefish sims (on the CPUs).
# Number of sims follows the CPU request (~1.3 cores per sim incl. its ROS nodes):
#   RACE2_STAGE=3 RACE2_SURGE_REWARD=horizon TRAINER=ppo STEPS=1000000 \
#     sbatch -p rtx-batch --cpus-per-task=32 --mem=96G --time=02:30:00 --job-name=NAME ~/ros2_ws/train_vec.sbatch
#   TRAINER=ddpg EXTRA="--buffer_size 1000000 --seed 1" sbatch ...
# Measured (rtx-batch, 32 CPUs -> 23 sims, 10 Hz): ~205 env steps/s; 1M steps ~ 80 min.
#
# Every job records its run info (for the paper: time + hardware) in the trainer's log dir, under run_info/:
#   config/         snapshot of sb3/config at launch
#   job.json        job id/name, node, partition, CPUs, memory, GPU model, RACE2_* settings, git commits,
#                   start/end time, wall time, trainer/steps/sims, log dir
#   gpu_usage.csv   nvidia-smi samples every 30 s (utilization %, memory MiB, power W)
#   cpu_load.csv    CPU cores in use + resident memory (this user's processes) every 30 s
# After the job, `python3 infra/collect_run_stats.py` adds Slurm accounting (CPU time, MaxRSS) and builds a table.
#SBATCH --job-name=race2_vec
#SBATCH --partition=b200-batch
#SBATCH --gpus=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out

# Refuse to start when home is nearly full (cap 100 GiB, invisible to `quota`): on 2026-10-30 a full home froze every
# running job and blocked logins. Bulky outputs go to /scratch (checkpoints, ROS logs).
HOME_GB=$(timeout 300 du -s --block-size=1G "$HOME" 2>/dev/null | cut -f1)
if [ -n "$HOME_GB" ] && [ "$HOME_GB" -gt "${HOME_MAX_GB:-85}" ]; then
  echo "home uses ${HOME_GB} GB (> ${HOME_MAX_GB:-85}): not starting. Free space first (sb3/logs, ~/.cache, ~/ros2_ws/runs)." >&2
  exit 1
fi

TRAINER=${TRAINER:-ppo}
STEPS=${STEPS:-250000}
SIMS=${SIMS:-$(( SLURM_CPUS_PER_TASK * 10 / 13 - 1 ))}
export RENDER_SIM0=${RENDER_SIM0:-0}   # nobody can watch it in batch

INFO=$(mktemp -d "$HOME/ros2_ws/runs/job_${SLURM_JOB_ID}_XXXX")
cp -r "$HOME/race2_q_learning_control/sb3/config" "$INFO/config"   # config snapshot at launch (reproducibility)
START=$(date +%s)
echo "job $SLURM_JOB_ID on $(hostname): $TRAINER, $SIMS sims, $STEPS steps (run info: $INFO)"
nvidia-smi -L

# Resource monitors (every 30 s)
( echo "time_s,gpu_util_pct,gpu_mem_mib,gpu_power_w"
  while true; do
    echo "$(( $(date +%s) - START )),$(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits -i 0 | tr -d ' ')"
    sleep 30
  done ) > "$INFO/gpu_usage.csv" &
GPU_MON=$!
( echo "time_s,cpu_cores_used,rss_mib"   # this user's processes only (nodes are shared)
  while true; do
    # 2nd top sample = CPU over the last 2 s (the 1st sample is unreliable)
    echo "$(( $(date +%s) - START )),$(top -b -n2 -d2 -u "$USER" | awk '/^top -/{n++} n==2 && $1 ~ /^[0-9]+$/ {s+=$9} END{printf "%.1f", s/100}'),$(ps -u "$USER" -o rss= | awk '{s+=$1} END{printf "%d", s/1024}')"
    sleep 28
  done ) > "$INFO/cpu_load.csv" &
CPU_MON=$!

"$HOME/ros2_ws/race2.sh" vec "$SIMS" "$TRAINER" -- --timesteps "$STEPS" $EXTRA
STATUS=$?
END=$(date +%s)
kill $GPU_MON $CPU_MON 2>/dev/null

LOGDIR=$(grep -m1 -oE "Logging to:? logs/[^ ]+" "$SLURM_SUBMIT_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" 2>/dev/null | awk '{print $NF}')
python3 - "$INFO/job.json" <<EOF
import json, os, subprocess, sys
def sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
    except Exception: return ''
info = {
  'job_id': os.environ.get('SLURM_JOB_ID'), 'job_name': os.environ.get('SLURM_JOB_NAME'),
  'node': os.uname().nodename, 'partition': os.environ.get('SLURM_JOB_PARTITION'),
  'cpus_allocated': int(os.environ.get('SLURM_CPUS_PER_TASK', 0)), 'mem_allocated_mb': os.environ.get('SLURM_MEM_PER_NODE'),
  'gpu': sh('nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader -i 0'),
  'cpu_model': sh("lscpu | sed -n 's/^Model name: *//p'"),
  'trainer': '$TRAINER', 'steps_requested': $STEPS, 'sims': $SIMS, 'extra_args': '''$EXTRA''',
  'settings': {k: v for k, v in os.environ.items() if k.startswith('RACE2_')},
  'start_unix': $START, 'end_unix': $END, 'wall_time_s': $END - $START, 'exit_status': $STATUS,
  'log_dir': '$LOGDIR',
  'git': {r: sh(f'git -C {p} rev-parse --short HEAD') + (' (dirty)' if sh(f'git -C {p} status --porcelain') else '')
          for r, p in [('race2_q_learning_control', '$HOME/race2_q_learning_control'),
                       ('race2_auv', '$HOME/ros2_ws/src/race2_auv'), ('world_of_stonefish', '$HOME/ros2_ws/src/world_of_stonefish'),
                       ('mvp_control', '$HOME/ros2_ws/src/mvp_control')]},
}
json.dump(info, open(sys.argv[1], 'w'), indent=2)
EOF
# move run info next to the model
if [ -n "$LOGDIR" ] && [ -d "$HOME/race2_q_learning_control/sb3/$LOGDIR" ]; then
  mv "$INFO" "$HOME/race2_q_learning_control/sb3/$LOGDIR/run_info"
  echo "run info -> sb3/$LOGDIR/run_info"
else
  echo "run info -> $INFO (log dir not found)"
fi
exit $STATUS
