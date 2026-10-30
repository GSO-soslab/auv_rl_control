#!/bin/bash
# Fig-4 eval of every run whose job name starts with $JOB_PREFIX (comma-separated list; from run_info/job.json), at $SURGES.
# PROTOCOL=artic|att90 runs that protocol instead (once per model).
#   JOB_PREFIX=art90_ SURGES="0.15 0.5" sbatch -p rtx-batch --cpus-per-task=32 --mem=64G --time=01:30:00 \
#     --dependency=afterany:<training job ids> infra/eval_fig4.sbatch
#SBATCH --job-name=fig4_eval
#SBATCH --gpus=1
#SBATCH --output=%x-%j.out

# Refuse to start when home is nearly full (cap 100 GiB, invisible to `quota`): on 2026-10-30 a full home froze every
# running job and blocked logins. Bulky outputs go to /scratch (checkpoints, ROS logs).
HOME_GB=$(timeout 300 du -s --block-size=1G "$HOME" 2>/dev/null | cut -f1)
if [ -n "$HOME_GB" ] && [ "$HOME_GB" -gt "${HOME_MAX_GB:-85}" ]; then
  echo "home uses ${HOME_GB} GB (> ${HOME_MAX_GB:-85}): not starting. Free space first (sb3/logs, ~/.cache, ~/ros2_ws/runs)." >&2
  exit 1
fi
cd "$HOME/race2_q_learning_control"
RUNS=$(python3 - <<PY
import glob, json, os
for f in sorted(glob.glob('sb3/logs/*/run_info/job.json')):
    j = json.load(open(f))
    if (j.get('job_name') or '').startswith(tuple('$JOB_PREFIX'.split(','))) and j.get('exit_status') == 0:
        print(os.path.dirname(os.path.dirname(f)))
PY
)
echo "evaluating at surge ${SURGES}:"; echo "$RUNS"
START=$(date +%s)
infra/run_fig4_runs.sh "$SURGES" $RUNS
echo "eval wall time: $(( $(date +%s) - START )) s"
