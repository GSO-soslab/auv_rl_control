#!/bin/bash
# IROS 2025 baseline (mvp_control QP + PID) on the evaluation protocols, one headless sim per case, in parallel.
#   LIMITS="45 90 135" OUT=results/<race2|race_auv>/<dir> sbatch -p rtx-batch --cpus-per-task=16 --mem=32G --time=01:00:00 infra/eval_mvp.sbatch
# Cases per servo limit: Fig-4 at +-0.15 and +-0.5 m/s, and the articulation protocol.
# Output: $OUT/lim<L>/{fig4_v015,fig4_v05,artic}/ (same files as eval_fig4.py)
#SBATCH --job-name=eval_mvp
#SBATCH --gpus=1
#SBATCH --output=%x-%j.out
cd "$HOME/race2_q_learning_control"
LIMITS=${LIMITS:-"45 90 135"}
OUT=$(readlink -f "${OUT:-results/race2/baseline_mvp}")
R=$HOME/ros2_ws/race2.sh
dom=${DOMAIN_BASE:-81}
START=$(date +%s)
for L in $LIMITS; do
  for c in "fig4_v015|--protocol fig4 --surge 0.15" "fig4_v05|--protocol fig4 --surge 0.5" "artic|--protocol artic"; do
    tag=${c%%|*}; pargs=${c#*|}; out=$OUT/lim$L/$tag; mkdir -p "$out"
    ( export ROS_DOMAIN_ID=$dom RACE2_SERVO_LIMIT_DEG=$L
      setsid $R sim --nogpu > "$out/sim.log" 2>&1 &
      SIMPG=$!
      for k in $(seq 1 60); do $R exec "timeout 4 ros2 topic echo --once /race2_auv/odometry/ground_truth" >/dev/null 2>&1 && break; sleep 3; done
      $R exec "cd ~/race2_q_learning_control/sb3 && python3 -u eval_mvp.py $pargs --out $out" > "$out/eval.log" 2>&1
      echo "done lim$L $tag (exit $?)"
      kill -INT -- -$SIMPG 2>/dev/null; sleep 8; kill -KILL -- -$SIMPG 2>/dev/null ) &
    dom=$((dom + 1)); sleep 3
  done
done
wait
echo "eval wall time: $(( $(date +%s) - START )) s"
