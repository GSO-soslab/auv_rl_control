#!/bin/bash
# Fig-4 eval of any trained runs, in parallel, one headless sim per ROS domain.
# Each model is evaluated with the RACE2_* settings it was trained with (read from run_info/job.json),
# so observation/action layout always matches the network.
#   infra/run_fig4_runs.sh "0.15 0.5" sb3/logs/ppo_..._seed-1 sb3/logs/ppo_..._seed-2 ...
# Env: DOMAIN_BASE (default: a free per-node block from infra/domain_lock.sh); PROTOCOL=artic|att90 runs that
# protocol once per model (SURGES ignored).
# Output: <run>/fig4_v<surge>/ or <run>/<protocol>/ (fig4_steps.json, csv, metrics, fig4.png)
SURGES=$1; shift
[ "${PROTOCOL:-fig4}" != fig4 ] && SURGES=$PROTOCOL
R=$HOME/ros2_ws/race2.sh
n_evals=$(( $# * $(echo $SURGES | wc -w) ))
if [ -z "${DOMAIN_BASE:-}" ]; then
  dom=$("$(dirname "$0")/domain_lock.sh" acquire $n_evals $$) || exit 1
  trap '"$(dirname "$0")/domain_lock.sh" release $$' EXIT
else dom=$DOMAIN_BASE; fi
echo "ROS domains $dom-$((dom + n_evals - 1))"
for run in "$@"; do
  run=$(readlink -f "$run")
  model=$(ls "$run"/final_model.zip "$run"/final_model.pt 2>/dev/null | head -1)   # SB3 .zip, Dreamer .pt
  [ -n "$model" ] || { echo "skip $run (no final_model)"; continue; }
  algo=$(basename "$run" | cut -d_ -f1)
  envs=$(python3 -c "import json,sys; print(' '.join(f'{k}={v}' for k,v in json.load(open(sys.argv[1]))['settings'].items()))" \
         "$run/run_info/job.json" 2>/dev/null)
  for s in $SURGES; do
    if [ "$s" = "${PROTOCOL:-fig4}" ]; then tag=$s; pargs="--protocol $s"; else tag=fig4_v${s/./}; pargs="--surge $s"; fi
    ( export ROS_DOMAIN_ID=$dom
      [ -n "$envs" ] && export $envs   # the sim too: RACE2_VEHICLE picks the vehicle, RACE2_SERVO_LIMIT_DEG its joints
      ns=$(cd "$(dirname "$R")" && cd ~/race2_q_learning_control/sb3 && python3 -c "from vehicles import get_vehicle; print(get_vehicle()[1]['ns'])")
      setsid $R sim --rl --nogpu > "$run/${tag}_sim.log" 2>&1 &
      SIMPG=$!
      for k in $(seq 1 60); do $R exec "timeout 4 ros2 topic echo --once /$ns/odometry/ground_truth" >/dev/null 2>&1 && break; sleep 3; done
      $R exec "cd ~/race2_q_learning_control/sb3 && env $envs python3 -u eval_fig4.py $model --algo $algo $pargs --out $run/$tag" \
        > "$run/${tag}.log" 2>&1
      echo "done $(basename "$run") $tag (exit $?)"
      kill -INT -- -$SIMPG 2>/dev/null; sleep 8; kill -KILL -- -$SIMPG 2>/dev/null ) &
    dom=$((dom + 1))
    sleep 3
  done
done
wait
