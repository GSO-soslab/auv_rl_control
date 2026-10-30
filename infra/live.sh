#!/bin/bash
# Live demo: rendered Stonefish + a trained policy, setpoints from ~/ros2_ws/runs/live/setpoint.json.
#   infra/live.sh start <run_dir>     settings (RACE2_*) are read from <run_dir>/run_info/job.json
#   infra/live.sh stop                kills only processes on the live ROS domain (never batch jobs on this node)
# Edit the control file to command: {"depth": 3, "surge": 0.5, "yaw_deg": 0, "pitch_deg": 0, "roll_deg": 0}
DOM=${LIVE_DOMAIN:-95}
R=$HOME/ros2_ws/race2.sh
L=$HOME/ros2_ws/runs/live
mkdir -p "$L"

stop() {
  local pids p mine=" " a=$$
  while [ "$a" -gt 1 ]; do mine="$mine$a "; a=$(ps -o ppid= -p "$a" | tr -d ' '); done   # never kill our own callers
  for sig in INT KILL; do
    pids=$(for p in $(pgrep -u "$USER"); do [[ "$mine" == *" $p "* ]] && continue
           [[ "$mine" == *" $(ps -o pgid= -p $p | tr -d ' ') "* ]] && continue
           grep -qz "^ROS_DOMAIN_ID=$DOM$" /proc/$p/environ 2>/dev/null && echo $p; done)
    [ -z "$pids" ] && break
    for p in $pids; do kill -$sig -- -"$(ps -o pgid= -p $p | tr -d ' ')" 2>/dev/null; done
    sleep 6
  done
  echo "live session on domain $DOM stopped"
}

case "$1" in
  stop) stop ;;
  start)
    run=$(readlink -f "$2"); [ -f "$run/final_model.zip" ] || { echo "no final_model.zip in $2"; exit 1; }
    stop >/dev/null
    envs=$(python3 -c "import json,sys; print(' '.join(f'{k}={v}' for k,v in json.load(open(sys.argv[1]))['settings'].items()))" "$run/run_info/job.json")
    algo=$(basename "$run" | cut -d_ -f1)
    echo '{"depth": 3.0, "surge": 0.0, "yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}' > "$L/setpoint.json"
    export ROS_DOMAIN_ID=$DOM
    setsid "$R" sim --rl > "$L/sim.log" 2>&1 < /dev/null &
    for k in $(seq 1 40); do "$R" exec "timeout 4 ros2 topic echo --once /race2_auv/odometry/ground_truth" >/dev/null 2>&1 && break; sleep 3; done
    setsid "$R" exec "cd ~/race2_q_learning_control/sb3 && env $envs python3 -u live_policy.py $run/final_model.zip --algo $algo" \
      > "$L/policy.log" 2>&1 < /dev/null &
    echo "live: $(basename "$run") on domain $DOM | control $L/setpoint.json | log $L/policy.log" ;;
  *) echo "usage: $0 start <run_dir> | stop"; exit 1 ;;
esac
