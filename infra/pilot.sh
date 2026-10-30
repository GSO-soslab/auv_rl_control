#!/bin/bash
# Pilot a trained policy live in Stonefish (GPU-rendered on your desktop) with the on-screen keyboard/mouse joystick.
#   infra/pilot.sh <run_dir> [model]       e.g. infra/pilot.sh sb3/logs/ppo_20261028_092221_..._seed-1
# Starts: the sim with rendering, sb3/joystick_pilot.py (setpoint window), sb3/live_policy.py (the policy).
# Algorithm = run dir prefix (ppo/td3/dreamer); env settings (vehicle, RACE2_*) from <run_dir>/run_info/job.json;
# model default = <run_dir>/final_model.{zip,pt}. Own ROS domain (PILOT_DOMAIN, default: a free per-node block from
# infra/domain_lock.sh, so it never shares a domain with a batch job on this node).
# Ctrl-C stops all three (only these processes; never `race2 stop`, which kills every sim of yours on the node).
# Status of the policy: tail -f ~/ros2_ws/runs/live/policy.log
set -euo pipefail
RUN=$(cd "${1:?usage: infra/pilot.sh <run_dir> [model]}" && pwd)
ALGO=$(basename "$RUN" | cut -d_ -f1)
MODEL=${2:-}
for m in "$RUN"/final_model.zip "$RUN"/final_model.pt; do if [ -z "$MODEL" ] && [ -f "$m" ]; then MODEL=$m; fi; done
[ -f "$MODEL" ] || { echo "no model in $RUN" >&2; exit 1; }
if [ ! -f "$RUN/run_info/job.json" ]; then echo "no $RUN/run_info/job.json: export the run's RACE2_* settings yourself" >&2
else eval "$(python3 -c "import json; print(' '.join(f'export {k}={v}' for k, v in json.load(open('$RUN/run_info/job.json'))['settings'].items()))")"; fi
DL="$(cd "$(dirname "$0")" && pwd)/domain_lock.sh"
export RACE2_VEHICLE=${RACE2_VEHICLE:-race2} ROS_DOMAIN_ID=${PILOT_DOMAIN:-$("$DL" acquire 1 $$)} DISPLAY=${DISPLAY:?needs a desktop display}
NS=$(cd "$(dirname "$0")/../sb3" && python3 -c "from vehicles import VEHICLES; print(VEHICLES['$RACE2_VEHICLE']['ns'])")
R2=~/ros2_ws/race2.sh
LIVE=~/ros2_ws/runs/live; CTL=$LIVE/setpoint.json; mkdir -p "$LIVE"
echo '{"depth": 2.0, "surge": 0.0, "yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0, "respawn": 0}' > "$CTL"

PGIDS=()
stop() { trap - INT TERM EXIT; "$DL" release $$ || true; for g in "${PGIDS[@]}"; do kill -INT -- "-$g" 2>/dev/null || true; done; sleep 3
         for g in "${PGIDS[@]}"; do kill -KILL -- "-$g" 2>/dev/null || true; done; echo "pilot stopped"; }
trap stop INT TERM EXIT

echo "$ALGO $(basename "$MODEL") | vehicle $RACE2_VEHICLE | ROS domain $ROS_DOMAIN_ID | logs $LIVE/"
setsid "$R2" sim --rl > "$LIVE/sim.log" 2>&1 & PGIDS+=($!)
for k in $(seq 1 60); do "$R2" exec "timeout 4 ros2 topic echo --once /$NS/odometry/ground_truth" >/dev/null 2>&1 && break; sleep 3; done
setsid "$R2" exec "cd ~/race2_q_learning_control/sb3 && python3 -u joystick_pilot.py --control $CTL" > "$LIVE/joystick.log" 2>&1 & PGIDS+=($!)
sleep 3
setsid "$R2" exec "cd ~/race2_q_learning_control/sb3 && python3 -u live_policy.py $MODEL --algo $ALGO --control $CTL" > "$LIVE/policy.log" 2>&1 & PGIDS+=($!)
echo "running (Ctrl-C to stop). Status: tail -f $LIVE/policy.log"
wait
