#!/bin/bash
# Fig-4 eval of the 5 stage-3 models in parallel, one headless sim per ROS domain.
SURGE=${1:-0.15}; TAG=fig4_v${SURGE/./}
L=$HOME/race2_q_learning_control/sb3/logs; R=$HOME/ros2_ws/race2.sh
declare -a RUNS=(
 "71|ppo|instant|ppo_20261027_164019_stage-3_surge_reward-instant"
 "72|ppo|horizon|ppo_20261027_164137_stage-3_surge_reward-horizon"
 "73|ppo|integral_obs|ppo_20261027_164206_stage-3_surge_reward-integral_obs"
 "74|ddpg|instant|ddpg_20261027_164315_stage-3_surge_reward-instant"
 "75|ddpg|horizon|ddpg_20261027_164351_stage-3_surge_reward-horizon")
for r in "${RUNS[@]}"; do
  IFS='|' read -r dom algo mode dir <<<"$r"
  ( export ROS_DOMAIN_ID=$dom
    setsid $R sim --rl --nogpu > $L/$dir/${TAG}_sim.log 2>&1 &
    SIMPG=$!
    for k in $(seq 1 60); do $R exec "timeout 4 ros2 topic echo --once /race2_auv/odometry/ground_truth" >/dev/null 2>&1 && break; sleep 3; done
    RACE2_SURGE_REWARD=$mode $R exec "cd ~/race2_q_learning_control/sb3 && python3 -u eval_fig4.py $L/$dir/final_model.zip --algo $algo --surge $SURGE --out $L/$dir/$TAG" > $L/$dir/${TAG}.log 2>&1
    kill -INT -- -$SIMPG 2>/dev/null; sleep 8; kill -KILL -- -$SIMPG 2>/dev/null ) &
  sleep 3
done
wait
