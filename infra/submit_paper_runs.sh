#!/bin/bash
# Fresh paper training set (2026-10-30, after the per-node ROS domain fix) + its evaluations, as dependent jobs.
#   infra/submit_paper_runs.sh [groups...]     groups: fixed slew rauv (default all)  [DRY=1: print only]
#   ALGOS="td3" limits the PPO/TD3 part (and its eval job prefixes); DREAMER_{FIXED,SLEW,RAUV}_SEEDS="" skips Dreamer.
# Per group: PPO + TD3 at 3M steps x seeds 1-3, evaluations queued after them (afterany):
#   fixed = race2 fixed thrusters (v4 flags)          -> Fig-4 (0.15, 0.5) + checkpoint eval
#   slew  = race2 articulated + servo slew cost 0.02  -> Fig-4 + artic protocol + checkpoint eval
#   rauv  = race_auv, roll/pitch targets to 90 deg    -> Fig-4 + att90 + checkpoint eval
# Dreamer runs as resumable 10M chains (DREAMER_SEEDS per group, 4 parts, --run_id $DREAMER_RUN_ID); its
# evaluations are queued after the whole chain. Job names: <group>_<algo>_s<seed> (eval prefixes: fx_ sl_ ra_).
set -euo pipefail
cd "$(dirname "$0")/.."
V4="RACE2_STAGE=3 RACE2_SURGE_REWARD=horizon RACE2_PAPER_TERMS=1 RACE2_DEPTH_INV=1 RACE2_ATT_SP=1 RACE2_REF_ALL=1 RACE2_INTEGRAL=1 RACE2_ROLL_MAX_DEG=12 RACE2_PITCH_MAX_DEG=20 RACE2_DEPTH_MAX=12 RACE2_SERVO_LIMIT_DEG=90"
declare -A ENVS=(
  [fixed]="RACE2_VEHICLE=race2 $V4 RACE2_SERVOS=0 RACE2_K_SLEW=0"
  [slew]="RACE2_VEHICLE=race2 $V4 RACE2_SERVOS=2 RACE2_ACTION=rate RACE2_K_SLEW=0.02"
  [rauv]="RACE2_VEHICLE=race_auv RACE2_SERVOS=0 RACE2_STAGE=3 RACE2_SURGE_REWARD=horizon RACE2_PAPER_TERMS=1 RACE2_DEPTH_INV=1 RACE2_ATT_SP=1 RACE2_REF_ALL=1 RACE2_INTEGRAL=1 RACE2_ROLL_MAX_DEG=90 RACE2_PITCH_MAX_DEG=90 RACE2_DEPTH_MAX=12 RACE2_ATT_REP=rot"
)
declare -A PREFIX=([fixed]=fx [slew]=sl [rauv]=ra)
declare -A PROTO=([fixed]="" [slew]=artic [rauv]=att90)
DREAMER_RUN_ID=${DREAMER_RUN_ID:-20261030_10M}
declare -A DREAMER_SEEDS=([fixed]="${DREAMER_FIXED_SEEDS-2 3}" [slew]="${DREAMER_SLEW_SEEDS-1 2 3}" [rauv]="${DREAMER_RAUV_SEEDS-}")
SB="sbatch --parsable -p rtx-batch --exclude=a0001"
sub() { if [ -n "${DRY:-}" ]; then echo "DRY:$*" >&2; echo 0; else "$@"; fi; }

for grp in ${@:-fixed slew rauv}; do
  E=${ENVS[$grp]}; P=${PREFIX[$grp]}; ids=()
  JP=$(for a in ${ALGOS:-ppo td3}; do printf '%s_%s,' $P $a; done); JP=${JP%,}
  for a in ${ALGOS:-ppo td3}; do for s in 1 2 3; do
    ids+=($(sub env $E TRAINER=$a STEPS=3000000 EXTRA="--seed $s" $SB --cpus-per-task=32 --mem=96G --time=08:00:00 \
            --job-name=${P}_${a}_s$s ~/ros2_ws/train_vec.sbatch))
  done; done
  dep=afterany:$(IFS=:; echo "${ids[*]}")
  ev=$(sub env JOB_PREFIX=$JP SURGES="0.15 0.5" $SB --cpus-per-task=32 --mem=96G --time=01:30:00 --dependency=$dep \
       --job-name=${P}_eval_fig4 infra/eval_fig4.sbatch)
  [ -n "${PROTO[$grp]}" ] && ev+=" $(sub env JOB_PREFIX=$JP PROTOCOL=${PROTO[$grp]} SURGES=x $SB --cpus-per-task=32 \
       --mem=64G --time=01:30:00 --dependency=$dep --job-name=${P}_eval_${PROTO[$grp]} infra/eval_fig4.sbatch)"
  ev+=" $(sub env $E TRAINER=evalckpt SIMS=20 STEPS=0 EXTRA="--runs-from-jobs $JP --every 100000" $SB --cpus-per-task=32 \
       --mem=96G --time=08:00:00 --dependency=$dep --job-name=${P}_evalckpt ~/ros2_ws/train_vec.sbatch)"
  echo "$grp: train ${ids[*]} | evals $ev"
  for s in ${DREAMER_SEEDS[$grp]}; do
    A="--seed $s --run_id $DREAMER_RUN_ID --max_hours 7.4 --checkpoint_every 250000"; prev=""; chain=""
    for part in 1 2 3 4; do
      prev=$(sub env $E TRAINER=dreamer STEPS=10000000 EXTRA="$A" $SB --cpus-per-task=32 --mem=96G --time=08:00:00 \
             ${prev:+--dependency=afterany:$prev} --job-name=${P}_dreamer10m_s${s}_p$part ~/ros2_ws/train_vec.sbatch); chain+="$prev "
    done
    echo "  dreamer 10M s$s: $chain"
  done
done
