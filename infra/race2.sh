#!/usr/bin/env bash
# race2.sh - run ROS 2 Jazzy + Stonefish + race2_auv + RL training inside Apptainer (no sudo needed).
#
#   race2 setup          clone all repos (if missing) + build image (if missing) + colcon build
#   race2 build-image    (re)build the Apptainer image from container/ros2_jazzy_stonefish.def
#   race2 build [args]   colcon build the workspace (extra args go to colcon)
#   race2 sim [--rl] [--headless|--nogpu] [--cpu]  Stonefish (GPU via VirtualGL) + vehicle bringup
#                        vehicle from RACE2_VEHICLE: race2 (default, race2_auv) | race_auv (the real RACE AUV)
#                        --nogpu: no rendering at all (~20x cheaper; no camera)
#                        --rl: mvp_control no longer drives the thrusters (use for RL training)
#   race2 ddpg|ppo|sac [args]  online SB3 trainer in sb3/ (needs `race2 sim --rl` running)
#   race2 train [args]   = sac (also offline: --csv_directory DIR)
#   race2 vec N ddpg|ppo [-- args]      ONE learner fed by N headless sims (fast training)
#   race2 multi N [ddpg,ppo] [-- args]  N independent sim+trainer pairs (seed/algo sweeps)
#   race2 shell          interactive bash inside the container, ROS sourced
#   race2 exec CMD...    run one command inside the container, ROS sourced
#   race2 stop           kill leftover sim/ROS processes of yours on this node
#   race2 check          quick health check (GPU, GL, ROS topics)
set -euo pipefail

WS="${RACE2_WS:-$HOME/ros2_ws}"
RL_REPO="${RACE2_RL_REPO:-$HOME/race2_q_learning_control}"
SIF="$WS/container/ros2_jazzy_stonefish.sif"
DEF="$WS/container/ros2_jazzy_stonefish.def"

# Keep ROS traffic on this node and away from other users sharing it.
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-$(( $(id -u) % 100 + 1 ))}"
# ROS logs (one dir per sim launch) off the capped home: /scratch when it exists
[ -d "/scratch/$USER" ] && export ROS_LOG_DIR="${ROS_LOG_DIR:-/scratch/$USER/ros_log}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-/scratch/$USER/.apptainer}"

# repo|branch-or-commit|dest (relative to $WS/src)
REPOS=(
  "https://github.com/GSO-soslab/race2_auv|jazzy-devel|race2_auv"
  "https://github.com/GSO-soslab/stonefish_ros2|jazzy-devel|stonefish_ros2"
  # docking-station branch: has race_auv.scn (the real vehicle); race2 scenes unchanged (checked 2026-10-29).
  # Local patch: infra/patches/world_of_stonefish_ground_truth_odometry.patch
  "https://github.com/GSO-soslab/world_of_stonefish|jazzy-devel-race-docking-station|world_of_stonefish"
  # The real RACE AUV (submodules perception/sim/factorgraph are not needed for RL and not built)
  "https://github.com/GSO-soslab/race_auv|jazzy-devel-perception|race_auv"
  "https://github.com/GSO-soslab/acomms_msgs|jazzy-devel|acomms_msgs"
  "https://github.com/uri-ocean-robotics/mvp_msgs|jazzy-devel|mvp_msgs"
  "https://github.com/uri-ocean-robotics/mvp_utilities|jazzy-devel|mvp_utilities"
  # jazzy-devel (6cfea2d on 2026-10-28): needs 'v' gains + 'delta_limit' in the vehicle config
  # (race_auv has them; race2_auv via infra/patches/race2_auv_local.patch)
  "https://github.com/uri-ocean-robotics/mvp_control|jazzy-devel|mvp_control"
)
PKGS=(race2_auv_bringup race2_auv_config race2_auv_description world_of_stonefish
      race_auv_bringup race_auv_config race_auv_description
      stonefish_ros2 mvp_control mvp_localization_utilities)

# Vehicle (RACE2_VEHICLE, same names as sb3/vehicles.py): ROS namespace + simulation launch
case "${RACE2_VEHICLE:-race2}" in
  race2)    VEH_NS=race2_auv VEH_LAUNCH="race2_auv_bringup bringup_simulation.launch.py" ;;
  race_auv) VEH_NS=race_auv  VEH_LAUNCH="$RL_REPO/infra/launch/race_auv_rl_sim.launch.py" ;;
  *) echo "unknown RACE2_VEHICLE=${RACE2_VEHICLE} (race2 | race_auv)" >&2; exit 1 ;;
esac
export VEH_NS VEH_LAUNCH

log() { echo -e "\033[1;34m[race2]\033[0m $*"; }

in_container() {
  [[ -f "$SIF" ]] || { echo "Image missing: $SIF  -> run: race2 build-image" >&2; exit 1; }
  local binds="" ; [[ -d /scratch ]] && binds="--bind /scratch"
  apptainer exec --nv $binds "$SIF" bash -c '
    source /opt/ros/jazzy/setup.bash
    [[ -f "'"$WS"'/install/setup.bash" ]] && source "'"$WS"'/install/setup.bash"
    '"$1"
}

clone_repos() {
  mkdir -p "$WS/src"
  for entry in "${REPOS[@]}"; do
    IFS='|' read -r url ref dest <<<"$entry"
    if [[ -d "$WS/src/$dest/.git" ]]; then log "have $dest"; continue; fi
    log "cloning $dest ($ref)"
    GIT_TERMINAL_PROMPT=0 git clone -q "$url" "$WS/src/$dest"
    git -C "$WS/src/$dest" checkout -q "$ref"
  done
  [[ -d "$RL_REPO/.git" ]] || GIT_TERMINAL_PROMPT=0 git clone -q -b jazzy-devel \
      https://github.com/GSO-soslab/race2_q_learning_control "$RL_REPO"
}

build_image() {
  mkdir -p "$APPTAINER_CACHEDIR" "/tmp/$USER-apptmp"
  log "building image (~15 min) -> $SIF"
  (cd "$WS/container" && APPTAINER_TMPDIR="/tmp/$USER-apptmp" \
     apptainer build --fakeroot "$SIF.new" "$DEF" && mv -f "$SIF.new" "$SIF")
}

colcon_build() {
  log "colcon build in $WS"
  in_container "cd '$WS' && colcon build --symlink-install \
      --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-up-to ${PKGS[*]} $*"
}

run_sim() {
  # GPU rendering: VirtualGL redirects Stonefish's GLX context to the NVIDIA EGL device
  # (VGL_DEVICE=egl0, egl1, ... to pick another GPU; --cpu falls back to Mesa llvmpipe).
  local pre="vglrun -d ${VGL_DEVICE:-egl}" headless=0 largs="" nogpu=0
  for a in "$@"; do
    case "$a" in --headless) headless=1 ;; --cpu) pre="" ;; --rl) largs+=" rl_mode:=true" ;;
                 --nogpu) nogpu=1 pre="" largs+=" render:=false" ;; esac
  done
  # Servo angle limit for the sim joint and mvp's QP (same as the RL action clip), e.g. RACE2_SERVO_LIMIT_DEG=90
  [ -n "${RACE2_SERVO_LIMIT_DEG:-}" ] && [ "$VEH_NS" = race2_auv ] && largs+=" servo_limit_deg:=$RACE2_SERVO_LIMIT_DEG"
  if [[ $nogpu == 0 ]] && [[ $headless == 1 || -z "${DISPLAY:-}" ]]; then
    log "headless -> virtual X display via xvfb-run"
    pre='xvfb-run -a -s "-screen 0 1280x1024x24" '"$pre"
  fi
  log "ROS_DOMAIN_ID=$ROS_DOMAIN_ID  DISPLAY=${DISPLAY:-xvfb}  render=$( ((nogpu)) && echo none || echo "${pre:-cpu}")  ${largs}"
  log "vehicle ${RACE2_VEHICLE:-race2}: $VEH_LAUNCH"
  in_container "$pre ros2 launch --noninteractive $VEH_LAUNCH $largs"
}

# N independent (headless sim + trainer) pairs, each on its own ROS_DOMAIN_ID, sharing the GPU.
# Trainers cycle through a comma list, e.g. `race2 multi 4 ddpg,ppo -- --timesteps 200000`.
GPU_MB_PER_PAIR="${GPU_MB_PER_PAIR:-2000}"   # measured: stonefish ~670 MiB + trainer ~920 MiB
CPUS_PER_PAIR="${CPUS_PER_PAIR:-2}"    # measured: stonefish ~1 core + trainer/nodes ~0.5
GPU_MAX_USED_PCT="${GPU_MAX_USED_PCT:-90}"   # watchdog stops the newest pair above this

gpu_mem() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i 0 | tr -d ' '; }

run_multi() {
  local n="${1:?usage: race2 multi N [ddpg,ppo,sac] [-- trainer args]}"; shift
  local trainers="ddpg"; [[ $# -gt 0 && "$1" != "--" ]] && { trainers="$1"; shift; }
  [[ "${1:-}" == "--" ]] && shift
  local targs="$*"

  IFS=, read -r used total <<<"$(gpu_mem)"
  local max_gpu=$(( (total * GPU_MAX_USED_PCT / 100 - used) / GPU_MB_PER_PAIR ))
  local max_cpu=$(( $(nproc) / CPUS_PER_PAIR ))
  local max=$(( max_gpu < max_cpu ? max_gpu : max_cpu ))
  log "GPU ${used}/${total} MiB used -> room for $max_gpu pairs; $(nproc) CPUs -> $max_cpu pairs"
  if (( n > max )) && [[ -z "${FORCE:-}" ]]; then
    echo "Refusing $n pairs (limit $max). Lower N, or FORCE=1 to override." >&2; exit 1
  fi

  local out="$WS/runs/multi_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$out"
  local base=$ROS_DOMAIN_ID pgids=() i
  stop_all() {
    trap - INT TERM EXIT
    log "stopping ${#pgids[@]} pairs"
    for g in "${pgids[@]}"; do kill -INT -- "-$g" || true; kill -TERM -- "-$g" || true; done 2>/dev/null; sleep 10
    for g in "${pgids[@]}"; do kill -KILL -- "-$g" 2>/dev/null || true; done; true
  }
  trap stop_all INT TERM EXIT

  IFS=, read -ra tl <<<"$trainers"
  for (( i = 0; i < n; i++ )); do
    local dom=$(( 1 + (base - 1 + i) % 101 )) tr="${tl[i % ${#tl[@]}]}"
    # Each pair in its own session -> one process group we can signal as a unit.
    ROS_DOMAIN_ID=$dom setsid bash -c "
      '$0' sim --rl --headless > '$out/pair${i}_sim.log' 2>&1 &
      for k in \$(seq 1 60); do
        '$0' exec 'timeout 5 ros2 topic echo --once /$VEH_NS/odometry/filtered' >/dev/null 2>&1 && break; sleep 3
      done
      '$0' $tr $targs > '$out/pair${i}_${tr}.log' 2>&1
      kill -TERM -- -\$\$" &
    pgids+=($!)
    log "pair $i: $tr  ROS_DOMAIN_ID=$dom  logs: $out/pair${i}_*.log"
    sleep 20 & wait $!   # stagger (interruptible): sim startup + distinct log-dir timestamps
  done

  log "all $n pairs up. Ctrl-C stops everything. GPU watchdog at ${GPU_MAX_USED_PCT}%."
  while (( ${#pgids[@]} )); do
    sleep 30 & wait $!   # interruptible so Ctrl-C/TERM stop at once
    local alive=() g; for g in "${pgids[@]}"; do kill -0 -- "-$g" 2>/dev/null && alive+=("$g"); done
    pgids=("${alive[@]}")
    IFS=, read -r used total <<<"$(gpu_mem)"
    if (( used * 100 > total * GPU_MAX_USED_PCT && ${#pgids[@]} )); then
      g="${pgids[-1]}"; log "GPU at ${used}/${total} MiB -> stopping newest pair (pgid $g)"
      kill -INT -- "-$g"; kill -TERM -- "-$g"; sleep 10; kill -KILL -- "-$g" 2>/dev/null
      unset 'pgids[-1]'
    fi
  done
  log "all pairs finished"
}

# ONE learner fed by N headless sims (SB3 SubprocVecEnv, sim i on ROS domain base+i).
#   race2 vec 20 ppo -- --timesteps 250000
run_vec() {
  local n="${1:?usage: race2 vec N ddpg|ppo|td3|dreamer [-- trainer args]}" tr="${2:?trainer: ddpg|ppo|td3|dreamer}"; shift 2
  [[ "${1:-}" == "--" ]] && shift
  local targs="$*" base
  # own block of ROS domains on this node (jobs sharing a node used to share domains 53.. and cross-talk)
  base=$("$RL_REPO/infra/domain_lock.sh" acquire "$n" $$) || exit 1
  log "ROS domains $base-$((base + n - 1)) (per-node lock: infra/domain_lock.sh status)"
  local max_cpu=$(( $(nproc) * 10 / 13 - 1 ))   # ~1.3 cores per sim + learner
  log "$n sims, $(nproc) CPUs -> CPU room for ~$max_cpu sims"
  if (( n > max_cpu )) && [[ -z "${FORCE:-}" ]]; then
    echo "Refusing $n sims (limit $max_cpu). Lower N, request more CPUs, or FORCE=1." >&2; exit 1
  fi
  local out="$WS/runs/vec_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$out"
  # (set -e also applies inside traps: a kill on an already-exited group must not turn into exit status 1)
  stop_vec() { trap - INT TERM EXIT; "$RL_REPO/infra/domain_lock.sh" release $$ || true; [ -n "${VEC_PG:-}" ] || return 0
    kill -0 -- "-$VEC_PG" 2>/dev/null || return 0   # session already ended by itself (normal finish)
    log "stopping"; kill -INT -- "-$VEC_PG" 2>/dev/null || true; kill -TERM -- "-$VEC_PG" 2>/dev/null || true
    sleep 10; kill -KILL -- "-$VEC_PG" 2>/dev/null || true; }
  trap stop_vec INT TERM EXIT
  # Everything in one session -> one process group to stop.
  setsid bash -c "
    for (( i = 0; i < $n; i++ )); do
      d=\$(( 1 + ($base - 1 + i) % 101 ))
      # sim 0 renders on the GPU (watch it on \$DISPLAY if set); the rest run without rendering
      if (( i == 0 && ${RENDER_SIM0:-1} )); then mode=''; else mode='--nogpu'; fi
      ROS_DOMAIN_ID=\$d '$0' sim --rl \$mode > \"$out/sim\$i.log\" 2>&1 &
      sleep 2
    done
    for (( i = 0; i < $n; i++ )); do
      d=\$(( 1 + ($base - 1 + i) % 101 ))
      for k in \$(seq 1 120); do
        ROS_DOMAIN_ID=\$d '$0' exec 'timeout 5 ros2 topic echo --once /$VEH_NS/odometry/filtered' >/dev/null 2>&1 && break; sleep 2
      done
      echo \"sim \$i (domain \$d) up\"
    done
    '$0' $tr --n_envs $n --domain_base $base $targs 2>&1 | tee '$out/trainer_$tr.log'
    kill -TERM -- -\$\$" &
  VEC_PG=$!   # global: the EXIT trap runs after this function's locals are gone
  log "sims + $tr trainer starting; logs: $out/  (Ctrl-C stops all)"
  wait "$VEC_PG" || true
}

case "${1:-help}" in
  setup)       clone_repos; [[ -f "$SIF" ]] || build_image; colcon_build ;;
  build-image) build_image ;;
  build)       shift; colcon_build "$@" ;;
  sim)         shift; run_sim "$@" ;;
  train)       shift; in_container "cd '$RL_REPO/sb3' && python3 sac_sb3.py $*" ;;
  ddpg)        shift; in_container "cd '$RL_REPO/sb3' && python3 ddpg_sb3.py $*" ;;
  ppo)         shift; in_container "cd '$RL_REPO/sb3' && python3 ppo_sb3.py $*" ;;
  sac)         shift; in_container "cd '$RL_REPO/sb3' && python3 sac_sb3.py $*" ;;
  td3)         shift; in_container "cd '$RL_REPO/sb3' && python3 td3_sb3.py $*" ;;
  dreamer)     shift; in_container "cd '$RL_REPO/sb3' && python3 dreamer_train.py $*" ;;
  evalckpt)    shift; in_container "cd '$RL_REPO/sb3' && python3 eval_checkpoints.py $*" ;;   # deterministic eval of saved checkpoints (via vec)
  multi)       shift; run_multi "$@" ;;
  vec)         shift; run_vec "$@" ;;
  shell)       log "ROS_DOMAIN_ID=$ROS_DOMAIN_ID"; in_container "exec bash --norc -i" ;;
  exec)        shift; in_container "$*" ;;
  stop)        pkill -u "$USER" -f "stonefish_simulator|ros2 launch race2|mvp_control_ros_node|ekf_node|_driver_node|pressure_sensor_node|ros2-daemon|world_odom_transform|robot_state_publisher|static_transform_publisher" || true
               log "stopped" ;;
  check)       in_container 'nvidia-smi -L; python3 -c "import torch;print(\"torch cuda:\",torch.cuda.is_available())";
                 vglrun -d egl glxinfo -B 2>/dev/null | grep -E "renderer string|core profile version" || echo "no X display (sim needs --headless)";
                 echo "ROS topics:"; ros2 topic list | grep $VEH_NS | head -20 || true' ;;
  *)           sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//' ;;
esac
