#!/bin/bash
# Per-node ROS domain allocation, so jobs that share a node never share a domain.
# (Before 2026-10-30 every job used ROS_DOMAIN_ID = uid%100+1 = 53...; two jobs on one node then exchanged
#  ground truth, thruster commands and respawns: 37 overlapping job pairs since 2026-10-27.)
#
#   base=$(infra/domain_lock.sh acquire N OWNER_PID)   -> first domain of N consecutive free domains; held while OWNER_PID lives
#   infra/domain_lock.sh release OWNER_PID             -> free every block held by OWNER_PID
#   infra/domain_lock.sh status
# Domains 1..100 in 4 blocks of 25 (ROS 2 domain ids on Linux: 0..101). Locks are directories in node-local /tmp
# (mkdir is atomic); a lock whose owner process is gone is stale and reclaimed.
set -euo pipefail
DIR=/tmp/ros_domain_locks_$USER
BLOCK=25; NBLOCKS=4
mkdir -p "$DIR"

owner_alive() { local f="$DIR/b$1/owner"; [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null; }

case "${1:-status}" in
  acquire)
    n=${2:?N}; owner=${3:-$PPID}; need=$(( (n + BLOCK - 1) / BLOCK ))
    for attempt in $(seq 1 30); do
      # block 2 (51-75) last: jobs started before this allocator existed use 53-75 without a lock
      starts=$(seq 0 $((NBLOCKS - need)) | awk -v n=$need '{print ($1 <= 2 && $1 + n - 1 >= 2) ? "1 " $1 : "0 " $1}' | sort -n | cut -d' ' -f2)
      for start in $starts; do
        got=()
        for b in $(seq $start $((start + need - 1))); do
          if mkdir "$DIR/b$b" 2>/dev/null || { ! owner_alive $b && rm -rf "$DIR/b$b" && mkdir "$DIR/b$b" 2>/dev/null; }; then
            echo "$owner" > "$DIR/b$b/owner"; got+=($b)
          else
            break
          fi
        done
        if [ ${#got[@]} -eq $need ]; then
          sleep 0.5   # two reclaimers of the same stale lock: the later owner write wins, the other retries
          mine=1; for b in "${got[@]}"; do [ "$(cat "$DIR/b$b/owner" 2>/dev/null)" = "$owner" ] || mine=0; done
          if [ $mine = 1 ]; then echo $((start * BLOCK + 1)); exit 0; fi
          for b in "${got[@]}"; do [ "$(cat "$DIR/b$b/owner" 2>/dev/null)" = "$owner" ] && rm -rf "$DIR/b$b"; done
          got=(); continue
        fi
        for b in "${got[@]}"; do rm -rf "$DIR/b$b"; done   # partial: give back, try the next start
      done
      sleep $((RANDOM % 5 + 2))
    done
    echo "domain_lock: no $need free block(s) of $BLOCK domains on $(hostname)" >&2; exit 1 ;;
  release)
    owner=${2:-$PPID}
    for d in "$DIR"/b*; do [ -f "$d/owner" ] && [ "$(cat "$d/owner")" = "$owner" ] && rm -rf "$d"; done; true ;;
  status)
    for b in $(seq 0 $((NBLOCKS - 1))); do
      if [ -d "$DIR/b$b" ]; then s="owner $(cat "$DIR/b$b/owner" 2>/dev/null) $(owner_alive $b && echo alive || echo STALE)"; else s=free; fi
      echo "block $b (domains $((b * BLOCK + 1))-$(((b + 1) * BLOCK))): $s"
    done ;;
esac
