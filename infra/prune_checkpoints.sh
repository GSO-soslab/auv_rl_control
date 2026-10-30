#!/bin/bash
# Keep the 5 newest SB3 checkpoints + one per 25k steps; run until the trainer exits.
# usage: prune_checkpoints.sh <checkpoints_dir>
dir="$1"
while pgrep -u "$USER" -f "_sb3.py" >/dev/null; do
  ls -t "$dir"/*_steps.zip 2>/dev/null | tail -n +6 | while read -r f; do
    n=$(basename "$f" | grep -oE '[0-9]+_steps' | grep -oE '[0-9]+')
    (( n % 25000 < 550 )) || rm -f "$f"
  done
  sleep 60
done
