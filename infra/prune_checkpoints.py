#!/usr/bin/env python3
"""Delete the training checkpoints nothing uses: per run keep the checkpoint nearest each multiple of --every (the set
sb3/eval_checkpoints.py evaluates) and the newest one; files written in the last 2 minutes are never touched.

usage: python3 infra/prune_checkpoints.py [--every 100000] [--dry] RUN_DIR [RUN_DIR ...]
(SB3 runs used to checkpoint every 10k steps = 300 files per 3M run; the eval uses 30.)
"""
import argparse
import glob
import os
import re
import time

import numpy as np

p = argparse.ArgumentParser()
p.add_argument('runs', nargs='+')
p.add_argument('--every', type=int, default=100000)
p.add_argument('--dry', action='store_true')
args = p.parse_args()

total = 0
for run in args.runs:
    files = sorted((int(m[1]), f) for f in glob.glob(os.path.join(run, 'checkpoints', '*_steps.*'))
                   if (m := re.search(r'_(\d+)_steps\.(zip|pt)$', f)))
    if not files:
        continue
    steps = np.array([s for s, _ in files])
    keep = {files[-1][1]} | {files[int(np.argmin(abs(steps - k * args.every)))][1]
                             for k in range(1, int(steps.max() // args.every) + 1)}
    drop = [f for _, f in files if f not in keep and time.time() - os.path.getmtime(f) > 120]
    size = sum(os.path.getsize(f) for f in drop)
    total += size
    print(f'{os.path.basename(os.path.normpath(run))[:60]}: keep {len(keep)}, delete {len(drop)} ({size / 1e9:.2f} GB)')
    if not args.dry:
        for f in drop:
            os.remove(f)
print(f'{"would free" if args.dry else "freed"} {total / 1e9:.2f} GB')
