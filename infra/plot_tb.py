"""Snapshot of an SB3 TensorBoard run as a PNG (small multiples, one series each).
usage: python3 plot_tb.py <run_dir> <out.png>"""
import glob, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run, out = sys.argv[1], sys.argv[2]
ev = EventAccumulator(glob.glob(f"{run}/**/events.out*", recursive=True)[0]); ev.Reload()
tags = [("rollout/ep_rew_mean", "Episode reward (mean)"), ("rollout/ep_len_mean", "Episode length (steps)"),
        ("train/actor_loss", "Actor loss"), ("train/critic_loss", "Critic loss")]
INK, MUTED, BLUE = "#0b0b0b", "#52514e", "#2a78d6"
fig, axs = plt.subplots(2, 2, figsize=(10, 6.5), facecolor="#fcfcfb")
for ax, (tag, title) in zip(axs.flat, tags):
    s = ev.Scalars(tag) if tag in ev.Tags()["scalars"] else []
    ax.set_facecolor("#fcfcfb")
    ax.plot([p.step for p in s], [p.value for p in s], color=BLUE, lw=2, marker="o", ms=4)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color="#e4e3df", lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color("#c9c8c2")
    ax.set_xlabel("env steps", color=MUTED, fontsize=8)
fig.suptitle(run.rstrip("/").split("/")[-1], color=INK, fontsize=12, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(out, dpi=120)
print(out)
