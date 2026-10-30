# Handoff: where we are and what's next

**Last updated:** 2026-10-27 22:15 (ET).

**Read first:** `docs/rl_training_pipeline.md` (architecture, env, reward, surge modes §5b, articulated §5a, decision log).
This file is only the *current state*. Update both when things change.

## Goal (Farhang Naderi, URI)

The paper: deep RL (MLP policies; **no transformers**, that's the PI's direction) for the RACE2 AUV with
**articulated (azimuth) surge thrusters**.
- **Baseline:** the user's own Zhou, Naderi, Yuan, IROS 2025 controller (MIMO PID + convex-QP allocation), already in the pinned mvp_control. PDF in the repo root.
- **On top of it:** a horizon/integral surge-tracking formulation, plus an action space, constraints and reward derived from the IROS paper.
- **Comparison:** the paper's Fig. 4 protocol, with plots.

## Done (2026-10-27)

**Surge tracking works for the first time.** The root causes were env bugs, not the algorithm (decision log §8):
- mvp zero-thrust spam
- all-zero error signal
- mixed-unit reward
- no resets / EKF divergence
- no exploration

**Five stage-3 runs, all finished.** Settings shared by all:
- all DOF commanded, setpoint every 20 s, 20% hover
- surge ±1 m/s, surge weight 3
- fixed thrusters

Training metrics, mean over the last 10% (they include exploration noise and transients):

| Run (`sb3/logs/`) | surge err (m/s) | depth (m) | yaw (rad) |
|---|---|---|---|
| `ppo_20261027_164019_stage-3_surge_reward-instant` | 0.176 | 1.07 | 0.44 |
| `ppo_20261027_164137_stage-3_surge_reward-horizon` | 0.161 | 0.93 | **0.34** |
| `ppo_20261027_164206_stage-3_surge_reward-integral_obs` | **0.078** | 0.89 | 0.51 |
| `ddpg_20261027_164315_stage-3_surge_reward-instant` | 0.101 | 1.51 | 0.72 |
| `ddpg_20261027_164351_stage-3_surge_reward-horizon` | 0.174 | 2.23 | 0.57 |

**Live tests** (rendered, `sb3/live_policy.py`, setpoints via `~/ros2_ws/runs/live/setpoint.json`):

- **PPO integral_obs:**
  - surge 0.5 m/s reached in ~3 s and held at ±0.01
  - stop in ~3 s
  - yaw step to 65° in ~6 s with 1° overshoot
  - weak spots: a −3° steady yaw offset under way, and depth sags 0.1–0.5 m while cruising
- **DDPG instant:**
  - **bang-bang**, saturated thrusters
  - heading drifted 34° during surge, and a 0.1 m/s steady surge overshoot
  - unexplained: in hover it held both surge thrusters near −0.9 for ~40 s with the vehicle not moving. Investigate.

## Fig-4 evaluation (done 2026-10-27 21:00)

All 5 models were evaluated on the IROS Fig-4 protocol (deterministic, surge ±0.15 m/s).
- **Archive:** `results/race2/2026-10-27_surge_formulation_study/` (models, configs, code patches, `fig4_v015/`, and `compare/` plots and table; git-ignored)
- **Surge RMSE:** PPO instant 0.050; **horizon 0.025** (best balance: lowest energy and pitch, best PPO yaw); **integral_obs 0.016** (tightest surge, worse yaw/depth); DDPG instant 0.019 but **failed the dive**; DDPG horizon 0.026. DDPG uses 4–8× PPO's energy.
- **One seed, one speed.** Before paper claims: run 3 seeds each of PPO instant/horizon/integral_obs and evaluate at ±0.15 and ±0.5 m/s.
- **Re-evaluate:** `infra/run_fig4_all.sh <surge>` (edit its RUNS list).
- **Compare plots:** `python3 sb3/plot_fig4_compare.py --out DIR label=path/fig4_steps.json ...`

## Running now (2026-10-28): v4 = v3 + error integrals + feasible attitude targets

## race_auv port (2026-10-29): steps 1–4 done, no training yet

**User rule:** every vehicle has its own config across env, policies and pipeline (race2, race_auv; ALPHA later).

**Vehicle config:**
- `sb3/vehicles.py` lists each vehicle: namespace, thrusters in action order, servos, IMU, roll authority, `gt_roll_offset`.
- `RACE2_VEHICLE` selects it (default `race2`, so all race2 models and results are unchanged).
- `race2.sh` maps it to the namespace and launch (`VEH_NS`/`VEH_LAUNCH`).
- Run tags get a `vehicle-<name>` field when the variable is set.
- Results live in `results/<vehicle>/`.

**Workspace:**
- world_of_stonefish is on `jazzy-devel-race-docking-station` (d51d59e). The race2 scenes are unchanged there; the only difference is a typo inside a commented-out block.
- `~/ros2_ws/src/race_auv` is on `jazzy-devel-perception` (c7c7263). Submodules are not pulled; they aren't needed for RL.
- mvp_control is on jazzy-devel.
- Local patches are saved in `infra/patches/`:
  - ground-truth odometry sensors (20 Hz) for race2_auv.scn and race_auv.scn
  - race2_auv launch/config edits
- The `race2.sh` REPOS list holds these branches.

**Sim:**
- `infra/launch/race_auv_rl_sim.launch.py`: Stonefish + race_auv thruster/IMU/DVL/pressure drivers + description + EKF.
  - mvp_control only without `--rl`.
  - With `--nogpu`, a temp copy of the vehicle without cameras is written (the no-GPU simulator can't run cameras).
- World: `infra/stonefish/race_auv_rl.scn`, the open water of race2_test.scn (0.1 m/s current, 40 m seabed). race_auv_test.scn is the ORL tank, too small for RL episodes.
- Start: `RACE2_VEHICLE=race_auv race2.sh sim [--rl] [--nogpu]`.

**Checked on the sim (2026-10-29):**
- **Ground-truth roll offset:** the odom sensor reads roll π when upright (`gt_roll_offset = π`; respawn is NOT rotated).
- **Stable** upright for 10 s at zero thrust; slightly positively buoyant (rises 0.036 m/s).
- **Rates:** every reported body rate matches finite differences.
- **Thruster signs:**
  - heave_bow + → dives, nose down
  - stern heave + → stern down, nose up, rolls to that side
  - surge pair + → yaws to the opposite side
  - sway_stern + → yaw
- **Depth** is measured at the Base point (the nose of the URDF frame), not at the CG.
- **Observation** is 45-dim with v4 flags; action is 6 thrusters.

**Open-loop authority at 6 m:**
- Pitch: >85° at full differential (tumbles through 90°); 30–40° at 50%.
- Roll: ±41° peak, ±28° held.

**race2 regression:** a v4 PPO fixed model evaluated normally under the refactored env.

**Rotation-based attitude (`RACE2_ATT_REP=rot`, 2026-10-29):**
- **Error:** the rotation vector of Rᵀ·R_d, continuous through 90°. Checked: 85° → 90.1° → 95° pitch gives errors 5° → 0.1° → −5°.
- **Observation:** R_err columns 1–2 plus gravity in the body frame. 42-dim for race_auv.
- **Termination:** tilt error (body z axis vs reference) > 1.4 rad; heading is ignored.
- **race2 without the flag is unchanged.**

**race_auv training submitted 2026-10-29 (user: roll AND pitch targets 0–90°):**
- Jobs: `rauv_{ppo,td3,dreamer}_s{1,2,3}`, Slurm 1582027–1582035; 3M steps, 8 h limit. PPO s1 (1582027) lost its sims at 7 min to a `race2.sh stop` from the interactive session on the same node (a0003); cancelled, run dir moved to `sb3/logs/_failed/`, resubmitted as 1583493 (excludes a0003). Evals 1582089/1582090 now depend on 1582028–1582035 + 1583493.
- Settings: `RACE2_VEHICLE=race_auv RACE2_SERVOS=0 RACE2_STAGE=3 RACE2_SURGE_REWARD=horizon RACE2_PAPER_TERMS=1 RACE2_DEPTH_INV=1 RACE2_ATT_SP=1 RACE2_REF_ALL=1 RACE2_INTEGRAL=1 RACE2_ROLL_MAX_DEG=90 RACE2_PITCH_MAX_DEG=90 RACE2_DEPTH_MAX=12 RACE2_ATT_REP=rot`
- **Expect roll to saturate:** open-loop roll authority is only ±41° peak / ±28° held.
- **Evaluations queued** (start after all 9 jobs): **1582089** (Fig-4 ±0.15/±0.5) and **1582090** (new `att90` protocol in `sb3/fig4_protocols.py`: pitch ±45/±80, roll ±30/+60, climb, banked turn; dry-run OK). Results go into `results/race_auv/`; the runs themselves stay in `sb3/logs/*vehicle-race_auv*`.

**Not done:** EKF-state training, domain randomization, a deployment node.

**Three-algorithm v4 results (2026-10-29; `results/race2/2026-10-28_v4/`):**
- **Files:**
  - `summary_{fig4_v015,fig4_v05,artic}.md/.csv`
  - `plots/<algo>_<fixed|artic>_s<seed>_<test>.png` (54 plots)
  - `compare3_{fixed,artic}_{fig4_v015,artic}/` (seed 1; TD3 artic uses s2)
  - `training_details.md`
- **Fig-4 ±0.15, fixed thrusters:**
  - PPO: depth steady state 0.06 m, yaw steady state 0.4°, max pitch 3.7°, energy 266
  - TD3: depth steady state 0.04 m, yaw steady state 2.1°, max pitch 18°, energy 3287 (12× PPO, bang-bang)
  - Dreamer: depth steady state 0.66 ± 0.77 m, yaw steady state 28 ± 36° (1 seed bad), energy 1122
- **Fig-4 ±0.15, articulated:**
  - PPO: 2/3 seeds OK (s2 sank)
  - TD3: 2/3 (s1 diverged in training); servos chatter, energy 2398
  - Dreamer: 3/3; depth steady state 0.11 m, yaw steady state 2.7°, energy 294. Servo travel ~1000 rad (chatter), port servo near −90° at times
- **Artic protocol:** no algorithm tracks roll (roll-command RMSE ~10° on ±10° commands). Pitch-command RMSE: PPO/TD3 ~4°, Dreamer ~7°.
- **Summary means include failed seeds.** "n (1 failed)" rows: read per-run CSVs for the successful-seed values.
- **Timeouts:** td3_rate_s1 (diverged) and dreamer_fixed_s2 (2.70M checkpoint) both hit the 7 h limit. Their job.json files were written by hand with a `note`.
- **Eval fix 2026-10-29:** `torch.set_num_threads(1)` in eval_fig4/live_policy. Without it, the first Dreamer evaluations starved the real-time sims (19–75 s of frozen state); they were rerun as jobs 1581525/1581526. PPO/TD3 evaluations were unaffected (≤1.2 s).
- **Hardware target (2026-10-29):**
  - The real vehicle is **GSO-soslab/race_auv, branch jazzy-devel-perception**, NOT race2_auv (ROS 1, old).
  - It has 6 FIXED thrusters: surge port/stbd tilted 46° and toed 30°, heave bow + stern port/stbd, sway stern. No servos; Nortek DVL.
  - Stonefish model: `world_of_stonefish/vehicles/race_auv.scn` and `world/race_auv_test.scn` (no ground-truth odometry sensor yet).
  - **Branches to pull (user, 2026-10-29):** race_auv `jazzy-devel-perception` (with submodules) and world_of_stonefish `jazzy-devel-race-docking-station`. They are also recorded as TODO lines in `infra/race2.sh` REPOS.
  - Deployment path: generalize AUVEnv to vehicle configs, then retrain the PPO v4 recipe with randomization/EKF state, then a ROS 2 inference node.

**PPO v4 results (evaluated 2026-10-28; tables in `results/race2/2026-10-28_v4/summary_*.md`):**
- **Fixed thrusters, 3/3 seeds: good.**
  - Fig-4 ±0.15: depth steady state 0.06 ± 0.03 m (v1 had 0.5–1.2 m), yaw steady state 0.4°, surge RMSE 0.020 m/s.
  - Pitch commands tracked to ~4° RMSE. Roll is not tracked (expected: no roll authority).
- **Articulated: not better than fixed.**
  - Roll-command RMSE ~10° on ±10° commands, i.e. roll is ignored. Pitch ~4°, the same as fixed.
  - Seed 2 diverged: it terminated in Fig-4 block 2 and in the artic nose-down dive.
  - Seeds 1 and 3 behave like the fixed policy, with servos moving ≤15°.
- **Next:** articulated plan B (see the decision log), and compare with TD3/Dreamer once their evaluations finish.

**User rule (2026-10-28): ONE RL policy per vehicle configuration**, both with the same recipe: fixed thrusters, and articulated (rate action, ±90°). Each covers all maneuvers. Seeds are fine; no angle sweeps, no per-task variants. The main comparison is RL articulated vs the IROS baseline at ±90°.

**Jobs** (PPO **3M steps**, ~4.2 h each, `--exclude=a0001` = desktop node):
- `v4_rate_s{1,2,3}`: Slurm 1565754, 1565756, 1565758
- `v4_fixed_s{1,2,3}`: Slurm 1566094–1566096 (resubmitted after a brief cancel)
- Settings: `RACE2_STAGE=3 RACE2_SURGE_REWARD=horizon RACE2_PAPER_TERMS=1 RACE2_SERVO_LIMIT_DEG=90 RACE2_DEPTH_INV=1 RACE2_ATT_SP=1 RACE2_K_SLEW=0 RACE2_REF_ALL=1 RACE2_INTEGRAL=1 RACE2_ROLL_MAX_DEG=12 RACE2_PITCH_MAX_DEG=20 RACE2_DEPTH_MAX=12`
- fixed: `RACE2_SERVOS=0`; rate: `RACE2_SERVOS=2 RACE2_ACTION=rate`
- Evaluations queued: **1565759** (Fig-4 ±0.15/±0.5) and **1565760** (articulation protocol, now with feasible targets: roll ±10°, pitch ±15°)
- Every run now snapshots `sb3/config/` into `run_info/config/`

**TD3 v4** (submitted 2026-10-28 11:50 at the user's request): same settings, 3M steps, `TRAINER=td3`.
- `td3_fixed_s{1,2,3}`: Slurm 1567179, 1567181, 1567183
- `td3_rate_s{1,2,3}`: Slurm 1567180, 1567182, 1567184
- Script `sb3/td3_sb3.py`, hyperparameters `sb3/config/config_td3.yaml` (Fujimoto 2018 defaults: lr 1e-3, batch 256, σ 0.1, policy delay 2, 256×256)
- The env and reward come from `config_sac.yaml`, as for PPO
- `race2 vec N td3`; eval_fig4 / live_policy / eval_policy accept `--algo td3`
- Evaluations queued after training: **1567502** (Fig-4 ±0.15/±0.5) and **1567503** (artic protocol)
- Speed at start: ~210 steps/s fixed, ~175 articulated
- **td3_rate_s1 TIMEOUT** at 2.77M of 3M steps (slow node). `final_model.zip` = last checkpoint, and `run_info/job.json` was written by hand (see its `note`). Evaluated separately: job 1572892.
  - **That seed DIVERGED in training at ~2.4M steps:** episode length fell from ~600 to ~18 steps (it flips at once, terminated rate 0.1 per step). The final model flips in 1.8 s in every test. Report it as a failed seed (TD3 instability). The node was not the cause; the respawns made the job slow.

**DreamerV3 v4** (submitted 2026-10-28 12:15): same settings, 3M steps, `TRAINER=dreamer`.
- **⚠ NOT TESTED ON STONEFISH.** Only the offline fake-env test ran. The first real job is its first contact with the sims (SubprocVecEnv, real infos, GIL contention). Check a job `.out` and `progress.csv` first (fps ~200, `train/updates_per_s` > 0, no traceback). If one job failed, all 6 likely did the same way.
- `dreamer_fixed_s{1,2,3}`: Slurm 1567371, 1567373, 1567375
- `dreamer_rate_s{1,2,3}`: Slurm 1567372, 1567374, 1567376
- **Code.** Upstream is the PyTorch port NM512/dreamerv3-torch at `~/dreamerv3-torch`, commit 6ef8646, used unchanged.
  - RSSM (GRU) world model with MLP heads; no transformer.
  - `ruamel.yaml` is installed in `~/dreamerv3-torch/_deps`, because the container's pip is PEP 668 locked.
- **Our files:**
  - `sb3/dreamer_agent.py`: config, agent, and `DreamerPolicy` (load/predict for evaluation)
  - `sb3/dreamer_train.py`: the driver
  - `sb3/config/config_dreamer.yaml`: upstream `dmc_proprio` (19.9M parameters) plus action_repeat 1, prefill 10k, train_ratio cap 512, compile on
- **Design.** The sims run in real time, so collection (main thread, acting copy of the agent) and training (learner thread) are decoupled. Upstream trains between env steps, which would stretch the 10 Hz control period.
- **Offline test** (fake 23-env real-time vec env): env 216–220 steps/s; learner 9.4 updates/s with compile (7.7 without). The effective replay ratio is ~35–45, not 512. That is ~135k updates over 3M steps, about half the dmc_proprio budget. `train/replay_ratio` in progress.csv shows the real value.
- **Output:** `final_model.pt` (not .zip). `run_fig4_runs.sh`, `eval_fig4.py --algo dreamer` and `live_policy.py --algo dreamer` handle it.
- Evaluations queued after training: **1567504** (Fig-4) and **1567505** (artic protocol)

**Why v4:** actuator-authority test plus live tests (pipeline doc §5a "Actuator authority" and the decision log). The user watched v1 fixed, v1 naive, v2 rate s1 and v2 rate s2 live; none was good enough.
- v1: steady depth offsets of 0.5–1.2 m; ignores pitch/roll commands (never trained on them)
- v2: discarded

**IROS baseline evaluation:** job **1566085** (`eval_mvp`), servo limit ±90° only, × (Fig-4 ±0.15, Fig-4 ±0.5, articulation protocol). Output in `results/race2/2026-10-28_baseline_mvp/lim90/<tag>/`. Script `sb3/eval_mvp.py`; see pipeline doc §5a "Planned comparison".

**Launch change (local, race2_auv):** `servo_limit_deg:=N` patches the scene joint limits and mvp angle_limits. `race2.sh sim` passes it when `RACE2_SERVO_LIMIT_DEG` is set, so batch training sims started after 2026-10-28 09:20 run with a ±90° joint limit. This is identical for RL, because the policy is clipped to ±90° anyway.

**What to check first:**
1. steady-state depth offset (should be ≪ 0.5 m)
2. pitch smoothness and bow-thruster chatter
3. roll ±10° tracking for rate vs fixed (fixed cannot roll)
4. servos not pinned at ±90°

## v3 (2026-10-28) — stopped at ~10 % (230k of 2M steps); targets were infeasible, see v4

**Jobs:** `v3_{fixed,rate}_s{1,2,3}`, Slurm 1565311–1565313, 1565315, 1565316, plus **1565372** (rate s2, resubmitted after it was killed). PPO 2M steps.
- Settings: same as v2 plus `RACE2_REF_ALL=1`
- Fig-4 evaluation queued as **job 1565317** (`fig4_eval_v3`, `JOB_PREFIX=v3_`)
- **Articulation protocol** (`eval_fig4.py --protocol artic`; output `<run>/artic/`): 9 × 60 s combo blocks — settle, roll ±20°, roll 15° + surge 0.5, nose-down dive (pitch −20°, surge 0.5, depth 8), nose-up climb, fast dive (surge 0.8, depth 8), banked turn (yaw 90° + roll 15° + surge 0.5), level. Fixed thrusters cannot roll (centerline heave thrusters), so they should fail the roll blocks.
  - v1 + v2 models: job 1565369 (`artic_eval_v1v2`)
  - v3: job 1565370 (`artic_eval_v3`, after training)
  - Batch: `PROTOCOL=artic JOB_PREFIX=a,b SURGES=x sbatch … infra/eval_fig4.sbatch`

**v2 results** (`results/race2/2026-10-28_articulated_v1_v2/`):
- `summary_fig4_v0{15,5}.md`: mean ± std over seeds
- `compare_v0{15,5}/`: overlays of one seed per variant
- `fig4_v05_*.png`: per-model plots

**v2 failed on depth:** Fig-4 depth RMSE 4–8 m, and fixed s3 and rate s3 sank to 28 m. Yaw and surge were better than v1 for fixed. Rate s1 parked its port servo at −90° (thruster pointing down) and stopped surging entirely; rate s2 and s3 barely moved their servos. Cause and fix are in the pipeline doc decision log (v3 entry).

Aggregation script: `sb3/summarize_fig4.py --out DIR "label=glob" ...`. Run plots inside the container (`race2.sh exec`), since host python has no matplotlib.

## v2 articulated task (2026-10-28, done) — **DISCARDED by the user, do not use v2 models**

Live checks (2026-10-28):
- rate s1: port servo pinned at −90°, never surges
- rate s2: surges, but pitch oscillates −14..+19° with a bang-bang bow heave thruster (−1 ↔ +0.95), and depth is 1.4 m off
- s3 and fixed s3 sank in Fig-4

Files are kept only as a record. For v3, check pitch smoothness and bow-thruster chatter first; if they are still bad, raise `k_rate`.
Live demos: `infra/live.sh start <run_dir>` / `infra/live.sh stop`. Stop kills only processes on the live ROS domain (95).

**Why v2:** live tests of the v1 naive articulated policy showed the servos never move more than ±10°. Even with both heave thrusters saturated during a dive while surging, they stayed put, and the policy ignores sway. Causes:
- nothing in the task needed articulation
- servo motion was penalized twice
- **the servos tilt thrust vertically**, so articulation gives heave/pitch/**roll** authority, never sway

v1 also stalled 0.5 m short of a 10 m target: absolute depth was in the observation, and targets only went 1–8 m. Details are in pipeline doc §5a and the decision log.

**Jobs:** 6 batch jobs `v2_{fixed,rate}_s{1,2,3}`, Slurm 1558514–1558519, PPO, 2M steps (~2.8 h each), rtx-batch 32 CPUs.
- Common settings: `RACE2_STAGE=3 RACE2_SURGE_REWARD=horizon RACE2_PAPER_TERMS=1 RACE2_SERVO_LIMIT_DEG=90 RACE2_DEPTH_INV=1 RACE2_ATT_SP=1 RACE2_K_SLEW=0`
- fixed: `RACE2_SERVOS=0`, pitch setpoints only (no roll authority)
- rate: `RACE2_SERVOS=2 RACE2_ACTION=rate`, pitch and roll setpoints

**After they finish:**
1. Fig-4 evaluation via `infra/run_fig4_runs.sh` or `eval_fig4.sbatch` with `JOB_PREFIX=v2_`.
2. **Add an attitude-step test** to the evaluation: roll/pitch steps to ±15/30/45°, a large dive (e.g. 2 → 20 m) while surging, and a check of servo use.
3. Live demo with `live_policy.py`. The control file now takes `pitch_deg`, `roll_deg` and `sway` keys.

Queued: **v2 Fig-4 evaluation, job 1558555** (`fig4_eval_v2`), which runs after the 6 v2 jobs (±0.15 and ±0.5 m/s). The **v1 Fig-4 evaluation, job 1558226**, started 2026-10-28; its results are in each `art90_*` run dir under `fig4_v015/` and `fig4_v05/`.

## Earlier: paper-grade articulated experiment v1 (submitted 2026-10-27 22:12, done)

**Settings:**
- 6 batch jobs: `art90_{fixed,naive}_s{1,2,3}`, Slurm IDs 1557386–1557391
- PPO + **horizon** surge + `RACE2_PAPER_TERMS=1`, azimuth limit ±90°, stage 3, 1M steps, seeds 1/2/3
- rtx-batch, 32 CPUs → 23 sims each
- Variants:
  - fixed: `RACE2_SERVOS=0`
  - naive: `RACE2_SERVOS=2 RACE2_ACTION=naive` (end-to-end: network outputs 4 thrusts + 2 absolute servo angles)

**Timing:** each run takes ~80 min once started.

**Log dirs:** `sb3/logs/ppo_<time>_stage-3_surge_reward-horizon[_action-…]_servos-…_servo_limit_deg-90_paper_terms-1_seed-N`.

**Timing + hardware are logged automatically** (the user wants this for the paper):
- Each run gets `<log dir>/run_info/`:
  - `job.json`: node, GPU/CPU model, CPUs, settings, git commits, wall time, exit status
  - `gpu_usage.csv`: utilization / memory / power every 30 s
  - `cpu_load.csv`: this user's CPU cores in use + RSS every 30 s
- Collect on the HOST (not in the container; it needs sacct): `python3 infra/collect_run_stats.py --filter <substr> ... --out results/<vehicle>/<run>/run_stats` writes per-run + per-algorithm mean ± std tables. Paper v4 race2: `--filter ref_all-1 integral-1 --out results/race2/2026-10-28_v4/run_stats`; race_auv: `--filter vehicle-race_auv`.

**After they finish:**
1. **Fig-4 evaluation is already queued:** job **1558226** (`fig4_eval`) waits on the training jobs (`--dependency=afterany`).
   - It evaluates every run with job name `art90_*` and exit status 0, at ±0.15 and ±0.5 m/s (18 sims in parallel).
   - Output: `<run>/fig4_v015/` and `<run>/fig4_v05/`. Log: `~/ros2_ws/fig4_eval-1558226.out`.
   - Scripts: `infra/eval_fig4.sbatch` calls `infra/run_fig4_runs.sh "SURGES" run_dirs...`. The latter reads each model's RACE2_* settings from `run_info/job.json`, so the observation and action layout always match. Use it for any future evaluation; `run_fig4_all.sh` is the old hardcoded version.
   - Report mean ± std over seeds, steady-state metrics too (see the archive README's steady-state table).
2. Plots with `sb3/plot_fig4_compare.py`.
3. Archive to `results/race2/2026-10-27_articulated_90deg/` with the same structure as the surge study.
4. `python3 infra/collect_run_stats.py ...` on the host for the time/hardware table (see above).

**Agreed with the user (not started):** make the codebase modular and replicable *after* this evaluation and *before* TD3/Dreamer:
- one YAML per experiment (`experiments/*.yaml`)
- a single `train.py --exp … --seed N` entry point, with algorithms as modules
- copy the experiment file into the log dir
- keep RACE2_* env vars working as overrides so the current runs stay reproducible
- add a README section on reproducing each table
- also rename the shared `config_sac.yaml`, which every algorithm reads

Plan after that, per algorithm (TD3, then DreamerV3, whose RSSM is a GRU, not a transformer): fixed + the winning articulated action, 3 seeds each.

**PI question on Euler angles (answered 2026-10-27):** errors are wrapped, the policy sees (sin e, cos e), and the reward uses |e|, so it is continuous at ±π. Every 180° turn in Fig-4 is a single turn in one direction. The 20° pitch spike is DDPG horizon only, at t = 405 s (depth and surge reverse at the same time), caused by saturated heave thrusters. PPO max pitch is ≤ 9.6° and 0.8–1.6° RMS. Yaw/depth overshoot is step-response damping, to be fixed by a reference trajectory on depth/yaw.

## Next steps (in order)

1. **(done) Fig-4 results:** see above. **Next:** multi-seed + ±0.5 m/s evaluation, then decide the surge formulation.
   Old note: Decide which surge formulation goes in the paper. The training numbers favor `integral_obs`; horizon is best on yaw. Be honest in the paper (novelty notes in §5b).
   Optionally rerun at `--surge 0.5`.
2. **Build `eval_fig4.py --controller mvp`**, the IROS baseline:
   - Run `race2 sim` **without** `--rl`, so mvp drives the thrusters.
   - Enable with `/race2_auv/controller/set` (std_srvs/SetBool, true).
   - Publish setpoints on `/race2_auv/controller/process/set_point` with `control_mode: 4dof`.
   - mvp's state equals ground truth for depth, yaw, pitch and surge; **roll is π upright** (verified). Score on ground truth.
   - Don't respawn (it breaks the EKF). The default spawn is 0.5 m, heading 0, which the RL evaluation also uses now.
   - Sweep the angle limit via the scenario arg `servo_min/max_ang` (race2_test.scn) and race2's mvp `angle_limits`.
3. **Overlay plotting script:** read any set of `fig4_steps.csv` files. Produce time-series overlays (paper Figs. 5–7 style) and per-block bar charts (RMSE depth/yaw/surge, peak roll/pitch, energy, servo travel).
4. **Fix DDPG → TD3** (user agreed it's a problem; not yet approved to implement):
   - add `--algo td3` to `ddpg_sb3.py` (SB3 TD3)
   - observation normalization (`VecNormalize`)
   - `k_rate` 0.05 → 0.2
   - keep DDPG as a baseline row
5. **Articulated experiment** (the novelty): **end-to-end RL replaces the IROS QP+PID**. The QP allocation is never used inside the policy loop (user decision 2026-10-27); it is only the baseline. One MLP policy for all DOFs, horizon surge, `RACE2_PAPER_TERMS=1`:
   - fixed: `RACE2_SERVOS=0`
   - articulated: `RACE2_SERVOS=2 RACE2_ACTION=rate` (v2; `naive` = absolute angles is the v1 ablation)
   - then `RACE2_SERVO_LIMIT_DEG=45/90/135`
   - Submit with: `RACE2_STAGE=3 RACE2_SURGE_REWARD=horizon RACE2_SERVOS=2 RACE2_ACTION=naive RACE2_PAPER_TERMS=1 RACE2_SERVO_LIMIT_DEG=90 TRAINER=ppo STEPS=1000000 EXTRA="--seed N" sbatch -p rtx-batch --cpus-per-task=32 --mem=96G --time=02:45:00 --job-name=<name> ~/ros2_ws/train_vec.sbatch`
6. **Paper comparison:** IROS QP+PID baseline (articulated) vs RL fixed / RL articulated, plus TD3 and DreamerV3 rows later. The headline is RL articulated vs the baseline.
7. **Commit:** nothing is committed. Commit or push **only when the user says so**. Local changes:
   - this repo: `sb3/`, `docs/`, `infra/`, README
   - `~/ros2_ws/src/race2_auv`: rl_mode/render launch args
   - `~/ros2_ws/src/world_of_stonefish`: `odom_gt` sensor

## User preferences (important)

- **No push or PR** unless explicitly asked.
- **No transformer models.**
- **One network for everything:** no separate "hold" stages. Holding means setpoints at 0 or constant.
- Match the IROS paper's DOFs: surge, depth and yaw commanded; pitch and roll regulated to 0; sway uncontrolled.
- Surge must work up to ±1 m/s. The user cares most about surge tracking.
- Concise answers. The user likes watching renders and wants paper-quality plots and exact formulas.

## Gotchas learned the hard way

- **Batch jobs can land on the OnDemand desktop's node** (rtx-batch included a0001, the desktop node). `pgrep -f stonefish` / `pkill` on the desktop then also hits the batch job's sims. On 2026-10-28 this killed v3_rate_s2 after 5 min, and the job still reported exit 0, because the trainer handles SIGINT quietly and no final_model.zip was written. **Kill local sims only by the process group you started** (save `$!` from `setsid`). Submit with `--exclude=<desktop node>` when testing locally at the same time.

- **CPU limits:** the OnDemand desktop is capped at 16 CPUs (11 training sims). Batch jobs on `rtx-batch` get 32 CPUs (23 sims). The GPU is not the bottleneck; real-time Stonefish needs ~1 CPU core per sim.
- **ROS domains:** `race2 vec` and `multi` use ROS domains base..base+N on the node (base 53). Never run two vec runs on one node. Use spare domains (70+) for side sims.
- **Stopping runs:**
  - `pkill -f "<pattern>"` from a shell whose command line contains the pattern kills that shell. Stop by process group.
  - `race2 stop` kills **all** of the user's sims on the node.
  - To stop one sim: `ps -o pgid=` on its stonefish process, then `kill -INT -- -<pgid>`.
- **Training metrics include exploration noise and setpoint transients.** Use `eval_fig4.py` or `eval_policy.py` for clean numbers.
- **eval_fig4 must disable random setpoint changes** (`setpoint_change_seconds = 0`; fixed 2026-10-27).
- **Env vars must match the model:** `RACE2_SURGE_REWARD=horizon|integral_obs` models have a 31-D observation, instant models 28-D.
- **mvp_control** (pinned `e87da17`) already implements the IROS 2025 QP:
  - `auto_vector_direction`
  - constraints at `mvp_control_ros.cpp:990`
  - λ sigmoid at `mvp_control.cpp:239`
  - `atan` in `vector_thruster_ros.cpp:207`

## Checkpoint evaluation (evaluation-return curves, 2026-10-29)

- `sb3/eval_checkpoints.py`: deterministic policy on the checkpoint nearest every 100k steps + the final model, one 60 s
  episode per sim, env i seeded `1000+i` (same setpoint sequences for every checkpoint/run/algorithm). Env settings per run
  come from `run_info/job.json`. Writes `<run>/eval_checkpoints.csv` (reward, length, terminated, RMSE depth/yaw/surge/pitch/roll,
  energy sum u^2 dt); resumable. Launch: `RACE2_VEHICLE=... TRAINER=evalckpt SIMS=20 EXTRA="--runs <dirs> --every 100000" sbatch ~/ros2_ws/train_vec.sbatch`
  (`race2.sh` got an `evalckpt` dispatch; it was swapped in atomically because running jobs execute that script).
- race2 v4 (18 runs): jobs 1583799–1583804, 3 runs each (one ppo/td3/dreamer per job).
- Plot: `python3 sb3/plot_eval_curves.py --out results/race2/2026-10-28_v4/eval_curves --filter ref_all-1 integral-1`
  (PNG + PDF, mean over seeds with 95% bootstrap CI; `_final.md` = IQM with stratified bootstrap CI, rliable-style).
- Trial outputs `eval_checkpoints_test.csv` (4 sims) in ppo_20261028_092819 and dreamer_20261028_121526 can be deleted.
- race_auv (9 runs): jobs evalckpt_rauv_{0,1,2}, queued after its training (afterany); timed-out runs fall back to the eval job's RACE2_* settings (= the training flags).

## Live piloting with the keyboard/mouse joystick (2026-10-29)

`infra/pilot.sh sb3/logs/<run>` (from an OOD desktop; needs DISPLAY): GPU-rendered sim + `sb3/joystick_pilot.py` window +
`sb3/live_policy.py`, on ROS domain 93 (`PILOT_DOMAIN`), env settings from the run's job.json, any algo/vehicle. Pad: up/down
surge (holds), left/right yaw rate; W/S surge, A/D yaw, Up/Down depth, I/K pitch, J/L roll (clamped to the training ranges),
Space stop, R respawn. Ctrl-C stops only these processes. Logs/status in `~/ros2_ws/runs/live/`. USB gamepads don't reach the
cluster over VNC, hence the on-screen pad.

## Pending decision: TD3 runs whose plots the user deleted (2026-10-29)

User deleted the plots of td3_artic_s1 (td3_20261028_114139, diverged/timed out), td3_fixed_s1 (td3_20261028_114029) and 2 of 3
of td3_fixed_s2 (td3_20261028_114143). Runs KEPT for now; user may want them wiped after the checkpoint evaluation
(jobs 1583799–1583804). Ask before deleting; the paper should still report all seeds (dropping bad ones = cherry-picking).
Agreed plan: after the checkpoint eval, `sb3/select_best_policies.py` keeps the best seed per vehicle x config x algo
(final-model eval-return IQM) in `results/<vehicle>/<date>/policies/` (+ .json settings, SELECTION.md); then, with the user's OK,
wipe the other seeds' checkpoints/models (keep progress.csv, eval CSVs, run_info).

## Servo-cost retrain + Dreamer 10M (submitted 2026-10-29 ~17:15)

Why: piloting Dreamer artic s1 showed the port servo parked at -90 deg and thruster sign flips; v4 has RACE2_K_SLEW=0 (servo
motion free). User chose: servo slew cost on for articulated (all algos) + Dreamer to 10M steps (also race2 fixed, race_auv).
- `slew_{ppo,td3}_s{1,2,3}` (1584251–1584256): race2 articulated, v4 flags + RACE2_K_SLEW=0.02, 3M.
- `dr10m_{slew,fixed,rauv}_s{1,2,3}_p{1..4}` (1584257–1584292): Dreamer 10M, 4 chained parts (afterany) per run,
  `--run_id 20261029_10M --max_hours 7.4 --checkpoint_every 250000`; slew = articulated + K_SLEW 0.02, fixed = v4 fixed flags,
  rauv = the race_auv flags. Run dirs `sb3/logs/dreamer_20261029_10M_<tags>_seed-N`.
- Dreamer resume (new in dreamer_train.py): `--run_id` fixes the dir; each job trains to --timesteps or --max_hours, saves
  resume_state.pt (weights, optimizers, replay, counters, logs; also every --resume_every 500k) and exits 0; the next job resumes;
  a finished run exits at once. Tested 2026-10-29 with a 2-part tiny run (steps/updates/time continue; deleted after).
  Later parts' run_info is nested in run_info/job_<id>_*/; collect_run_stats.py sums all parts.
- Paper: compare all algos at 3M (Dreamer's 3M checkpoint) + report Dreamer 10M separately. Run eval_checkpoints.py on the
  new runs when done (Dreamer: --every 250000).

## 2026-10-30: ROS domain cross-talk; everything rerun fresh

- Bug: every job used ROS_DOMAIN_ID 53.. (uid%100+1), so jobs on the same node shared topics/services (ground truth,
  thruster commands, /respawn_robot). Audit: 37 overlapping job pairs since 2026-10-27 (sacct node + time). Symptom that
  found it: PPO artic s1 checkpoint-eval return -279 in job 1583800 (shared a0007 with 1583799 + training jobs).
  Earlier "bad seeds" (PPO artic s2 sinking, TD3 artic s1 divergence, Dreamer fixed s2 yaw) coincide with overlaps.
- Fix: `infra/domain_lock.sh` (4 blocks of 25 domains per node, mkdir locks in /tmp, stale = owner pid dead);
  used by race2.sh vec, infra/run_fig4_runs.sh, infra/pilot.sh. Also: run_fig4_runs.sh now exports the run's RACE2_*
  to the SIM launch (race_auv Fig-4/att90 evals had launched race2 sims -> all-zero state), and AUVEnv raises if no
  ground truth arrives after a reset. Never use `race2 stop` (see memory).
- Old runs moved to sb3/logs/_pre_domainfix/ (60) and sb3/logs/_contaminated/ (5 cancelled Dreamer 10M chains);
  results/race2/2026-10-28_v4 and results/race_auv marked CONTAMINATED.md. The pending TD3 deletion decision is moot.
- Fresh paper set (user: "Everything fresh"): `infra/submit_paper_runs.sh` — PPO/TD3 3M x seeds 1-3 for fixed (fx_),
  slew = articulated + K_SLEW 0.02 (sl_), race_auv (ra_), each with Fig-4 + artic/att90 + checkpoint evals queued afterany;
  Dreamer 10M chains run_id 20261030_10M: fixed s2,s3 and slew s1-3. Kept (clean, still running, run_id 20261029_10M):
  Dreamer 10M fixed s1, race_auv s1-3. Dreamer evals: queue after the chains finish (prefix fx_dreamer10m / sl_dreamer10m /
  dr10m_rauv; eval_checkpoints --every 250000).
- Jobs 2026-10-30 ~02:05: fixed 1589869-1589884, slew 1589885-1589905, rauv 1589906-1589914.

## Disk (2026-10-30): home filled up and blocked login

Home is capped at 100 GiB (docs.aicr.ai/files/overview; not shown by `quota`). Scratch: 10 TiB, files older than 30 days are PURGED, no backup: keep selected policies/results in home (results/) or /work. User deleted sb3/logs/_pre_domainfix (old runs, ~36 GB) to log in.
- Checkpoints now live on /scratch/farhang_naderi_uri/race2_logs/<run>/checkpoints, symlinked as sb3/logs/<run>/checkpoints
  (vec_env_utils.checkpoint_dir; existing runs were moved + linked). ROS logs: ROS_LOG_DIR=/scratch/$USER/ros_log (race2.sh).
- Less written: PPO/TD3/DDPG checkpoint every 100k (was 10k = 300 files/run); Dreamer deletes resume_state.pt at the end.
- `infra/prune_checkpoints.py <runs>` keeps the 100k-spaced checkpoints + newest. Run it again on sb3/logs/{ppo,td3}_20261030*
  once those jobs finish (they still save every 10k). _contaminated deleted (user OK).
- Budget: home ~21 GB now, peak ~33 GB while the 9 Dreamer resume states exist; scratch ~13 GB, ~45 GB at the end.
- Guard: train_vec.sbatch and eval_fig4.sbatch refuse to start when home > 85 GB (HOME_MAX_GB; du takes ~4 s).

## 2026-10-30 ~07:00-12:00: home full froze every running job; recovery (~14:30)

- When home filled, every running job froze (a failed save raised, ROS threads kept the process alive until TIMEOUT; the
  traceback was lost too, the .out is in home) and jobs starting then failed in 1 s (no .out file). Fixed: trainers now exit
  on any exception (ppo/td3/dreamer `__main__`), checkpoints on scratch. All files written since 05:00 pass a zip CRC check.
- Kept (finished clean before 07:00): fx_ppo s1-3, fx_td3 s1-3, sl_ppo s1-3. Frozen partial runs -> sb3/logs/_failed/.
- Resubmitted: ra_ppo/ra_td3 s1-3 1596268-73 (+evals 1596274-76), sl_td3 s1-3 1596277-79 (+TD3-only evals 1596280-82),
  fx evals 1596283-84 (sl PPO evals 1589891/1589892 ran as submitted). Dreamer: 7 chains resume from resume_state.pt
  (fx s1 1596315-17 — old dir name has no vehicle tag, so submitted WITHOUT RACE2_VEHICLE; fx s2/s3, ra s1-3, sl s2: 3 parts
  each), sl s1/s3 fresh with run_id 20261030b_10M (4 parts).
- First clean curves (results/race2/2026-10-30_paper/training_curves_partial.png): fixed PPO ~505, Dreamer ~495 (2 seeds),
  TD3 ~400; articulated PPO ~445 with all 3 seeds tight. The old "bad seeds" (PPO artic s2, Dreamer fixed yaw) are gone,
  consistent with cross-talk having caused them.

## Repo renamed (2026-10-30)

GitHub repo is now GSO-soslab/auv_rl_control (origin updated; old URL redirects). The local folder is still
~/race2_q_learning_control: rename it only when no jobs run, together with the hard-coded paths (9 files in infra/,
sb3/, docs/; RL_REPO default in infra/race2.sh; the ~/ros2_ws/race2.sh and train_vec.sbatch symlinks).
