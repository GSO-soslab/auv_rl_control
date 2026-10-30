"""Evaluate the IROS 2025 baseline (Zhou, Naderi, Yuan: convex constrained QP allocation with azimuth thrusters
+ MIMO PID, as implemented in the pinned mvp_control) on the same protocols as eval_fig4.py, scored on Stonefish
ground truth. Output has the same format (fig4_steps.json/.csv, fig4_metrics.csv, fig4.png), so summarize_fig4.py
and plot_fig4_compare.py work unchanged.

Run with the normal sim, where mvp drives the thrusters (NOT `--rl`):
  race2 sim --nogpu      then      python3 eval_mvp.py [--protocol fig4|artic] [--surge 0.15] [--out DIR]
Notes:
  * Mode `4dof` of race2's mvp config controls z, u, v, pitch, yaw. Roll is not a controlled DOF in that mode, so
    the artic protocol's roll blocks are expected to fail for the baseline as configured.
  * Setpoints use mvp's own frame ids (read from controller/process/value). mvp's depth/pitch/yaw/surge equal ground
    truth; its roll is pi when upright (verified 2026-10-27), so the roll setpoint is pi + commanded roll.
  * No respawn (it disturbs the EKF mvp runs on): the vehicle starts at the default spawn (0.5 m, heading 0),
    the same start the RL evaluation uses.
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import rclpy
from mvp_msgs.msg import ControlProcess
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
from std_srvs.srv import SetBool

from fig4_protocols import PROTOCOLS, block_setpoint

NS = '/race2_auv/'
THRUSTERS = ('heave_bow', 'heave_stern', 'surge_port', 'surge_starboard')   # same order as the RL env's thr0..thr3
SERVO_JOINTS = ('race2_auv/surge_port_servo_joint', 'race2_auv/surge_starboard_servo_joint')

p = argparse.ArgumentParser()
p.add_argument('--protocol', default='fig4', choices=PROTOCOLS)
p.add_argument('--surge', type=float, default=0.15, help='fig4 only: surge magnitude of the move blocks')
p.add_argument('--block-seconds', type=float, default=None)
p.add_argument('--mode', default='4dof', help='mvp control mode (race2 config: 4dof = z, u, v, pitch, yaw)')
p.add_argument('--out', required=True)
args = p.parse_args()
blocks, default_len = PROTOCOLS[args.protocol]
block_seconds = args.block_seconds or default_len
os.makedirs(args.out, exist_ok=True)
wrap = lambda a: (a + np.pi) % (2 * np.pi) - np.pi

rclpy.init()
node = rclpy.create_node('eval_mvp')
st = {'thr': np.zeros(4), 'servo': np.zeros(2)}


def on_gt(m):
    q, v = m.pose.pose.orientation, m.twist.twist.linear   # same conversion as AUVEnv.ground_truth_callback
    st['roll'] = np.arctan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
    st['pitch'] = np.arcsin(np.clip(2 * (q.w * q.y - q.z * q.x), -1.0, 1.0))
    st['yaw'] = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    st['depth'], st['surge'], st['sway'] = m.pose.pose.position.z, v.x, v.y


def on_value(m):
    st['frames'] = (m.header.frame_id, m.child_frame_id)


def on_joints(m):
    for i, name in enumerate(SERVO_JOINTS):
        if name in m.name:
            st['servo'][i] = m.position[m.name.index(name)]


node.create_subscription(Odometry, NS + 'odometry/ground_truth', on_gt, 10)
node.create_subscription(ControlProcess, NS + 'controller/process/value', on_value, 10)
node.create_subscription(JointState, NS + 'stonefish/servo/joint_states', on_joints, 10)
for i, name in enumerate(THRUSTERS):
    node.create_subscription(Float64, NS + f'control/thruster/{name}', lambda m, i=i: st['thr'].__setitem__(i, m.data), 10)
sp_pub = node.create_publisher(ControlProcess, NS + 'controller/process/set_point', 10)


def spin_for(sec):
    t_end = time.time() + sec
    while time.time() < t_end:
        rclpy.spin_once(node, timeout_sec=min(0.01, max(t_end - time.time(), 0.0)))


# Wait for ground truth and mvp's process value (EKF up), then enable the controller
t0 = time.time()
while ('frames' not in st or 'depth' not in st) and time.time() - t0 < 60:
    spin_for(0.1)
if 'frames' not in st:
    raise SystemExit('no controller/process/value from mvp_control: is the sim running without --rl?')
print('mvp frames:', st['frames'], flush=True)


def setpoint(depth, yaw, surge, roll, pitch):
    m = ControlProcess()
    m.header.stamp = node.get_clock().now().to_msg()
    m.header.frame_id, m.child_frame_id = st['frames']
    m.control_mode = args.mode
    m.position.z = depth
    m.orientation.x, m.orientation.y, m.orientation.z = float(wrap(np.pi + roll)), pitch, yaw   # mvp roll = pi upright
    m.velocity.x = surge
    return m


depth0, yaw0, surge0, roll0, pitch0 = block_setpoint(args.protocol, blocks[0], args.surge)
sp_pub.publish(setpoint(depth0, yaw0, surge0, roll0, pitch0))
cli = node.create_client(SetBool, NS + 'controller/set')
cli.wait_for_service(timeout_sec=10)
fut = cli.call_async(SetBool.Request(data=True))
rclpy.spin_until_future_complete(node, fut, timeout_sec=5)
print('controller enabled:', fut.result().success if fut.result() else 'no response', flush=True)

dt, rows, failed = 0.1, [], None
steps = int(round(block_seconds / dt))
next_t = time.time()
for b, blk in enumerate(blocks):
    mode = blk[0]
    depth, yaw, surge, roll_sp, pitch_sp = block_setpoint(args.protocol, blk, args.surge)
    for k in range(steps):
        sp_pub.publish(setpoint(depth, yaw, surge, roll_sp, pitch_sp))   # re-sent every step (mvp has a setpoint timeout)
        next_t += dt
        spin_for(max(next_t - time.time(), 0.0))
        e_u = surge - st['surge']
        rows.append({
            't': (b * steps + k) * dt, 'block': b, 'mode': mode,
            'sp_depth': depth, 'sp_yaw': yaw, 'sp_surge': surge, 'sp_roll': roll_sp, 'sp_pitch': pitch_sp,
            'depth': st['depth'], 'roll': st['roll'], 'pitch': st['pitch'], 'yaw': st['yaw'],
            'surge': st['surge'], 'sway': st['sway'],
            **{f'thr{i}': float(v) for i, v in enumerate(st['thr'])},
            **{f'servo{i}': float(v) for i, v in enumerate(st['servo'])},
            'reward': '',
            # absolute errors to the commanded setpoints, as info['abs_err'] in the RL evaluation
            'err_depth': abs(depth - st['depth']), 'err_surge': abs(e_u), 'err_surge_cmd': abs(e_u),
            'err_sway': abs(st['sway']), 'err_roll': abs(wrap(roll_sp - st['roll'])),
            'err_pitch': abs(wrap(pitch_sp - st['pitch'])), 'err_yaw': abs(wrap(yaw - st['yaw'])),
        })
        if not (0.0 <= st['depth'] <= 28.0) or abs(st['roll']) > 1.4 or abs(st['pitch']) > 1.4:
            failed = f"left the envelope in block {b} ({mode}) at t={rows[-1]['t']:.1f}s"
            break
    rb = [x for x in rows if x['block'] == b]
    print(f"block {b} {mode:13s} depth {depth:4.1f} yaw {np.degrees(yaw):+4.0f} surge {surge:+.2f} roll {np.degrees(roll_sp):+3.0f} "
          f"pitch {np.degrees(pitch_sp):+3.0f}: depth err {np.mean([x['err_depth'] for x in rb]):.3f} m, "
          f"yaw err {np.degrees(np.mean([x['err_yaw'] for x in rb])):.1f} deg, surge err {np.mean([x['err_surge'] for x in rb]):.3f}",
          flush=True)
    if failed:
        print(failed)
        break

fut = cli.call_async(SetBool.Request(data=False))   # leave the sim with the controller off
rclpy.spin_until_future_complete(node, fut, timeout_sec=5)

with open(os.path.join(args.out, 'fig4_steps.json'), 'w') as f:
    json.dump({'rows': rows, 'failed': failed, 'model': f'mvp_control ({args.mode}, IROS 2025 QP + PID)',
               'surge': args.surge, 'protocol': args.protocol, 'block_names': [blk[0] for blk in blocks],
               'block_seconds': block_seconds, 'action_mode': 'mvp_qp', 'num_servos': 2, 'surge_mode': None},
              f, default=float)
with open(os.path.join(args.out, 'fig4_steps.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def metrics(rs):
    thr = np.array([[x[f'thr{i}'] for i in range(4)] for x in rs])
    sv = np.array([[x[f'servo{i}'] for i in range(2)] for x in rs])
    return {'rmse_depth_m': rms([x['err_depth'] for x in rs]), 'rmse_yaw_rad': rms([x['err_yaw'] for x in rs]),
            'rmse_surge_ms': rms([x['err_surge'] for x in rs]),
            'rmse_roll_deg': float(np.degrees(rms([x['err_roll'] for x in rs]))),
            'rmse_pitch_deg': float(np.degrees(rms([x['err_pitch'] for x in rs]))),
            'max_abs_roll_deg': float(np.degrees(np.max(np.abs([x['roll'] for x in rs])))),
            'max_abs_pitch_deg': float(np.degrees(np.max(np.abs([x['pitch'] for x in rs])))),
            'energy_sum_thr2_dt': float(np.sum(thr ** 2) * dt),
            'servo_travel_rad': float(np.sum(np.abs(np.diff(sv, axis=0))))}


with open(os.path.join(args.out, 'fig4_metrics.csv'), 'w', newline='') as f:
    groups = [('all', rows)] + [(f'{b}_{blocks[b][0]}', [x for x in rows if x['block'] == b])
                                 for b in sorted({x['block'] for x in rows})]
    keys = list(metrics(rows).keys())
    w = csv.writer(f)
    w.writerow(['segment'] + keys)
    for name, rs in groups:
        mm = metrics(rs)
        w.writerow([name] + [f'{mm[k]:.4f}' for k in keys])
print("overall:", {k: round(v, 3) for k, v in metrics(rows).items()}, "| failed:", failed)

from plot_fig4 import plot
print('wrote', plot(os.path.join(args.out, 'fig4_steps.json'), label=f'IROS 2025 baseline: mvp_control {args.mode}'))
