"""Evaluation protocols shared by eval_fig4.py (RL policies) and eval_mvp.py (IROS 2025 QP+PID baseline).

Each block: (mode, depth [m], yaw [rad, NED], surge, roll [rad], pitch [rad]).
FIG4: Zhou, Naderi, Yuan (IROS 2025) Fig. 4 sequence; surge is a sign, multiplied by --surge.
ARTIC: combo maneuvers that need vertical thrust at the stern (articulation); surge in m/s.
ATT90: large-attitude protocol for vehicles trained with roll/pitch targets up to +-90 deg (race_auv, RACE2_ATT_REP=rot).
"""
import numpy as np

W, E = -np.pi / 2, np.pi / 2
D = np.radians
FIG4 = [  # (mode, depth, yaw, surge [x --surge], roll, pitch)
    ('dive', 5.0, W, 0, 0, 0), ('forward', 5.0, W, 1, 0, 0), ('forward', 5.0, E, 1, 0, 0), ('forward', 2.5, E, 1, 0, 0),
    ('backward', 2.5, E, -1, 0, 0), ('backward', 2.5, W, -1, 0, 0), ('backward', 5.0, W, -1, 0, 0),
    ('hover', 5.0, W, 0, 0, 0), ('hover', 5.0, E, 0, 0, 0),
]
ARTIC = [  # (mode, depth, yaw, surge [m/s], roll, pitch). Feasible targets (open loop at 6 m: roll <= +-25 deg at
           # full surge thrust, pitch <= +-30 deg with heave differential); depths <= 8 m (v1 models trained 1-8 m)
    ('settle', 5.0, 0.0, 0.0, 0, 0), ('roll+10', 5.0, 0.0, 0.0, D(10), 0), ('roll-10', 5.0, 0.0, 0.0, D(-10), 0),
    ('roll+surge', 5.0, 0.0, 0.5, D(10), 0), ('nose-dn dive', 8.0, 0.0, 0.5, 0, D(-15)),
    ('nose-up climb', 3.0, 0.0, 0.5, 0, D(15)), ('fast dive', 8.0, 0.0, 0.8, 0, 0),
    ('banked turn', 8.0, E, 0.5, D(10), 0), ('level', 5.0, E, 0.0, 0, 0),
]


ATT90 = [  # (mode, depth, yaw, surge [m/s], roll, pitch). race_auv open loop at 6 m: pitch > 85 deg at full heave
           # differential, roll +-41 deg peak / +-28 deg held, so the roll+60 block probes the limit
    ('settle', 6.0, 0.0, 0.0, 0, 0), ('pitch+45', 6.0, 0.0, 0.0, 0, D(45)), ('pitch-45', 6.0, 0.0, 0.0, 0, D(-45)),
    ('pitch+80', 6.0, 0.0, 0.0, 0, D(80)), ('pitch-80', 6.0, 0.0, 0.0, 0, D(-80)), ('roll+30', 6.0, 0.0, 0.0, D(30), 0),
    ('roll-30', 6.0, 0.0, 0.0, D(-30), 0), ('roll+60', 6.0, 0.0, 0.0, D(60), 0),
    ('climb pitch+30', 3.0, 0.0, 0.5, 0, D(30)), ('roll20 pitch20 turn', 6.0, E, 0.3, D(20), D(20)), ('level', 6.0, E, 0.0, 0, 0),
]


PROTOCOLS = {'fig4': (FIG4, 100.0), 'artic': (ARTIC, 60.0), 'att90': (ATT90, 60.0)}   # blocks, default block length [s]


def block_setpoint(protocol, blk, surge_scale):
    """(depth, yaw, surge, roll, pitch) as plain floats (ROS float fields reject ints)."""
    mode, depth, yaw, surge, roll, pitch = blk
    surge = surge * surge_scale if protocol == 'fig4' else surge
    return float(depth), float(yaw), float(surge), float(roll), float(pitch)
