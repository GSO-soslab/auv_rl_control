"""Vehicle definitions for AUVEnv, selected with env RACE2_VEHICLE (default race2, so existing models are unchanged).

Per vehicle:
  ns          ROS namespace (also the Stonefish robot name used for respawn)
  thrusters   thruster names in ACTION ORDER; commands go to /<ns>/control/thruster/<name> in [-1, 1]
  servos      servo names: commands to /<ns>/control/<name> (rad), measured joint <ns>/<name>_joint
  imu         IMU topic (relative to ns)
  roll_without_servos  whether roll can be commanded with fixed thrusters (attitude setpoints)
  gt_roll_offset       roll (rad) that the ground-truth odometry reports for an upright vehicle; the env rotates the
                       ground truth back by it, so upright = roll 0 for every vehicle (respawn is unaffected)
"""
import math
import os

VEHICLES = {
    # race2_auv jazzy-devel + world_of_stonefish world/race2_test.scn: 4 thrusters + 2 articulated surge servos
    'race2': {
        'ns': 'race2_auv',
        'thrusters': ['heave_bow', 'heave_stern', 'surge_port', 'surge_starboard'],
        'servos': ['surge_port_servo', 'surge_starboard_servo'],
        'imu': 'imu/data',
        'roll_without_servos': False,   # heave thrusters on the centerline
        'gt_roll_offset': 0.0,
    },
    # GSO-soslab/race_auv jazzy-devel-perception (the real vehicle) + infra/stonefish/race_auv_rl.scn:
    # 6 fixed thrusters (surge pair tilted 46 deg and toed out 30 deg, heave bow + stern pair, sway stern), no servos.
    # Topic names are the hardware's (pwm_driver), so a trained policy publishes to the same topics on the vehicle.
    'race_auv': {
        'ns': 'race_auv',
        'thrusters': ['heave_bow', 'heave_stern_port', 'heave_stern_stbd', 'surge_port', 'surge_stbd', 'sway_stern'],
        'servos': [],
        'imu': 'imu/stonefish/data',
        'roll_without_servos': True,    # differential stern heave pair
        'gt_roll_offset': math.pi,      # odom sensor reads roll pi when upright (measured 2026-10-29; rates and depth
                                        # rate checked against finite differences); the hull mesh is rotated by pi
    },
}


def get_vehicle():
    name = os.environ.get('RACE2_VEHICLE', 'race2')
    if name not in VEHICLES:
        raise ValueError(f"unknown RACE2_VEHICLE {name!r} (known: {', '.join(VEHICLES)})")
    return name, VEHICLES[name]
