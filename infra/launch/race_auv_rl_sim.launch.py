"""RACE AUV simulation for RL training/evaluation (started by `RACE2_VEHICLE=race_auv race2.sh sim`).

Same pieces as race_auv_bringup/bringup_simulation.launch.py (GSO-soslab/race_auv jazzy-devel-perception), minus mission,
C2, joystick, AprilTag and RViz, with the arguments race2's bringup has:
  render:=false   stonefish_simulator_nogpu (no rendering, batch training)
  rl_mode:=true   no mvp_control (the policy drives the thrusters); false = IROS baseline mvp_control drives them
  scenario:=...   Stonefish world; default infra/stonefish/race_auv_rl.scn (open water, as race2_test.scn)
Thruster topics are the hardware's: /race_auv/control/thruster/<name> (thruster_driver_node -> Stonefish).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

ROBOT = 'race_auv'
HERE = os.path.dirname(os.path.realpath(__file__))


def launch_setup(context):
    from launch.substitutions import LaunchConfiguration
    render = LaunchConfiguration('render').perform(context).lower() in ('true', '1')
    rl_mode = LaunchConfiguration('rl_mode').perform(context).lower() in ('true', '1')
    scenario = LaunchConfiguration('scenario').perform(context)
    bringup = get_package_share_directory(ROBOT + '_bringup')
    data = os.path.join(get_package_share_directory('world_of_stonefish'), 'data/')
    params = os.path.join(bringup, 'config', 'sim_params.yaml')
    include = lambda *path: IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(bringup, 'launch', *path)))
    if not render:   # stonefish_simulator_nogpu cannot run cameras: temp copies of vehicle + world without them
        import re
        import tempfile
        vehicle = os.path.join(get_package_share_directory('world_of_stonefish'), 'vehicles', ROBOT + '.scn')
        tmp = tempfile.mkdtemp(prefix=ROBOT + '_rl_')
        veh_txt, n_cam = re.subn(r'<sensor\b[^>]*type="camera"[^>]*>.*?</sensor>\s*', '', open(vehicle).read(), flags=re.S)
        open(os.path.join(tmp, ROBOT + '.scn'), 'w').write(veh_txt)
        world_txt, n_inc = re.subn(r'\$\(find world_of_stonefish\)/vehicles/' + ROBOT + r'\.scn',
                                   os.path.join(tmp, ROBOT + '.scn'), open(scenario).read())
        assert n_cam >= 1 and n_inc == 1, (n_cam, n_inc)
        scenario = os.path.join(tmp, 'world.scn')
        open(scenario, 'w').write(world_txt)

    sim = (Node(package='stonefish_ros2', executable='stonefish_simulator', name='stonefish_simulator',
                arguments=[data, scenario, '100', '1200', '800', 'high'], parameters=[params]) if render else
           Node(package='stonefish_ros2', executable='stonefish_simulator_nogpu', name='stonefish_simulator',
                arguments=[data, scenario, '100'], parameters=[params]))
    nodes = [
        sim,
        Node(package='world_of_stonefish', executable='thruster_driver_node', namespace=ROBOT,
             name='thruster_driver_node', parameters=[params]),
        Node(package='world_of_stonefish', executable='imu_driver_node', namespace=ROBOT, name='imu_driver_node',
             remappings=[('imu_in/data', 'imu/stonefish/data'), ('imu_out/data', 'ekf/imu/data')],
             parameters=[{'frame_id': ROBOT + '/imu_sf'}, params]),
        Node(package='world_of_stonefish', executable='dvl_driver_node', namespace=ROBOT, name='dvl_driver_node',
             parameters=[params], remappings=[('/race_auv/dvl/twist', '/race_auv/dvl/raw_twist')]),
        Node(package='world_of_stonefish', executable='pressure_sensor_node', namespace=ROBOT,
             name='pressure_sensor_node', remappings=[('depth', 'depth/odometry')],
             parameters=[{'frame_id': ROBOT + '/world'}]),
        include('include', 'description.launch.py'),
        include('include', 'simulation', 'localization_sim.launch.py'),
    ]
    if not rl_mode:
        nodes.append(include('include', 'simulation', 'mvp_control_sim.launch.py'))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('render', default_value='true', description='false: stonefish_simulator_nogpu'),
        DeclareLaunchArgument('rl_mode', default_value='false', description='true: no mvp_control'),
        DeclareLaunchArgument('scenario', default_value=os.path.join(HERE, '..', 'stonefish', 'race_auv_rl.scn')),
        OpaqueFunction(function=launch_setup),
    ])
