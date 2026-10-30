import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os, time, datetime
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float32, Header
from geometry_msgs.msg import Vector3
from mvp_msgs.msg import ControlProcess
from sensor_msgs.msg import Imu
import yaml
from rclpy.clock import Clock
import random

# Import the coupling reward calculator
from coupling_rewards import CouplingAwareRewardCalculator
from csv_data_manager import CSVDataManager
from vehicles import get_vehicle

REF_DOFS = ('depth', 'yaw', 'pitch', 'roll')   # DOFs with a reference model when RACE2_REF_ALL=1


def rot_zyx(roll, pitch, yaw):
    """Body-to-world rotation from ZYX Euler angles (well defined at any angle; only the inverse is singular)."""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def rot_log(R):
    """Rotation vector (axis * angle, |angle| <= pi) of a rotation matrix."""
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if angle < 1e-6:
        return 0.5 * w
    if np.pi - angle < 1e-4:   # near pi: axis from the symmetric part
        axis = np.sqrt(np.maximum((np.diag(R) + 1.0) / 2.0, 0.0))
        axis *= np.sign(w + (w == 0))
        return angle * axis / max(np.linalg.norm(axis), 1e-9)
    return angle * w / (2.0 * np.sin(angle))

class SetpointManager:
    """Manages setpoint generation for episode-based training"""
    
    def __init__(self, config):
        # Load setpoint configuration from config or use defaults
        setpoint_config = config.get('setpoint', {})
        
        self.pos_z_range = setpoint_config.get('pos_z_range', [1.0, 8.0])
        self.ori_z_range = setpoint_config.get('ori_z_range', [-2.14, 2.14])
        self.ori_y_range = setpoint_config.get('ori_y_range', [-0.1, 0.1])
        self.vel_x_range = setpoint_config.get('vel_x_range', [-0.6, 0.6])
        
        # Fixed values
        ns = get_vehicle()[1]['ns']
        self.frame_id = f"{ns}/world_ned"
        self.child_frame_id = f"{ns}/cg_link"
        self.control_mode = "4dof"
        
        self.current_setpoint = None
        print(f"SetpointManager initialized with ranges:")
        print(f"  pos_z: {self.pos_z_range}")
        print(f"  ori_z: {self.ori_z_range}")
        print(f"  ori_y: {self.ori_y_range}")
        print(f"  vel_x: {self.vel_x_range}")
    
    def generate_new_setpoint(self):
        """Generate a new random setpoint for a new episode"""
        msg = ControlProcess()
        msg.header = Header()
        msg.header.frame_id = self.frame_id
        msg.child_frame_id = self.child_frame_id
        msg.control_mode = self.control_mode
        
        # Generate random values within specified ranges
        msg.position = Vector3(
            x=0.0, 
            y=0.0, 
            z=random.uniform(self.pos_z_range[0], self.pos_z_range[1])
        )
        msg.orientation = Vector3(
            x=3.14, 
            y=random.uniform(self.ori_y_range[0], self.ori_y_range[1]), 
            z=random.uniform(self.ori_z_range[0], self.ori_z_range[1])
        )
        msg.velocity = Vector3(
            x=random.uniform(self.vel_x_range[0], self.vel_x_range[1]), 
            y=0.0, 
            z=0.0
        )
        msg.angular_rate = Vector3(x=0.0, y=0.0, z=0.0)
        
        self.current_setpoint = msg
        return msg
    
    def get_current_setpoint(self):
        """Get the current setpoint"""
        return self.current_setpoint
    
    def get_setpoint_info(self):
        """Get human-readable info about current setpoint"""
        if self.current_setpoint is None:
            return "No setpoint generated yet"
        
        return (f"Setpoint - pos_z: {self.current_setpoint.position.z:.2f}, "
                f"ori_z: {self.current_setpoint.orientation.z:.2f}, "
                f"ori_y: {self.current_setpoint.orientation.y:.2f}, "
                f"vel_x: {self.current_setpoint.velocity.x:.2f}")

class AUVEnvNode(Node):
    """Node to handle ROS2 communications for the AUV environment"""
    
    def __init__(self, config):
        super().__init__('auv_env_node')
        
        self.config = config
        
        self.thruster_size = self.config['environment']['thruster_size']
        servo_size = self.config['environment']['servo_joints_size']
        
        # Initialize setpoint manager
        self.setpoint_manager = SetpointManager(config)
        
        # Initialize state variables
        self.position_state = np.zeros(3)
        self.orientation_state = np.zeros(3)
        self.R = np.eye(3)   # body-to-world rotation (ground truth), for RACE2_ATT_REP=rot
        self.v_state = np.zeros(3)
        self.omega_ref_state = np.zeros(3)
        
        # Initialize error variables
        self.position_err = np.zeros(3)
        self.orientation_err = np.zeros(3)
        self.v_err = np.zeros(3)
        self.omega_ref_err = np.zeros(3)
        self.linear_acceleration = np.zeros(3)

        # Initialize actuator variables
        self.joint_angles_port = 0.0
        self.joint_angles_starboard = 0.0
        self.joint_angles = np.zeros(servo_size)
        
        self.thrust_heave_bow = 0.0
        self.thrust_surge_port = 0.0
        self.thrust_surge_starboard = 0.0
        self.thrust_heave_stern = 0.0
        
        # Create publishers (vehicle from env RACE2_VEHICLE, see vehicles.py; action order = vehicle thruster list, then servos)
        self.vehicle_name, self.vehicle = get_vehicle()
        ns = self.vehicle['ns']
        self.thruster_names, self.servo_names = self.vehicle['thrusters'], self.vehicle['servos']
        self.thruster_pubs = {name: self.create_publisher(Float64, f'/{ns}/control/thruster/{name}', 1)
                              for name in self.thruster_names}
        self.servo_pubs = {name: self.create_publisher(Float64, f'/{ns}/control/{name}', 1) for name in self.servo_names}
        
        # Create setpoint publisher
        self.setpoint_publisher = self.create_publisher(
            ControlProcess,
            f'/{ns}/controller/process/set_point',
            10
        )
        
        self.smoothed_surge_pub = self.create_publisher(Float64, '/race2/smoothed_surge_velocity', 1)

        # Create timer for continuous setpoint publishing
        self.setpoint_publish_rate = 5.0  # 5 Hz
        self.setpoint_timer = self.create_timer(
            1.0 / self.setpoint_publish_rate, 
            self.publish_current_setpoint_callback
        )
        
        # Create subscribers
        self.create_subscription(ControlProcess,  
                                f'/{ns}/controller/process/value',
                                self.state_callback,
                                2)
        
        self.create_subscription(ControlProcess, 
                                f'/{ns}/controller/process/error', 
                                self.error_callback,
                                1)
        
        if self.vehicle_name == 'race2':   # echo of race2's actuator topics for the legacy (non-tracking) reward
            self.create_subscription(Float64, 
                                    '/race2_auv/control/surge_port_servo', 
                                    self.update_joint_port, 
                                    1)
        
            self.create_subscription(Float64, 
                                    '/race2_auv/control/surge_starboard_servo', 
                                    self.update_joint_starboard, 
                                    1)
        
            self.create_subscription(Float64, 
                                    '/race2_auv/control/thruster/heave_bow', 
                                    self.update_thrust_heave_bow, 
                                    1)
        
            self.create_subscription(Float64, 
                                    '/race2_auv/control/thruster/surge_port', 
                                    self.update_thrust_surge_port, 
                                    1)
        
            self.create_subscription(Float64, 
                                    '/race2_auv/control/thruster/surge_starboard', 
                                    self.update_thrust_surge_starboard, 
                                    1)
        
            self.create_subscription(Float64, 
                                    '/race2_auv/control/thruster/heave_stern', 
                                    self.update_thrust_heave_stern, 
                                    1)
        
        self.create_subscription(
            Imu,
            f"/{ns}/{self.vehicle['imu']}",
            self.imu_callback,
            1
        )
        
        # Tracking mode with Stonefish ground-truth state (bypasses EKF + mvp_control) and respawn resets
        self.use_ground_truth = (config.get('tracking_reward') or {}).get('state_source') == 'ground_truth'
        if self.use_ground_truth:
            from nav_msgs.msg import Odometry
            from stonefish_ros2.srv import Respawn
            self.create_subscription(Odometry, f'/{ns}/odometry/ground_truth', self.ground_truth_callback, 5)
            self.respawn_client = self.create_client(Respawn, '/respawn_robot')
            self.Respawn = Respawn
            # Measured servo angles (the servo slews at <= 1 rad/s, so the command is not the state)
            from sensor_msgs.msg import JointState
            self.servo_joint_names = [f'{ns}/{name}_joint' for name in self.servo_names]
            self.servo_measured = np.zeros(len(self.servo_names))
            if self.servo_names:
                self.create_subscription(JointState, f'/{ns}/stonefish/servo/joint_states', self.servo_state_callback, 5)

        # For communication between callbacks and the environment
        self.new_state_available = False
        self.new_error_available = False
        self.last_action_timestamp = 0
        self.last_state_timestamp = 0
        self.last_error_timestamp = 0
        
        # Declare parameters
        self.declare_parameter('max_steps', 500)
        self.max_steps = self.get_parameter('max_steps').value
    
    def publish_current_setpoint_callback(self):
        """Continuously publish the current setpoint """
        if self.setpoint_manager.current_setpoint is not None:
            # Update timestamp and publish
            self.setpoint_manager.current_setpoint.header.stamp = self.get_clock().now().to_msg()
            self.setpoint_publisher.publish(self.setpoint_manager.current_setpoint)
            self.get_logger().debug("Published current setpoint")
    
    def publish_new_setpoint(self):
        """Generate and publish a new setpoint for the episode"""
        setpoint = self.setpoint_manager.generate_new_setpoint()
        setpoint.header.stamp = self.get_clock().now().to_msg()
        
        # Publish the setpoint immediately
        self.setpoint_publisher.publish(setpoint)
        
        # Add some debugging
        self.get_logger().info(f"Published new setpoint: {self.setpoint_manager.get_setpoint_info()}")
        self.get_logger().info(f"Setpoint publisher topic: {self.setpoint_publisher.topic_name}")
        self.get_logger().info(f"Setpoint publisher subscriber count: {self.setpoint_publisher.get_subscription_count()}")
        
        return setpoint
    
    def get_current_setpoint_info(self):
        """Get information about the current setpoint"""
        return self.setpoint_manager.get_setpoint_info()
    
    def error_callback(self, data):
        """Process error updates"""
        self.get_logger().debug("Error callback triggered!")
        self.last_error_timestamp = time.time()
        self.position_err = np.array([data.position.x, data.position.y, data.position.z])
        self.orientation_err = np.array([data.orientation.x, data.orientation.y, data.orientation.z])
        self.v_err = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
        self.omega_ref_err = np.array([data.angular_rate.x, data.angular_rate.y, data.angular_rate.z])
        
        # Smooth surge velocity error (v_err[0])
        if not hasattr(self, 'surge_error_filter'):
            self.surge_error_filter = self.v_err[0]  # Initialize on first call
        
        alpha = 0.65  # Smoothing factor
        self.surge_error_filter = alpha * self.surge_error_filter + (1 - alpha) * self.v_err[0]
        
        # Use smoothed surge error
        smoothed_v_err = self.v_err.copy()
        smoothed_v_err[0] = self.surge_error_filter
        
        # Convert orientation errors to sin/cos representation to avoid angle wrapping issues
        roll_sin_err = np.sin(self.orientation_err[0])
        roll_cos_err = np.cos(self.orientation_err[0])
        pitch_sin_err = np.sin(self.orientation_err[1])
        pitch_cos_err = np.cos(self.orientation_err[1])
        yaw_sin_err = np.sin(self.orientation_err[2])
        yaw_cos_err = np.cos(self.orientation_err[2])
        
        # Update the state error array with the smoothed surge error
        self.state_err = np.concatenate([
            self.position_err[2:3],
            smoothed_v_err[:2],  # Using smoothed surge error here
            np.array([roll_sin_err, roll_cos_err, pitch_sin_err, pitch_cos_err, yaw_sin_err, yaw_cos_err]),
            self.omega_ref_err[:3],
        ])
        
        if hasattr(self, 'last_action_timestamp') and self.last_error_timestamp > self.last_action_timestamp:
            self.new_error_available = True
    
    def servo_state_callback(self, msg):
        for i, name in enumerate(self.servo_joint_names):
            if name in msg.name:
                self.servo_measured[i] = msg.position[msg.name.index(name)]

    def ground_truth_callback(self, msg):
        """Stonefish ground truth: NED position (z = depth), body-frame velocities, ZYX Euler angles."""
        self.gt_msgs = getattr(self, 'gt_msgs', 0) + 1
        p, q, v, w = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist.linear, msg.twist.twist.angular
        q, v, w = self._gt_to_upright(q, v, w)
        roll = np.arctan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y))
        pitch = np.arcsin(np.clip(2 * (q.w * q.y - q.z * q.x), -1.0, 1.0))
        yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.position_state = np.array([p.x, p.y, p.z])
        self.orientation_state = np.array([roll, pitch, yaw])
        self.R = np.array([[1 - 2 * (q.y ** 2 + q.z ** 2), 2 * (q.x * q.y - q.z * q.w), 2 * (q.x * q.z + q.y * q.w)],
                           [2 * (q.x * q.y + q.z * q.w), 1 - 2 * (q.x ** 2 + q.z ** 2), 2 * (q.y * q.z - q.x * q.w)],
                           [2 * (q.x * q.z - q.y * q.w), 2 * (q.y * q.z + q.x * q.w), 1 - 2 * (q.x ** 2 + q.y ** 2)]])
        self.v_state = np.array([v.x, v.y, v.z])
        self.omega_ref_state = np.array([w.x, w.y, w.z])
        self.last_state_timestamp = time.time()

    def _gt_to_upright(self, q, v, w):
        """Rotate ground truth by the vehicle's gt_roll_offset (phi) about body x, so upright = roll 0:
        q_up = q * Rx(phi); body vectors v_up = Rx(phi)^T v."""
        phi = self.vehicle.get('gt_roll_offset', 0.0)
        if not phi:
            return q, v, w
        from types import SimpleNamespace as V
        c, s = np.cos(phi / 2), np.sin(phi / 2)
        q = V(w=q.w * c - q.x * s, x=q.w * s + q.x * c, y=q.y * c + q.z * s, z=q.z * c - q.y * s)
        c, s = np.cos(phi), np.sin(phi)
        rot = lambda u: V(x=u.x, y=c * u.y + s * u.z, z=-s * u.y + c * u.z)
        return q, rot(v), rot(w)

    def state_callback(self, data):
        """Process state updates from sensors"""
        if self.use_ground_truth:
            return
        self.get_logger().debug("State callback triggered!")
        # Add timestamp to the state observation
        self.last_state_timestamp = time.time()
        # Extract state values
        self.position_state = np.array([data.position.x, data.position.y, data.position.z])
        self.orientation_state = np.array([data.orientation.x, data.orientation.y, data.orientation.z])
        self.v_state = np.array([data.velocity.x, data.velocity.y, data.velocity.z])
        self.omega_ref_state = np.array([data.angular_rate.x, data.angular_rate.y, data.angular_rate.z])
        
        # Smooth surge velocity state (v_state[0])
        if not hasattr(self, 'surge_state_filter'):
            self.surge_state_filter = self.v_state[0]  # Initialize on first call
        
        alpha = 0.65  # Same smoothing factor as error callback
        self.surge_state_filter = alpha * self.surge_state_filter + (1 - alpha) * self.v_state[0]
        
        # Use smoothed surge state
        smoothed_v_state = self.v_state.copy()
        smoothed_v_state[0] = self.surge_state_filter
        
        # Publish smoothed surge velocity
        from std_msgs.msg import Float64
        surge_msg = Float64()
        surge_msg.data = float(self.surge_state_filter)
        self.smoothed_surge_pub.publish(surge_msg)
        
        # Convert orientation to sin/cos representation to avoid angle wrapping issues
        roll_sin = np.sin(self.orientation_state[0])
        roll_cos = np.cos(self.orientation_state[0])
        pitch_sin = np.sin(self.orientation_state[1])
        pitch_cos = np.cos(self.orientation_state[1])
        yaw_sin = np.sin(self.orientation_state[2])
        yaw_cos = np.cos(self.orientation_state[2])
        
        # Update current state for RL agent with smoothed surge velocity
        self.current_state = np.concatenate([
            self.position_state[2:3],
            smoothed_v_state[:2],  # Using smoothed surge velocity here
            np.array([roll_sin, roll_cos, pitch_sin, pitch_cos, yaw_sin, yaw_cos]),
        ])
        
        # Flag indicating we have a new state after the last action
        if hasattr(self, 'last_action_timestamp') and self.last_state_timestamp > self.last_action_timestamp:
            self.new_state_available = True
    
    def imu_callback(self, data):
        """Process IMU data including linear accelerations"""
        self.get_logger().debug("IMU callback triggered!")
        
        # Store linear accelerations (NEW)
        self.linear_acceleration[0] = data.linear_acceleration.x
        self.linear_acceleration[1] = data.linear_acceleration.y
        self.linear_acceleration[2] = data.linear_acceleration.z

    def update_joint_port(self, data):
        self.joint_angles_port = data.data
        self.update_joints()

    def update_joint_starboard(self, data):
        self.joint_angles_starboard = data.data
        self.update_joints()

    def update_joints(self):
        self.joint_angles = [self.joint_angles_port, self.joint_angles_starboard]

    def update_thrust_surge_port(self, data):
        self.thrust_surge_port = data.data

    def update_thrust_surge_starboard(self, data):
        self.thrust_surge_starboard = data.data

    def update_thrust_heave_bow(self, data):
        self.thrust_heave_bow = data.data

    def update_thrust_heave_stern(self, data):
        self.thrust_heave_stern = data.data
    
    def publish_action(self, action, num_thrusters, num_servos):
        """Publish thruster commands (vehicle thruster order) and servo angles (normalized -> rad)."""
        thruster_cmds = [float(a) for a in action[:num_thrusters]]
        servo_angles_rad = [self.convert_servo_command_to_radians(a) for a in action[num_thrusters:num_thrusters + num_servos]]
        for name, value in zip(self.thruster_names, thruster_cmds):
            self.thruster_pubs[name].publish(Float64(data=value))
        for name, value in zip(self.servo_names, servo_angles_rad):
            self.servo_pubs[name].publish(Float64(data=float(value)))
        
        # Record action timestamp
        self.last_action_timestamp = time.time()
        
        # Reset the new state flag since we're waiting for a new state after this action
        self.new_state_available = False
        self.new_error_available = False
        
        return thruster_cmds, servo_angles_rad
    
    def convert_servo_command_to_radians(self, normalized_command):
        """
        Convert normalized servo command [-1, 1] to radians within specified min/max range
        """
        # Ensure normalized command is within [-1, 1]
        normalized_command = np.clip(normalized_command, -1.0, 1.0)
        
        min_angle_rad = self.config['environment']['min_servo_angle_rad']
        max_angle_rad = self.config['environment']['max_servo_angle_rad']
        # Map from [-1, 1] to [min_angle_rad, max_angle_rad]
        angle_rad = min_angle_rad + (normalized_command + 1.0) * (max_angle_rad - min_angle_rad) / 2.0
        
        return angle_rad


class AUVEnv(gym.Env):
    """Custom AUV Environment that follows gym interface"""
    
    # def __init__(self):
    def __init__(self, csv_directory=None):
        super(AUVEnv, self).__init__()
        
        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config_sac.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        if 'RACE2_SERVOS' in os.environ:  # 0 = fixed surge thrusters, 2 = articulated (servo angles in the action)
            self.config['environment']['servo_joints_size'] = int(os.environ['RACE2_SERVOS'])
        if 'RACE2_SERVO_LIMIT_DEG' in os.environ:  # azimuth angle limit, e.g. 45 / 90 / 135
            lim = np.deg2rad(float(os.environ['RACE2_SERVO_LIMIT_DEG']))
            self.config['environment']['min_servo_angle_rad'], self.config['environment']['max_servo_angle_rad'] = -lim, lim
        # Servo action (end-to-end, no model-based allocation): 'naive' = absolute angles,
        # 'rate' = angle step a*w*dt from the measured angle (always reachable within one control step)
        self.action_mode = os.environ.get('RACE2_ACTION', self.config['environment'].get('action_mode', 'naive'))
        if self.action_mode not in ('naive', 'rate'):
            raise ValueError(f"unknown action_mode {self.action_mode!r} (supported: naive, rate)")
        
        # Initialize ROS2 if not already done
        if not rclpy.ok():
            rclpy.init(args=None)
        
        self.csv_mode = csv_directory is not None
        
        if self.csv_mode:
            print("Initializing AUVEnv in CSV mode...")
            self.csv_manager = CSVDataManager(csv_directory, self.config)
            self.node = None  # No ROS2 node needed
        else:
            print("Initializing AUVEnv in online ROS2 mode...")
            # Initialize ROS2 if not already done
            if not rclpy.ok():
                rclpy.init(args=None)
            
            # Create ROS2 node
            self.node = AUVEnvNode(self.config)
            self.csv_manager = None
        
        # Set up action and observation spaces
        thruster_size = self.config['environment']['thruster_size']
        servo_size = self.config['environment']['servo_joints_size']
        self.vehicle_name, vehicle = get_vehicle()
        if not self.csv_mode:   # online: action layout from the vehicle definition (race2: 4 thrusters, as in the config)
            thruster_size = self.config['environment']['thruster_size'] = len(vehicle['thrusters'])
            if servo_size not in (0, len(vehicle['servos'])):
                raise ValueError(f"{self.vehicle_name} has {len(vehicle['servos'])} servos, RACE2_SERVOS={servo_size}")
        self.roll_authority = bool(servo_size) or vehicle['roll_without_servos']
        self.num_thrusters = thruster_size
        self.num_servos = servo_size
        
        self.action_space = spaces.Box(
            low=-1, 
            high=1, 
            shape=(thruster_size + servo_size,), 
            dtype=np.float32
        )
        
        # Tracking reward (config section `tracking_reward`): errors are computed here from the
        # setpoint and controller/process/value, because mvp_control only fills
        # controller/process/error while its PID is enabled (it is disabled for RL training).
        self.tracking_cfg = self.config.get('tracking_reward')
        self.tracking_mode = self.tracking_cfg is not None and not self.csv_mode
        # Surge tracking: 'instant' (per-step |u* - u|) or 'horizon' (reference model + virtual-target
        # displacement error, i.e. velocity tracked over time); env RACE2_SURGE_REWARD overrides
        self.surge_mode = os.environ.get('RACE2_SURGE_REWARD', (self.tracking_cfg or {}).get('surge_mode', 'instant'))
        # Task extensions (config tracking_reward.depth_invariant / attitude_setpoints), off by default
        self.depth_inv = self.tracking_mode and os.environ.get('RACE2_DEPTH_INV') == '1'
        self.att_sp = self.tracking_mode and os.environ.get('RACE2_ATT_SP') == '1'
        if self.depth_inv:
            di = self.tracking_cfg['depth_invariant']
            self.config['setpoint']['pos_z_range'] = di['pos_z_range']
            for k in ('spawn_depth', 'depth_bounds', 'terminate_depth'):
                self.tracking_cfg[k] = di[k]
        if self.depth_inv and 'RACE2_DEPTH_MAX' in os.environ:
            self.config['setpoint']['pos_z_range'] = [self.config['setpoint']['pos_z_range'][0], float(os.environ['RACE2_DEPTH_MAX'])]
        if self.att_sp:
            att = self.tracking_cfg['attitude_setpoints']
            for dof, var in (('roll', 'RACE2_ROLL_MAX_DEG'), ('pitch', 'RACE2_PITCH_MAX_DEG')):
                if var in os.environ:
                    att[f'{dof}_range_deg'] = [-float(os.environ[var]), float(os.environ[var])]
            self.tracking_cfg['terminate_roll_pitch'] = att['terminate_roll_pitch']
            for dof in ('pitch', 'roll') if self.roll_authority else ('pitch',):   # roll needs servos or off-axis heave thrusters
                self.tracking_cfg['dofs'][dof] = [att['weight'], self.tracking_cfg['dofs'][dof][1]]
        if self.tracking_mode and 'RACE2_K_SLEW' in os.environ:
            self.tracking_cfg['k_slew'] = float(os.environ['RACE2_K_SLEW'])
        # Reference model on depth/yaw/pitch/roll (config tracking_reward.reference_model), off by default
        self.ref_all = self.tracking_mode and os.environ.get('RACE2_REF_ALL') == '1'
        # Leaky error integral on depth/yaw/pitch/roll (config tracking_reward.error_integral), off by default
        self.integral = self.tracking_mode and os.environ.get('RACE2_INTEGRAL') == '1'
        # Attitude representation (env RACE2_ATT_REP): 'euler' (default) = wrapped roll/pitch/yaw differences;
        # 'rot' = rotation-based, continuous at any attitude (needed for roll/pitch targets near 90 deg):
        #   error = rotation vector of R^T R_d (body frame; ~ roll/pitch/yaw errors when small),
        #   observation = first two columns of R^T R_d + gravity direction in the body frame (no Euler angles)
        self.att_rot = self.tracking_mode and os.environ.get('RACE2_ATT_REP', 'euler') == 'rot'
        if os.environ.get('RACE2_ATT_REP', 'euler') not in ('euler', 'rot'):
            raise ValueError(f"unknown RACE2_ATT_REP {os.environ['RACE2_ATT_REP']!r} (euler | rot)")
        n_obs = 24 + (thruster_size + 3 * servo_size if self.tracking_mode else 0)  # + prev action, sin/cos servo angles
        if self.att_rot:
            n_obs -= 3   # 12 Euler sin/cos -> 6 (R^T R_d columns) + 3 (gravity in body)
        # 'integral_obs' (ablation, Sim2Swim-style): observe the velocity-error integral, reward stays instantaneous
        self.surge_obs_extra = self.tracking_mode and self.surge_mode in ('horizon', 'integral_obs')
        if self.surge_obs_extra:
            n_obs += 3  # u_ref, du_ref/dt, along-track displacement error (normalized)
        if self.ref_all:
            n_obs += 2 * len(REF_DOFS)  # reference rates + remaining offset target - reference
        if self.integral:
            n_obs += len(REF_DOFS)      # error integrals, normalized by their clamp

        # Update observation space to include angular rates and all velocities
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(n_obs,),  # 24 base + (tracking mode) previous action and servo angles
            dtype=np.float32
        )
        
        self.u_ref, self.u_ref_dot, self.s_err = 0.0, 0.0, 0.0   # horizon surge tracking state
        self.ref = {d: [0.0, 0.0] for d in REF_DOFS}             # [x_ref, x_ref'] per DOF (RACE2_REF_ALL)
        self.err_int = {d: 0.0 for d in REF_DOFS}                 # leaky error integrals (RACE2_INTEGRAL)

        # Initialize episode-related variables
        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_count = 0  # Track total episodes
        
        # Initialize history arrays for smoothness calculations
        self.joint_positions_history = np.zeros((10, self.num_servos))  # Store last 10 servo positions
        self.u_prev = np.zeros((100, self.num_thrusters))  # Store last 10 thruster commands
        self.thruster_command_action_prev = np.zeros((100, self.num_thrusters))  # Store last 10 thruster actions
        
        # Initialize tracking variables
        self.thruster_action = np.zeros(self.num_thrusters)
        self.joint_angles = np.zeros(self.num_servos)
        self.last_action = np.zeros(thruster_size + servo_size)
        
        # Initialize coupling-aware reward calculator
        self.coupling_calculator = None
        
        # Store episode setpoint info for logging
        self.current_episode_setpoint_info = "No setpoint set yet"
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state and return the initial observation"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        if self.tracking_mode and self.node.use_ground_truth:
            return self._reset_tracking(options or {})
        
        self.episode_step = 0
        self.episode_reward = 0
        self.episode_count += 1
        
        if self.csv_mode:
            episode_info = self.csv_manager.reset_episode(episode_length=500)
            self.current_episode_setpoint_info = f"CSV Episode {self.episode_count} (start: {episode_info['start_timestamp']:.1f}s)"
            print(f"Episode {self.episode_count} started - CSV data from {episode_info['start_timestamp']:.1f}s")
            
            initial_observation = self.csv_manager._get_current_state_observation()
            
        else:
            setpoint = self.node.publish_new_setpoint()
            self.current_episode_setpoint_info = self.node.get_current_setpoint_info()
            
            print(f"Episode {self.episode_count} started - {self.current_episode_setpoint_info}")
            
            # time.sleep(0.05)
            # self._spin_node(timeout_sec=0.3)
            
            # setpoint = self.node.publish_new_setpoint()
            # time.sleep(0.01)
            # self._spin_node(timeout_sec=0.3)
            
            time.sleep(0.1)           # Was: 0.05
            self._spin_node(timeout_sec=0.1)    # Was: 0.3

            setpoint = self.node.publish_new_setpoint()
            time.sleep(0.05)          # Was: 0.01  
            self._spin_node(timeout_sec=0.1)    # Was: 0.3

            depth = self.node.position_state[2:3]
            depth_error = self.node.position_err[2:3]
            surge_velocity_error = self.node.v_err[0:1]
            sway_velocity_error = self.node.v_err[1:2]
            
            roll_sin_err = np.array([np.sin(self.node.orientation_err[0])])
            roll_cos_err = np.array([np.cos(self.node.orientation_err[0])])
            pitch_sin_err = np.array([np.sin(self.node.orientation_err[1])])
            pitch_cos_err = np.array([np.cos(self.node.orientation_err[1])])
            yaw_sin_err = np.array([np.sin(self.node.orientation_err[2])])
            yaw_cos_err = np.array([np.cos(self.node.orientation_err[2])])
            
            roll_sin_current = np.array([np.sin(self.node.orientation_state[0])])
            roll_cos_current = np.array([np.cos(self.node.orientation_state[0])])
            pitch_sin_current = np.array([np.sin(self.node.orientation_state[1])])
            pitch_cos_current = np.array([np.cos(self.node.orientation_state[1])])
            yaw_sin_current = np.array([np.sin(self.node.orientation_state[2])])
            yaw_cos_current = np.array([np.cos(self.node.orientation_state[2])])
            
            surge_velocity = self.node.v_state[0:1]
            sway_velocity = self.node.v_state[1:2]
            heave_velocity = self.node.v_state[2:3]
            
            roll_rate = self.node.omega_ref_state[0:1]
            pitch_rate = self.node.omega_ref_state[1:2]
            yaw_rate = self.node.omega_ref_state[2:3]
            
            x_acceleration = self.node.linear_acceleration[0:1]
            y_acceleration = self.node.linear_acceleration[1:2]
            
            initial_observation = np.concatenate([
                depth,
                depth_error,
                surge_velocity_error,
                sway_velocity_error,
                roll_sin_err,
                roll_cos_err,
                pitch_sin_err,
                pitch_cos_err,
                yaw_sin_err,
                yaw_cos_err,
                roll_sin_current,
                roll_cos_current,
                pitch_sin_current,
                pitch_cos_current,
                yaw_sin_current,
                yaw_cos_current,
                surge_velocity,
                sway_velocity,
                heave_velocity,
                roll_rate,
                pitch_rate,
                yaw_rate,
                x_acceleration,
                y_acceleration,
            ])

        if self.tracking_mode:
            self.last_action = np.zeros(self.num_thrusters + self.num_servos)
            initial_observation = self._tracking_observation(self._tracking_errors())

        if self.coupling_calculator is None:
            from coupling_rewards import CouplingAwareRewardCalculator
            self.coupling_calculator = CouplingAwareRewardCalculator(self.config)

        info = {
            'episode_count': self.episode_count,
            'setpoint_info': self.current_episode_setpoint_info,
            'mode': 'csv' if self.csv_mode else 'online'
        }
        
        if self.csv_mode:
            info['csv_episode_info'] = episode_info
            stats = self.csv_manager.get_dataset_stats()
            info['dataset_stats'] = stats
        else:
            info['setpoint_values'] = {
                'pos_z': setpoint.position.z,
                'ori_z': setpoint.orientation.z,
                'ori_y': setpoint.orientation.y,
                'vel_x': setpoint.velocity.x
            }
        
        return initial_observation, info
    
    def step(self, action):
        """Execute action in the environment and return next state, reward, termination flag, etc."""
        
        if self.csv_mode:
            return self._step_csv_mode(action)
        else:
            return self._step_online_mode(action)

    def _step_csv_mode(self, action):
        """Handle step in CSV mode using recorded PID actions"""
        
        # Advance to next step in synchronized data
        episode_done, step_info = self.csv_manager.step_episode()
        
        # Get current observation from CSV data
        observation = self.csv_manager._get_current_state_observation()
        
        # Get state error array for reward calculation
        state_error_array = self.csv_manager.get_state_error_array()
        
        #Use recorded PID actions from CSV
        recorded_pid_action = self.csv_manager.get_recorded_action_at_current_step()
        
        # Store recorded actions for reward calculation
        self.thruster_action = recorded_pid_action[:4]  # PID thruster commands
        self.joint_angles = recorded_pid_action[4:6] if len(recorded_pid_action) > 4 else np.zeros(2)  # PID servo commands
        self.last_action = recorded_pid_action.copy()  # PID action, not agent action
        
        # Create mock node with recorded PID actions for reward calculation
        if not hasattr(self, '_mock_node'):
            self._mock_node = type('MockNode', (), {})()
        
        self._mock_node.thrust_heave_bow = recorded_pid_action[0]      # What PID actually did
        self._mock_node.thrust_heave_stern = recorded_pid_action[1]    # What PID actually did
        self._mock_node.thrust_surge_port = recorded_pid_action[2]     # What PID actually did
        self._mock_node.thrust_surge_starboard = recorded_pid_action[3] # What PID actually did
        
        # Use mock node for reward calculation
        original_node = self.node
        self.node = self._mock_node
        
        # Calculate reward based on PID's performance
        reward = self.calculate_reward(state_error_array)
        if isinstance(reward, np.ndarray):
            reward = float(reward.item())
        
        # Restore original node reference
        self.node = original_node
        
        # Store raw reward
        if not hasattr(self, 'episode_rewards'):
            self.episode_rewards = []
        self.episode_rewards.append(reward)
        
        self.episode_reward += reward
        
        # Check termination conditions
        terminated = episode_done
        truncated = False
        
        max_steps = getattr(self, 'max_steps', 500)
        if self.episode_step >= max_steps:
            truncated = True
            
        if step_info.get('reason') == 'data_complete':
            terminated = True
        
        self.episode_step += 1
        
        info = {
            'episode_step': self.episode_step,
            'episode_reward': self.episode_reward, 
            'setpoint_info': self.current_episode_setpoint_info,
            'csv_step_info': step_info,
            'simulation_time': step_info.get('timestamp', self.episode_step * 0.1),
            'data_frequency': '10Hz (0.1s per step)',
            'recorded_action': recorded_pid_action,  # The PID action that was actually used
            'agent_action': action  # The action the agent wanted to take (for comparison/analysis)
        }
        
        return observation, reward, terminated, truncated, info

    
    # ---------------------------------------------------------------- tracking reward mode
    @staticmethod
    def _wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def _tracking_errors(self):
        """Setpoint minus state, angles wrapped to [-pi, pi)."""
        sp, n = self.node.setpoint_manager.current_setpoint, self.node
        return {
            'depth': sp.position.z - n.position_state[2],
            'surge': (self.u_ref if self.surge_obs_extra else sp.velocity.x) - n.v_state[0],
            'surge_cmd': sp.velocity.x - n.v_state[0],          # error to the raw command (for logging)
            's_err': self.s_err if self.surge_obs_extra else 0.0,
            'sway': sp.velocity.y - n.v_state[1],
            **(dict(zip(('roll', 'pitch', 'yaw'), self._att_error(sp.orientation.x, sp.orientation.y, sp.orientation.z)))
               if self.att_rot else
               {'roll': self._wrap(sp.orientation.x - n.orientation_state[0]),
                'pitch': self._wrap(sp.orientation.y - n.orientation_state[1]),
                'yaw': self._wrap(sp.orientation.z - n.orientation_state[2])}),
            **({f'{d}_ref': self._ref_error(d) for d in REF_DOFS} if self.ref_all else {}),
        }

    def _att_error(self, roll_d, pitch_d, yaw_d):
        """Rotation-based attitude error (RACE2_ATT_REP=rot): rotation vector of R^T R_d, i.e. the body-frame rotation
        from the current to the desired attitude. Stores R^T R_d for the observation."""
        self._R_err = self.node.R.T @ rot_zyx(roll_d, pitch_d, yaw_d)
        return rot_log(self._R_err)

    def _tracking_observation(self, e):
        n, o = self.node, self.node.orientation_state
        servo = (n.servo_measured if n.use_ground_truth else np.asarray(n.joint_angles, dtype=float))[:self.num_servos]
        r = (lambda d: e[f'{d}_ref']) if self.ref_all else (lambda d: e[d])   # error to the reference, or to the target
        depth, e_depth = n.position_state[2], r('depth')
        if self.depth_inv:   # same layout, but no absolute depth below the surface region and a saturated error
            di = self.tracking_cfg['depth_invariant']
            depth, e_depth = min(depth, di['surface_feature']), float(np.clip(e_depth, -di['error_clip'], di['error_clip']))
        return np.concatenate([
            [depth, e_depth, e['surge'], e['sway']],
            *([self._R_err[:, 0], self._R_err[:, 1], n.R[2, :]] if self.att_rot else   # error rotation, gravity in body
              [[np.sin(r('roll')), np.cos(r('roll')), np.sin(r('pitch')), np.cos(r('pitch')),
                np.sin(r('yaw')), np.cos(r('yaw'))],
               [np.sin(o[0]), np.cos(o[0]), np.sin(o[1]), np.cos(o[1]), np.sin(o[2]), np.cos(o[2])]]),
            n.v_state, n.omega_ref_state, n.linear_acceleration[:2],
            self.last_action,                    # previous action, so the rate penalty is Markov
            np.sin(servo), np.cos(servo),        # servo angles (empty without servos)
            [self.u_ref, self.u_ref_dot, self.s_err / self.tracking_cfg.get('surge_horizon', {}).get('s_max', 2.0)]
            if self.surge_obs_extra else [],
            self._reference_observation() if self.ref_all else [],
            [self.err_int[d] / self.tracking_cfg['error_integral'][d]['max'] for d in REF_DOFS] if self.integral else [],
        ]).astype(np.float32)

    def _curriculum_stage(self):
        """1: depth only (hold spawn heading, zero surge); 2: + surge; 3: + yaw. Env var RACE2_STAGE overrides."""
        return int(os.environ.get('RACE2_STAGE', self.tracking_cfg.get('stage', 1)))

    def _sample_setpoint(self, sp, stage, hold_yaw):
        """Random targets: depth always; surge from stage 2 (hover_prob of segments at 0); yaw from stage 3."""
        sp_cfg, c = self.config.get('setpoint', {}), self.tracking_cfg
        sp.position.z = random.uniform(*sp_cfg.get('pos_z_range', [1.0, 8.0]))
        sp.orientation.x, sp.orientation.y = 0.0, 0.0
        if self.att_sp and random.random() >= c['attitude_setpoints']['level_prob']:
            att = c['attitude_setpoints']
            sp.orientation.y = float(np.radians(random.uniform(*att['pitch_range_deg'])))
            if self.roll_authority:
                sp.orientation.x = float(np.radians(random.uniform(*att['roll_range_deg'])))
        sp.orientation.z = random.uniform(-np.pi, np.pi) if stage >= 3 else hold_yaw
        hover = random.random() < c.get('hover_prob', 0.0)
        sp.velocity.x = 0.0 if (stage < 2 or hover) else random.uniform(*sp_cfg.get('vel_x_range', [-1.0, 1.0]))

    def _reset_tracking(self, options=None):
        """options (evaluation): spawn_depth, spawn_yaw to override the random respawn pose."""
        c, n = self.tracking_cfg, self.node
        self.episode_step, self.episode_reward = 0, 0.0
        self.episode_count += 1
        self.last_action = np.zeros(self.num_thrusters + self.num_servos)
        n.publish_action(self.last_action, self.num_thrusters, self.num_servos)  # stop thrusters before respawn

        # Respawn at a random depth and heading
        options = options or {}
        z0 = options.get('spawn_depth', random.uniform(*c.get('spawn_depth', [1.0, 6.0])))
        yaw0 = options.get('spawn_yaw', random.uniform(-np.pi, np.pi))
        req = n.Respawn.Request()
        req.name = n.vehicle['ns']
        req.origin.position.z = z0
        req.origin.orientation.z, req.origin.orientation.w = np.sin(yaw0 / 2), np.cos(yaw0 / 2)
        if not n.respawn_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("/respawn_robot not available")
        fut = n.respawn_client.call_async(req)
        rclpy.spin_until_future_complete(n, fut, timeout_sec=5.0)
        self._spin_node(timeout_sec=c.get('settle_seconds', 1.0))
        if n.use_ground_truth and not getattr(n, 'gt_msgs', 0):   # e.g. the sim runs another vehicle (2026-10-29: race2
            self._spin_node(timeout_sec=5.0)                       # sims for race_auv evals gave all-zero state, silently)
            if not getattr(n, 'gt_msgs', 0):
                raise RuntimeError(f"no ground truth on /{n.vehicle['ns']}/odometry/ground_truth: is the sim running "
                                   f"vehicle {n.vehicle_name} (RACE2_VEHICLE) on ROS domain {os.environ.get('ROS_DOMAIN_ID')}?")

        # Curriculum setpoint (roll/pitch targets level; ground-truth frame has roll = 0 upright)
        stage = self._curriculum_stage()
        sp = n.setpoint_manager.generate_new_setpoint()
        self._sample_setpoint(sp, stage, hold_yaw=yaw0)
        n.setpoint_publisher.publish(sp)
        self.current_episode_setpoint_info = f"stage {stage} | " + n.get_current_setpoint_info()

        self.max_steps = int(c.get('episode_seconds', 60) / c.get('control_dt', 0.1))
        self.u_ref, self.u_ref_dot, self.s_err = float(n.v_state[0]), 0.0, 0.0
        self.ref = {d: [float(self._measured(d)), 0.0] for d in REF_DOFS}   # references start at the current state
        self.err_int = {d: 0.0 for d in REF_DOFS}
        self.next_step_time = time.time()
        obs = self._tracking_observation(self._tracking_errors())
        return obs, {'episode_count': self.episode_count, 'setpoint_info': self.current_episode_setpoint_info,
                     'mode': 'online-ground-truth', 'stage': stage}

    def _servo_rate_to_angle(self, a):
        """Rate servo action: a in [-1, 1] -> angle step a*w*dt from the measured angle, clipped to the limits.
        The sim servo slews at exactly w (measured), so every command is reached within one control step.
        Returns the naive-layout command [thrusts, normalized servo angles]."""
        env_cfg = self.config['environment']
        lo, hi = env_cfg['min_servo_angle_rad'], env_cfg['max_servo_angle_rad']
        w_dt = env_cfg.get('servo_max_velocity', 1.0) * self.tracking_cfg.get('control_dt', 0.1)
        nt = self.num_thrusters
        beta = np.clip(np.asarray(self.node.servo_measured[:self.num_servos]) + np.asarray(a[nt:]) * w_dt, lo, hi)
        return np.concatenate([a[:nt], 2.0 * (beta - lo) / (hi - lo) - 1.0])

    def _measured(self, dof):
        n = self.node
        return n.position_state[2] if dof == 'depth' else n.orientation_state[('roll', 'pitch', 'yaw').index(dof)]

    def _target(self, dof):
        sp = self.node.setpoint_manager.current_setpoint
        return {'depth': sp.position.z, 'roll': sp.orientation.x, 'pitch': sp.orientation.y, 'yaw': sp.orientation.z}[dof]

    def _ref_error(self, dof):
        if self.att_rot and dof != 'depth':   # computed for all three axes at once from the reference attitude
            e = self._att_error(self.ref['roll'][0], self.ref['pitch'][0], self.ref['yaw'][0])
            return float(e[('roll', 'pitch', 'yaw').index(dof)])
        err = self.ref[dof][0] - self._measured(dof)
        return err if dof == 'depth' else self._wrap(err)

    def _update_references(self):
        """Reference model for depth and attitude (one control step), the same form as the surge horizon model:
            x_r'' = wn^2 (x* - x_r) - 2 wn x_r'   (critically damped),   |x_r'| <= v_max,   |x_r''| <= a_max
        A step command becomes a feasible ramp. Angles use the wrapped difference, so the reference turns the short way."""
        dt, cfg = self.tracking_cfg.get('control_dt', 0.1), self.tracking_cfg['reference_model']
        for d in REF_DOFS:
            p, (x, v) = cfg[d], self.ref[d]
            diff = self._target(d) - x
            diff = diff if d == 'depth' else self._wrap(diff)
            acc = float(np.clip(p['wn'] ** 2 * diff - 2.0 * p['wn'] * v, -p['a_max'], p['a_max']))
            v = float(np.clip(v + acc * dt, -p['v_max'], p['v_max']))
            x = x + v * dt
            self.ref[d] = [x if d == 'depth' else self._wrap(x), v]

    def _update_error_integrals(self):
        """Leaky, clamped integral of each DOF's error (to the reference with RACE2_REF_ALL, else to the target):
            I <- clip((I + e dt)(1 - dt/tau), -max, max)
        The same anti-windup integral as surge's s_err; it gives the policy integral action against steady offsets."""
        c = self.tracking_cfg['error_integral']
        dt = self.tracking_cfg.get('control_dt', 0.1)
        e = self._tracking_errors()
        for d in REF_DOFS:
            err = e[f'{d}_ref'] if self.ref_all else e[d]
            self.err_int[d] = float(np.clip((self.err_int[d] + err * dt) * (1 - dt / c['tau']), -c[d]['max'], c[d]['max']))

    def _reference_observation(self):
        """Reference rates (normalized by v_max) and the remaining offset target - reference (depth clipped to +-3 m)."""
        cfg = self.tracking_cfg['reference_model']
        rates = [self.ref[d][1] / cfg[d]['v_max'] for d in REF_DOFS]
        rem = [np.clip(self._target('depth') - self.ref['depth'][0], -3.0, 3.0) / 3.0] + \
              [self._wrap(self._target(d) - self.ref[d][0]) / np.pi for d in REF_DOFS[1:]]
        return rates + rem

    def _update_surge_reference(self):
        """Horizon surge tracking (one control step).

        Reference model (Fossen-style, critically damped 2nd order with acceleration limit):
            u_ref'' = wn^2 (u* - u_ref) - 2 zeta wn u_ref',   |u_ref'| <= a_max
        Virtual target moving at u_ref along the body x-axis; displacement (along-track) error
            s_err += (u_ref - u) dt,  with leak tau and clamp (anti-windup).
        """
        h, dt = self.tracking_cfg.get('surge_horizon', {}), self.tracking_cfg.get('control_dt', 0.1)
        wn, zeta, a_max = h.get('wn', 0.5), h.get('zeta', 1.0), h.get('a_max', 0.15)
        u_cmd = self.node.setpoint_manager.current_setpoint.velocity.x
        if self.surge_mode == 'integral_obs':      # ablation: no reference model, integrate the raw command error
            self.u_ref, self.u_ref_dot = float(u_cmd), 0.0
        else:
            acc = wn ** 2 * (u_cmd - self.u_ref) - 2 * zeta * wn * self.u_ref_dot
            self.u_ref_dot = float(np.clip(self.u_ref_dot + acc * dt, -a_max, a_max))
            self.u_ref = float(self.u_ref + self.u_ref_dot * dt)
        self.s_err = float(np.clip((self.s_err + (self.u_ref - self.node.v_state[0]) * dt) * (1 - dt / h.get('leak_tau', 20.0)),
                                   -h.get('s_max', 2.0), h.get('s_max', 2.0)))

    def _tracking_reward(self, e, action, prev_action):
        """Per-DOF bounded tracking terms in [0, 1] (fine Gaussian + coarse tanh for gradient far away),
        weighted and normalized, minus small action-rate / effort / out-of-bounds penalties."""
        c = self.tracking_cfg
        track, terms = 0.0, {}
        def score(err, sigma):
            return 0.5 * np.exp(-(err / sigma) ** 2) + 0.5 * (1.0 - np.tanh(err / (c.get('coarse_factor', 10) * sigma)))
        for dof, (w, sigma) in c['dofs'].items():
            err = abs(e[f'{dof}_ref'] if self.ref_all and dof in REF_DOFS else e[dof])
            r = score(err, sigma)
            if dof == 'surge' and self.surge_mode == 'horizon':   # velocity vs reference + displacement to the virtual target
                r = 0.5 * r + 0.5 * score(abs(e['s_err']), c.get('surge_horizon', {}).get('sigma_s', 0.25))
            elif self.integral and dof in REF_DOFS:               # same split for the other DOFs: error + its integral
                r = 0.5 * r + 0.5 * score(abs(self.err_int[dof]), c['error_integral'][dof]['sigma'])
            terms[dof] = r
            track += w * r
        track /= sum(w for w, _ in c['dofs'].values())
        # Action-rate penalty; with the rate servo action the servo outputs already are rates (their cost is k_slew)
        dims = slice(0, self.num_thrusters) if self.action_mode == 'rate' else slice(None)
        rate = float(np.mean((action[dims] - prev_action[dims]) ** 2))
        effort = float(np.mean(np.asarray(self.thruster_action, dtype=float) ** 2))  # actual thruster commands
        # Effort weight: constant k_effort, or adaptive (IROS 2025 Eq. 4, as implemented in mvp_control
        # mvp_control.cpp:239) lambda = k / (1 + exp(-a*d)), a > 0: rises from k/2 at zero demand to k at large
        # demand. d = RMS tracking error normalized by 10*sigma (proxy for the requested wrench norm).
        paper = os.environ.get('RACE2_PAPER_TERMS', str(c.get('paper_terms', False))).lower() in ('1', 'true')
        ad = c.get('effort_adaptive') if paper else None
        if ad:
            d = float(np.sqrt(np.mean([(e[k] / (c.get('coarse_factor', 10) * s_)) ** 2 for k, (_, s_) in c['dofs'].items()])))
            k_eff = ad['k'] / (1.0 + np.exp(-ad['a'] * d))
        else:
            k_eff = c.get('k_effort', 0.01)
        # Servo slew (measured angle change per step, normalized by the max step w*dt): energy + counter-torque
        w_dt = self.config['environment'].get('servo_max_velocity', 1.0) * c.get('control_dt', 0.1)
        slew = float(np.mean((self._servo_step / w_dt) ** 2)) if self.num_servos else 0.0
        z = self.node.position_state[2]
        lo, hi = c.get('depth_bounds', [0.3, 10.0])
        oob = 1.0 if (z < lo or z > hi) else 0.0
        yaw_rate = float(self.node.omega_ref_state[2] ** 2)
        k_slew = c.get('k_slew', 0.0) if paper else 0.0
        reward = (track - c.get('k_rate', 0.05) * rate - k_eff * effort - k_slew * slew
                  - c.get('k_oob', 1.0) * oob - c.get('k_yaw_rate', 0.0) * yaw_rate)
        return float(reward), {'track': track, 'rate': rate, 'effort': effort, 'effort_weight': k_eff,
                               'slew': slew, 'oob': oob, 'yaw_rate_sq': yaw_rate, **terms}

    def _step_tracking(self, action):
        action = np.asarray(action, dtype=float)
        prev_action = self.last_action.copy()
        servo_before = np.array(self.node.servo_measured[:self.num_servos]) if self.node.use_ground_truth else None
        cmd = self._servo_rate_to_angle(action) if self.action_mode == 'rate' and self.num_servos else action
        thruster_cmds, servo_angles_rad = self.node.publish_action(cmd, self.num_thrusters, self.num_servos)
        self.thruster_action, self.joint_angles, self.last_action = thruster_cmds, servo_angles_rad, action.copy()

        c = self.tracking_cfg
        if self.node.use_ground_truth:
            # Fixed control period: spin until control_dt after the previous step
            self.next_step_time = max(self.next_step_time + c.get('control_dt', 0.1), time.time())
            while time.time() < self.next_step_time:
                self._spin_node(timeout_sec=min(0.01, max(self.next_step_time - time.time(), 0.0)))
            max_steps = self.max_steps
        else:
            start = time.time()
            while not self.node.new_state_available and time.time() - start < 0.4:
                self._spin_node(timeout_sec=0.05)
            if not self.node.new_state_available:
                print("Warning: Timeout waiting for state update")
            max_steps = self.node.max_steps

        seg = c.get('setpoint_change_seconds', 0)
        if self.node.use_ground_truth and seg and self.episode_step > 0 and \
                self.episode_step % int(round(seg / c.get('control_dt', 0.1))) == 0:
            sp = self.node.setpoint_manager.current_setpoint     # new step command, vehicle keeps going
            self._sample_setpoint(sp, self._curriculum_stage(), hold_yaw=sp.orientation.z)
        if self.surge_obs_extra:
            self._update_surge_reference()
        if self.ref_all:
            self._update_references()
        if self.integral:
            self._update_error_integrals()
        self._servo_step = (np.array(self.node.servo_measured[:self.num_servos]) - servo_before
                            if servo_before is not None else np.zeros(self.num_servos))
        e = self._tracking_errors()
        obs = self._tracking_observation(e)
        reward, terms = self._tracking_reward(e, action, prev_action)

        # Terminate on leaving the depth envelope or large roll/pitch (the next reset respawns the vehicle)
        z, (roll, pitch, _) = self.node.position_state[2], self.node.orientation_state
        lo, hi = c.get('terminate_depth', [0.0, 15.0])
        lim = c.get('terminate_roll_pitch', 1.0)
        if self.att_rot:   # tilt error (angle between actual and desired body z axes; heading ignored) to the
            # reference (or target), instead of |roll|, |pitch| (targets reach 90 deg)
            src = (lambda d: self.ref[d][0]) if self.ref_all else self._target
            z_des = rot_zyx(src('roll'), src('pitch'), src('yaw'))[:, 2]
            tilted = float(np.arccos(np.clip(self.node.R[:, 2] @ z_des, -1.0, 1.0))) > c.get('terminate_att_error', 1.4)
        else:
            tilted = abs(roll) > lim or abs(pitch) > lim
        terminated = bool(self.node.use_ground_truth and (z < lo or z > hi or tilted))
        if terminated:
            reward -= c.get('k_terminate', 10.0)
        terms['terminated'] = float(terminated)

        self.episode_reward += reward
        self.episode_step += 1
        truncated = self.episode_step >= max_steps
        info = {'episode_step': self.episode_step, 'episode_reward': self.episode_reward,
                'setpoint_info': self.current_episode_setpoint_info,
                'abs_err': {k: abs(v) for k, v in e.items()}, 'reward_terms': terms}
        if self.episode_step * c.get('control_dt', 0.1) >= c.get('steady_state_after', 15.0):
            info['abs_err']['surge_steady'] = abs(e['surge_cmd'])   # vs the raw command, excludes the transient
        return obs, reward, terminated, truncated and not terminated, info

    def _step_online_mode(self, action):
        """Execute action in the environment and return next state, reward, termination flag, etc."""
        if self.tracking_mode:
            return self._step_tracking(action)

        state_error_array = None

        thruster_cmds, servo_angles_rad = self.node.publish_action(action, self.num_thrusters, self.num_servos)
        self.thruster_action = thruster_cmds
        self.joint_angles = servo_angles_rad
        self.last_action = action.copy()
        
        timeout_sec = 0.3
        start_time = time.time()
        
        while not (self.node.new_state_available and self.node.new_error_available):
            self._spin_node(timeout_sec=0.28)
            if time.time() - start_time > timeout_sec:
                print("Warning: Timeout waiting for state/error updates")
                break
                
        if self.node.new_state_available and self.node.new_error_available:
            depth = self.node.position_state[2:3]
            depth_error = self.node.position_err[2:3]
            surge_velocity_error = self.node.v_err[0:1]
            sway_velocity_error = self.node.v_err[1:2]

            roll_error = self.node.orientation_err[0:1]
            pitch_error = self.node.orientation_err[1:2]
            yaw_error = self.node.orientation_err[2:3]

            roll_sin_err = np.array([np.sin(self.node.orientation_err[0])])
            roll_cos_err = np.array([np.cos(self.node.orientation_err[0])])
            pitch_sin_err = np.array([np.sin(self.node.orientation_err[1])])
            pitch_cos_err = np.array([np.cos(self.node.orientation_err[1])])
            yaw_sin_err = np.array([np.sin(self.node.orientation_err[2])])
            yaw_cos_err = np.array([np.cos(self.node.orientation_err[2])])
            
            roll_sin_current = np.array([np.sin(self.node.orientation_state[0])])
            roll_cos_current = np.array([np.cos(self.node.orientation_state[0])])
            pitch_sin_current = np.array([np.sin(self.node.orientation_state[1])])
            pitch_cos_current = np.array([np.cos(self.node.orientation_state[1])])
            yaw_sin_current = np.array([np.sin(self.node.orientation_state[2])])
            yaw_cos_current = np.array([np.cos(self.node.orientation_state[2])])
            
            surge_velocity = self.node.v_state[0:1]
            sway_velocity = self.node.v_state[1:2]
            heave_velocity = self.node.v_state[2:3]
            
            roll_rate = self.node.omega_ref_state[0:1]
            pitch_rate = self.node.omega_ref_state[1:2]
            yaw_rate = self.node.omega_ref_state[2:3]
            
            x_acceleration = self.node.linear_acceleration[0:1]
            y_acceleration = self.node.linear_acceleration[1:2]
            
            observation = np.concatenate([
                depth,
                depth_error,
                surge_velocity_error,
                sway_velocity_error,
                roll_sin_err,
                roll_cos_err,
                pitch_sin_err,
                pitch_cos_err,
                yaw_sin_err,
                yaw_cos_err,
                roll_sin_current,
                roll_cos_current,
                pitch_sin_current,
                pitch_cos_current,
                yaw_sin_current,
                yaw_cos_current,
                surge_velocity,
                sway_velocity,
                heave_velocity,
                roll_rate,
                pitch_rate,
                yaw_rate,
                x_acceleration,
                y_acceleration,
            ])
            
            state_error_array = np.concatenate([
                depth_error,
                surge_velocity_error,
                sway_velocity_error,
                self.node.v_err[2:3], #heave error
                # np.array([np.sin(self.node.orientation_err[0]), np.cos(self.node.orientation_err[0])]),
                # np.array([np.sin(self.node.orientation_err[1]), np.cos(self.node.orientation_err[1])]),
                # np.array([np.sin(self.node.orientation_err[2]), np.cos(self.node.orientation_err[2])])
                roll_error,
                pitch_error,
                yaw_error
            ])

            terminated = False

        else:
            print("Warning: No new state/error available, returning dummy observation")
            observation = np.zeros(24)
            terminated = True
            state_error_array = np.zeros(7)

        reward = self.calculate_reward(state_error_array)

        if isinstance(reward, np.ndarray):
            reward = float(reward.item())

        if not hasattr(self, 'episode_rewards'):
            self.episode_rewards = []
        self.episode_rewards.append(reward)

        self.episode_reward += reward

        truncated = False
        if self.episode_step >= self.node.max_steps:
            truncated = True
            
        self.episode_step += 1

        info = {
            'episode_step': self.episode_step,
            'episode_reward': self.episode_reward,
            'setpoint_info': self.current_episode_setpoint_info
        }

        return observation, reward, terminated, truncated, info
    
    def _spin_node(self, timeout_sec=0.1):
        """Process ROS callbacks for a limited time"""
        end_time = time.time() + timeout_sec
        while time.time() < end_time:
            rclpy.spin_once(self.node, timeout_sec=0.01)
    
    def calculate_reward(self, state_error_array):
        """Calculate coupling-aware reward with physics-based coupling"""
        
        # Get coupling method from config
        coupling_method = self.config.get('coupling', {}).get('method', 'v1')
        
        # Calculate coupling-aware performance error
        if coupling_method == 'v1':
            performance_error = self.coupling_calculator.calculate_coupling_aware_reward_v1(
                state_error_array, self.episode_step)
        elif coupling_method == 'v2':
            performance_error = self.coupling_calculator.calculate_coupling_aware_reward_v2(
                state_error_array, self.episode_step)
        elif coupling_method == 'v3':
            performance_error = self.coupling_calculator.calculate_coupling_aware_reward_v3(
                state_error_array, self.episode_step)
        elif coupling_method == 'v4':
            performance_error = self.coupling_calculator.calculate_coupling_aware_reward_v4_enhanced(
                state_error_array, self.episode_step)
        elif coupling_method == 'standard':
            # Fallback to original reward calculation
            performance_error = self._calculate_standard_performance_error(state_error_array)
        else:
            # Default to v1 if method not recognized
            print(f"Warning: Unknown coupling method '{coupling_method}', defaulting to v1")
            performance_error = self.coupling_calculator.calculate_coupling_aware_reward_v1(
                state_error_array, self.episode_step)
        
        # Apply performance weight
        w = self.config['reward_function']
        coupling_performance_reward = w['w1'] * performance_error
        
        # Servo smoothness penalty using sine and cosine components
        servo_smoothness_penalty = 0
        delta_theta = np.zeros(self.num_servos)

        for i in range(self.num_servos):
            if len(self.joint_positions_history) > 0:
                # Compute average sine and cosine of the historical angles
                avg_sin = np.average(np.sin(self.joint_positions_history[:, i]))
                avg_cos = np.average(np.cos(self.joint_positions_history[:, i]))
                historical_avg_angle = np.arctan2(avg_sin, avg_cos)

                # Get the current angle from self.joint_angles
                current_angle = self.joint_angles[i] if i < len(self.joint_angles) else 0
                
                # Calculate the angular difference
                delta_theta[i] = np.abs(current_angle - historical_avg_angle)

        # Accumulate the smoothness penalty
        servo_smoothness_penalty = np.linalg.norm(delta_theta)
        
        # Update joint_positions_history
        if self.num_servos > 0:
            self.joint_positions_history = np.vstack((self.joint_positions_history[1:], self.joint_angles))

        # Thruster usage penalty
        u_t = np.array([
            self.node.thrust_heave_bow,
            self.node.thrust_heave_stern,
            self.node.thrust_surge_port,
            self.node.thrust_surge_starboard
        ])
        thruster_usage_penalty = np.sum(np.abs(u_t))

        # Thruster smoothness penalty
        thruster_smoothness_penalty = np.linalg.norm(u_t - np.average(self.u_prev, axis=0))

        # Update self.u_prev to store the history
        self.u_prev = np.vstack((self.u_prev[1:], u_t))

        # Thruster delta reward
        thruster_delta_reward = np.linalg.norm(u_t - self.u_prev[-2]) if len(self.u_prev) >= 2 else 0

        # Servo angle penalty
        servo_angle_penalty = np.linalg.norm(self.joint_angles) if len(self.joint_angles) > 0 else 0

        # Update thruster_command_action_prev with clear logic
        self.thruster_command_action_prev = np.vstack((
            self.thruster_command_action_prev[1:],  # Keep all except the first
            self.thruster_action  # Add the newest action
        ))

        # Thruster action penalty
        thruster_action_penalty = np.sum(np.abs(self.thruster_action - np.average(self.thruster_command_action_prev, axis=0)))

        # Thruster Direction Change Penalty
        direction_change_penalty = thruster_action_penalty ** 2  # Quadratic penalty (w8 applied below)

        # Final reward calculation with coupling-aware performance term
        reward = (
            coupling_performance_reward -  # This now includes coupling awareness
            w['w2'] * servo_smoothness_penalty -
            w['w3'] * thruster_usage_penalty -
            w['w4'] * thruster_smoothness_penalty -
            w['w5'] * servo_angle_penalty -
            w['w6'] * thruster_delta_reward -
            w['w7'] * thruster_action_penalty -
            w['w8'] * direction_change_penalty
        )
        
        # Optional: Add coupling-specific logging for debugging
        if hasattr(self.coupling_calculator, 'error_history') and len(self.coupling_calculator.error_history) > 0:
            current_errors = self.coupling_calculator.error_history[-1]
            
            # Log coupling metrics every 50 steps for debugging
            if self.episode_step % 50 == 0:
                diagnostics = self.coupling_calculator.get_diagnostics()
                if diagnostics:
                    print(f"Step {self.episode_step}: Coupling Metrics")
                    print(f"  Learning Phase: {diagnostics.get('learning_phase', 'unknown')}")
                    print(f"  Surge-Yaw: surge={current_errors['surge']:.4f}, yaw={current_errors['yaw']:.4f}")
                    print(f"  Pitch-Depth: pitch={current_errors['pitch']:.4f}, depth={current_errors['depth']:.4f}")
                    print(f"  Surge-Yaw Magnitude: {diagnostics['surge_yaw_magnitude']:.4f}")
                    print(f"  Pitch-Depth Magnitude: {diagnostics['pitch_depth_magnitude']:.4f}")
                    print(f"  Coupling Performance: {coupling_performance_reward:.4f}")
                    print(f"  Total Reward: {reward:.4f}")
        
        return reward
    
    def _calculate_standard_performance_error(self, state_error_array):
        """Standard performance error calculation for fallback"""
        original_weights = np.array(self.config['reward_function']['state_error_weights'])
        # state_error_weights = self._expand_weights_for_sincos(original_weights, state_error_array)

        state_error_weights = original_weights

        error_column = state_error_array.reshape(-1, 1)
        error_row = state_error_array.reshape(1, -1)
        weights_diag = np.diag(state_error_weights)
        
        performance_error = error_row @ weights_diag @ error_column
        return float((-performance_error).item())
    
    def _expand_weights_for_sincos(self, original_weights, state_error_array):
        """Expand weights array to account for sin/cos representation"""
        # Dynamically size based on actual state_error_array
        target_size = len(state_error_array)
        state_error_weights = np.zeros(target_size)
        
        # Fill as much as we can from original_weights
        original_weights = np.array(original_weights)
        
        if len(original_weights) == 7 and target_size == 10:
            # Standard case: 7 original -> 10 expanded (sin/cos for 3 angles)
            # Position and velocity errors (unchanged)
            state_error_weights[0] = original_weights[0]  # depth
            state_error_weights[1] = original_weights[1]  # surge
            state_error_weights[2] = original_weights[2]  # sway
            state_error_weights[3] = original_weights[3]  # heave
            
            # Orientation errors (distribute weights between sin/cos pairs)
            state_error_weights[4] = original_weights[4] / 2  # roll sin
            state_error_weights[5] = original_weights[4] / 2  # roll cos
            state_error_weights[6] = original_weights[5] / 2  # pitch sin
            state_error_weights[7] = original_weights[5] / 2  # pitch cos
            state_error_weights[8] = original_weights[6] / 2  # yaw sin
            state_error_weights[9] = original_weights[6] / 2  # yaw cos
            
        elif len(original_weights) == target_size:
            # Already the right size, use as-is
            state_error_weights = original_weights.copy()
            
        else:
            # General case: fill what we can, pad/truncate as needed
            min_size = min(len(original_weights), target_size)
            state_error_weights[:min_size] = original_weights[:min_size]
            
            # If we need more weights than we have, use the last weight value
            if target_size > len(original_weights):
                last_weight = original_weights[-1] if len(original_weights) > 0 else 0.001
                state_error_weights[len(original_weights):] = last_weight
        
        return state_error_weights
    
    def get_coupling_diagnostics(self):
        """Get detailed coupling diagnostics for analysis"""
        if self.coupling_calculator is None:
            return None
        return self.coupling_calculator.get_diagnostics()

    def close(self):
        """Clean up resources"""
        if hasattr(self, 'node') and self.node is not None:
            self.node.destroy_node()


def main(args=None, csv_directory=None):
    """
    Main function to initialize and return the environment.
    
    Args:
        args: ROS2 args (only used in online mode)
        csv_directory: Path to CSV directory for offline training (optional)
    """
    
    if csv_directory is None and not rclpy.ok():
        rclpy.init(args=args)
    
    return AUVEnv(csv_directory=csv_directory)


if __name__ == "__main__":
    main()