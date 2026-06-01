"""
bringup_rl.launch.py
─────────────────────────────────────────────────────────────────────────────
Launch file for RL pick-and-place training.

Starts:
  1. Gazebo Harmonic (pick_and_place.sdf)
  2. Robot State Publisher
  3. Robot spawn in Gazebo
  4. ROS-Gazebo bridge  (/clock, /joint_states, /camera, model poses)
  5. Controller spawners (joint_state_broadcaster, arm_controller, gripper_controller)
  6. DepthAnything+YOLO detector  (dep_any.py)
  7. RL training node  (optional, pass rl_mode:=train|test|off)

Usage
─────
# Start world + detectors only (then run train_ppo.py manually):
ros2 launch arm_bringup bringup_rl.launch.py rl_mode:=off

# Start everything for training:
ros2 launch arm_bringup bringup_rl.launch.py rl_mode:=train n_envs:=2

# Evaluation:
ros2 launch arm_bringup bringup_rl.launch.py rl_mode:=test \
    model_path:=/path/to/checkpoints/ppo_pickplace_final.zip
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    AppendEnvironmentVariable,
    RegisterEventHandler,
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
)
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('arm_bringup')
    pkg_parent = os.path.dirname(pkg_share)

    urdf_file  = os.path.join(pkg_share, 'urdf', 'arm.urdf')
    world_file = os.path.join(pkg_share, 'urdf', 'pick_and_place.sdf')

    with open(urdf_file, 'r') as f:
        robot_desc = f.read()

    # ── Launch arguments ──────────────────────────────────────────────────────
    rl_mode_arg = DeclareLaunchArgument(
        'rl_mode', default_value='off',
        description="'train', 'test', or 'off'")
    model_path_arg = DeclareLaunchArgument(
        'model_path', default_value='./checkpoints/ppo_pickplace_final.zip',
        description='Path to trained model .zip for test mode')
    vecnorm_path_arg = DeclareLaunchArgument(
        'vecnorm_path', default_value='./checkpoints/vecnorm_final.pkl',
        description='Path to VecNormalize .pkl')
    n_envs_arg = DeclareLaunchArgument(
        'n_envs', default_value='2',
        description='Number of parallel RL environments for training')
    object_arg = DeclareLaunchArgument(
        'object', default_value='random',
        description="Object to train on: random|small_cube|large_cube|peg_cylinder")

    rl_mode     = LaunchConfiguration('rl_mode')
    model_path  = LaunchConfiguration('model_path')
    vecnorm_path = LaunchConfiguration('vecnorm_path')
    n_envs      = LaunchConfiguration('n_envs')
    obj_name    = LaunchConfiguration('object')

    # ── GZ resource path ─────────────────────────────────────────────────────
    set_gz_resource_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH', pkg_parent)

    # ── Gazebo Harmonic ───────────────────────────────────────────────────────
    gazebo_ros_pkgs = get_package_share_directory('ros_gz_sim')
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_pkgs, 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': f'-r {world_file} '
                        '--physics-engine gz-physics-bullet-featherstone-plugin'
        }.items(),
    )

    # ── Robot State Publisher ─────────────────────────────────────────────────
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_desc, 'use_sim_time': True}],
    )

    # ── Spawn robot ───────────────────────────────────────────────────────────
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'arm_robot',
            '-topic', 'robot_description',
            '-x', '0.15', '-y', '0.0', '-z', '0.5',
        ],
        output='screen',
    )

    # ── ROS ↔ Gazebo bridge ───────────────────────────────────────────────────
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            # Camera
            '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            # Joint states
            '/world/pick_and_place_world/model/arm_robot/joint_state'
            '@sensor_msgs/msg/JointState[gz.msgs.Model',
            # Object model poses for reward / reset
            '/world/pick_and_place_world/model/small_cube/pose'
            '@geometry_msgs/msg/Pose[gz.msgs.Pose',
            '/world/pick_and_place_world/model/large_cube/pose'
            '@geometry_msgs/msg/Pose[gz.msgs.Pose',
            '/world/pick_and_place_world/model/peg_cylinder/pose'
            '@geometry_msgs/msg/Pose[gz.msgs.Pose',
        ],
        remappings=[
            ('/world/pick_and_place_world/model/arm_robot/joint_state',
             '/joint_states'),
        ],
        output='screen',
    )

    # ── Controller spawners ───────────────────────────────────────────────────
    jsb = Node(
        package='controller_manager', executable='spawner',
        arguments=['joint_state_broadcaster'], output='screen')
    arm_ctrl = Node(
        package='controller_manager', executable='spawner',
        arguments=['arm_controller'], output='screen')
    grip_ctrl = Node(
        package='controller_manager', executable='spawner',
        arguments=['gripper_controller'], output='screen')

    delay_jsb = RegisterEventHandler(
        OnProcessExit(target_action=spawn_robot, on_exit=[jsb]))
    delay_arm = RegisterEventHandler(
        OnProcessStart(target_action=jsb, on_start=[arm_ctrl, grip_ctrl]))

    # ── DepthAnything + YOLO detector ────────────────────────────────────────
    # Points to the updated dep_any.py with correct camera params
    dep_any_node = Node(
        package='arm_bringup',
        executable='dep_any.py',
        name='depth_anything_detector',
        output='screen',
        parameters=[{
            'use_sim_time':        True,
            'publish_debug_image': True,
            'yolo_conf':           0.15,
            'hybrid_depth_pass':   True,
            'yolo_model':          'yolov8n.pt',
        }],
    )

    # ── RL nodes (mode-dependent) ─────────────────────────────────────────────
    # NOTE: These are ExecuteProcess actions because train_ppo / test_agent
    # manage their own rclpy context and SB3 training loop.
    rl_dir = os.path.join(pkg_share, 'rl_agent')

    train_node = ExecuteProcess(
        cmd=[
            'python3', os.path.join(rl_dir, 'train_ppo.py'),
            '--n_envs', n_envs,
            '--object', obj_name,
            '--total_steps', '3000000',
            '--log_dir', '/tmp/rl_pick_place/logs',
            '--ckpt_dir', '/tmp/rl_pick_place/checkpoints',
        ],
        output='screen',
        condition=_EqCondition(rl_mode, 'train'),
    )

    test_node = ExecuteProcess(
        cmd=[
            'python3', os.path.join(rl_dir, 'test_agent.py'),
            '--model', model_path,
            '--vecnorm', vecnorm_path,
            '--object', obj_name,
            '--n_episodes', '20',
            '--verbose',
        ],
        output='screen',
        condition=_EqCondition(rl_mode, 'test'),
    )

    return LaunchDescription([
        # Args
        rl_mode_arg,
        model_path_arg,
        vecnorm_path_arg,
        n_envs_arg,
        object_arg,
        # Infrastructure
        set_gz_resource_path,
        gazebo,
        rsp,
        spawn_robot,
        bridge,
        delay_jsb,
        delay_arm,
        dep_any_node,
        # RL (conditional)
        train_node,
        test_node,
    ])


# ── Small helper: launch condition based on string equality ─────────────────
from launch.condition import Condition
from launch.launch_context import LaunchContext

class _EqCondition(Condition):
    """True when a LaunchConfiguration equals a target string."""
    def __init__(self, config, target):
        self._config = config
        self._target = target
        # Change 'evaluate_condition_function' to 'predicate'
        super().__init__(predicate=self._eval) 

    def _eval(self, context: LaunchContext) -> bool:
        val = context.perform_substitution(self._config) \
              if hasattr(self._config, 'perform') \
              else str(self._config)
        return val == self._target
