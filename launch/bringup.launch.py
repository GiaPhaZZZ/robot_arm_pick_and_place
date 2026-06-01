import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, AppendEnvironmentVariable, RegisterEventHandler
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Locate package directories
    pkg_share = get_package_share_directory('arm_bringup')
    
    # 2. Extract parent directory paths to help Gazebo resolve "package://" structure
    package_parent_dir = os.path.dirname(pkg_share) 

    # Path to your modified URDF (Double check if this is arm.urdf or arm_description.urdf)
    urdf_file = os.path.join(pkg_share, 'urdf', 'arm.urdf')
    with open(urdf_file, 'r') as infp:
        robot_desc = infp.read()

    # 3. Inject path rules directly into Gazebo's tracking ecosystem
    set_gz_resource_path = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        package_parent_dir
    )
    
    world_file = os.path.join(pkg_share, 'urdf', 'pick_and_place.sdf')

    # 4. Include Gazebo Sim Launch (Gazebo Harmonic)
    gazebo_ros_pkgs = get_package_share_directory('ros_gz_sim')
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(gazebo_ros_pkgs, 'launch', 'gz_sim.launch.py')
        ),
        # ADDED THE -s FLAG HERE
        launch_arguments={'gz_args': f'-r -s {world_file} --physics-engine gz-physics-bullet-featherstone-plugin'}.items(),
    )

    # 5. Robot State Publisher Node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_desc, 
            'use_sim_time': True
        }]
    )

    # 6. Spawn Robot Entity into Gazebo
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'arm_robot',
            '-topic', 'robot_description',
            '-x', '0.15',   
            '-y', '0.0',    
            '-z', '0.5'     
        ],
        output='screen'
    )
    
    # 7. Bridge Node for clocks, states, and the camera feed
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/model/target_cube/pose@geometry_msgs/msg/Pose@gz.msgs.Pose',
            '/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image'
        ],
        output='screen'
    )

    # 8. ROS 2 Controller Spawners
    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
        output="screen",
    )

    arm_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller"],
        output="screen",
    )

    gripper_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["gripper_controller"],
        output="screen",
    )

    # Delay the joint state broadcaster until the robot successfully spawns (creates) in Gazebo
    delay_joint_state_broadcaster = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_robot,
            on_exit=[joint_state_broadcaster],
        )
    )

    # Fix: Load the arm and gripper controllers as soon as the joint_state_broadcaster starts up active
    delay_arm_controllers = RegisterEventHandler(
        event_handler=OnProcessStart(
            target_action=joint_state_broadcaster,
            on_start=[arm_controller, gripper_controller],
        )
    )

    # 9. Depth Anything V2 Vision Target Estimation Node
    depth_anything_detector = Node(
        package='arm_bringup',             
        executable='dep_any.py',        
        name='depth_anything_detector',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'camera_height_m': 0.9,
            'camera_tilt_deg': 45.0,
            'publish_debug_image': True,
            'foreground_threshold': 0.4    
        }]
    )

    return LaunchDescription([
        set_gz_resource_path,
        gazebo,
        robot_state_publisher,
        spawn_robot,
        bridge,
        delay_joint_state_broadcaster,
        delay_arm_controllers,
        depth_anything_detector             
    ])