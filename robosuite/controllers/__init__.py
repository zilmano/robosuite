from .controller_factory import controller_factory, load_controller_config, reset_controllers, get_pybullet_server
from .osc import OperationalSpaceController
from .joint_pos import JointPositionController
from .joint_vel import JointVelocityController
from .joint_tor import JointTorqueController
from .locomotion_join_tor import LocomotionJointTorqueController


CONTROLLER_INFO = {
    "JOINT_VELOCITY":  "Joint Velocity",
    "JOINT_TORQUE":    "Joint Torque",
    "JOINT_POSITION":  "Joint Position",
    "OSC_POSITION": "Operational Space Control (Position Only)",
    "OSC_POSE":     "Operational Space Control (Position + Orientation)",
    "IK_POSE":      "Inverse Kinematics Control (Position + Orientation) (Note: must have PyBullet installed)",
    "LOCOMOTION_JOINT_TORQUE": "temporary hack to allow joint torque control for quadruped locomotion, as the base_controller currently include eef_name which doesn't exist in quadruped" 
}

ALL_CONTROLLERS = CONTROLLER_INFO.keys()
