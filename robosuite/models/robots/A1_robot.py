import numpy as np
from robosuite.models.robots.robot_model import RobotModel
from robosuite.utils.mjcf_utils import xml_path_completion


class A1(RobotModel):
    """
    Sawyer is a witty single-arm robot designed by Rethink Robotics.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
        bottom_offset (3-array): (x,y,z) offset desired from initial coordinates
    """

    def __init__(self, idn=0, bottom_offset=(0, 0, 0)):
        super().__init__(xml_path_completion("robots/A1/robot.xml"), idn=idn, bottom_offset=bottom_offset)

    @property
    def dof(self):
        return 12

    @property
    def base_dof(self):
        return 6

    @property
    def gripper(self):
        raise NotImplementedError

    @property
    def default_controller_config(self):
        return "default_A1"

    @property
    def init_qpos(self):
        # TODO: Determine which start is better
         return np.array([0.,  0., 0., 0.,
                          0.,  0., 0., 0.,
                          0.,  0., 0., 0.])

    @property
    def base_xpos_offset(self):
        raise NotImplementedError

    @property
    def arm_type(self):
        return "none"

    @property
    def _joints(self):
        return ["FR_hip_joint", "FR_upper_joint", "FR_lower_joint",
                "FL_hip_joint", "FL_upper_joint", "FL_lower_joint",
                "RR_hip_joint", "RR_upper_joint", "RR_lower_joint",
                "RL_hip_joint", "RL_upper_joint", "RL_lower_joint"]

    @property
    def _eef_name(self):
        raise NotImplementedError

    @property
    def _robot_base(self):
        return "chassis"

    @property
    def _actuators(self):
        return {
            "pos": [],  # No position actuators for sawyer
            "vel": [],  # No velocity actuators for sawyer
            "torq": ["FR_hip_motor", "FR_upper_motor", "FR_lower_motor",
                     "FL_hip_motor", "FL_upper_motor", "FL_lower_motor",
                     "RR_hip_motor", "RR_upper_motor", "RR_lower_motor",
                     "RL_hip_motor", "RL_upper_motor", "RL_lower_motor"]
        }

    @property
    def _contact_geoms(self):
        return ["FR_hip_collision", "FR_upper_collision", "FR_lower_collision", "FR_foot_collision",
                "FL_hip_collision", "FL_upper_collision", "FL_lower_collision", "FL_foot_collision",
                "RR_hip_collision", "RR_upper_collision", "RR_lower_collision", "RR_foot_collision",
                "RL_hip_collision", "RL_upper_collision", "RL_lower_collision", "RL_foot_collision",
                "chassis_collision"]

    @property
    def _root(self):
        return 'chassis'

    @property
    def _links(self):
        return ["FR_hip", "FR_upper", "FR_lower",
                "FL_hip", "FL_upper", "FL_lower",
                "RR_hip", "RR_upper", "RR_lower",
                "RL_hip", "RL_upper", "RL_lower"]

    @property
    def robot_base_free_joint(self):
        return self.correct_naming(self._robot_base+"_joint")
