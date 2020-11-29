import numpy as np
from robosuite.models.robots.robot_model import RobotModel
from robosuite.utils.mjcf_utils import xml_path_completion


class Laikago(RobotModel):
    """
    Sawyer is a witty single-arm robot designed by Rethink Robotics.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
        bottom_offset (3-array): (x,y,z) offset desired from initial coordinates
    """

    def __init__(self, idn=0, bottom_offset=(0, 0, 0)):
        super().__init__(xml_path_completion("robots/laikago/robot.xml"), idn=idn, bottom_offset=bottom_offset)

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
        return "default_laikago"

    @property
    def init_qpos(self):
        # TODO: Determine which start is better
        # return np.array([-0.5538, -0.8208, 0.4155, 1.8409, -0.4955, 0.6482, 1.9628])
        return np.array([ 6.98325e-06, -0.00573244, 0.0101778, -4.53138e-06,
                         -0.00573432, 0.0101795, 4.49748e-06, -0.00603851, 0.0107461,
                         -5.79075e-06, -0.00604182, 0.0107516])

    @property
    def base_xpos_offset(self):
        raise NotImplementedError
        '''return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length/2, 0, 0)
        }'''

    @property
    def arm_type(self):
        return "none"

    @property
    def _joints(self):
        return ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"]

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
            "torq": ["FR_hip_motor", "FR_thigh_motor", "FR_calf_motor",
                     "FL_hip_motor", "FL_thigh_motor", "FL_calf_motor",
                     "RR_hip_motor", "RR_thigh_motor", "RR_calf_motor",
                     "RL_hip_motor", "RL_thigh_motor", "RL_calf_motor"]
        }

    @property
    def _contact_geoms(self):
        return ["FR_hip_collision", "FR_thigh_collision", "FR_calf_collision", "FR_foot_collision",
                "FL_hip_collision", "FL_thigh_collision", "FL_calf_collision", "FL_foot_collision",
                "RR_hip_collision", "RR_thigh_collision", "RR_calf_collision", "RR_foot_collision",
                "RL_hip_collision", "RL_thigh_collision", "RL_calf_collision", "RL_foot_collision",
                "chassis_collision"]

    @property
    def _root(self):
        return 'chassis'

    @property
    def _links(self):
        return ["FR_hip", "FR_thigh", "FR_calf",
                "FL_hip", "FL_thigh", "FL_calf",
                "RR_hip", "RR_thigh", "RR_calf",
                "RL_hip", "RL_thigh", "RL_calf"]

    @property
    def robot_base_free_joint(self):
        return self.correct_naming(self._robot_base+"_joint")
