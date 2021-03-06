import numpy as np

from collections import OrderedDict

import robosuite.utils.transform_utils as T

from robosuite.models.grippers import gripper_factory
from robosuite.controllers import controller_factory, load_controller_config

from robosuite.robots.robot import Robot
from robosuite.utils.control_utils import DeltaBuffer, RingBuffer

import os
import copy


class Quadruped(Robot):
    """
    Initializes a single-armed robot simulation object.

    Args:
        robot_type (str): Specification for specific robot arm to be instantiated within this env (e.g: "Panda")

        idn (int or str): Unique ID of this robot. Should be different from others

        controller_config (dict): If set, contains relevant controller parameters for creating a custom controller.
            Else, uses the default controller for this specific task

        initial_qpos (sequence of float): If set, determines the initial joint positions of the robot to be
            instantiated for the task

        initialization_noise (dict): Dict containing the initialization noise parameters. The expected keys and
            corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to "None" or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            :Note: Specifying None will automatically create the required dict with "magnitude" set to 0.0

        gripper_type (str): type of gripper, used to instantiate
            gripper models from gripper factory. Default is "default", which is the default gripper associated
            within the 'robot' specification. None removes the gripper, and any other (valid) model overrides the
            default gripper

        gripper_visualization (bool): True if using gripper visualization.
            Useful for teleoperation.

        control_freq (float): how many control signals to receive
            in every second. This sets the amount of simulation time
            that passes between every action input.
    """

    def __init__(
        self,
        robot_type: str,
        idn=0,
        controller_config=None,
        initial_qpos=None,
        initialization_noise=None,
        control_freq=10
    ):

        self.controller = None
        self.controller_config = copy.deepcopy(controller_config)
        self.has_gripper = False
        self.control_freq = control_freq

        self.torques = None                                 # Current torques being applied

        self._ref_chassis_free_joint_index = None  # xml joint indexes for robot in mjsim
        self._ref_chassis_pos_indexes = None  # xml joint position indexes in mjsim
        self._ref_chassis_vel_indexes = None  # xml joint velocity indexes in mjsim

        self.recent_qpos = None                             # Current and last robot arm qpos
        self.recent_actions = None                          # Current and last action applied
        self.recent_torques = None                          # Current and last torques applied

        super().__init__(
            robot_type=robot_type,
            idn=idn,
            initial_qpos=initial_qpos,
            initialization_noise=initialization_noise,
        )

    def _load_controller(self):
        """
        Loads controller to be used for dynamic trajectories
        """
        # First, load the default controller if none is specified
        if not self.controller_config:
            # Need to update default for a single agent
            controller_path = os.path.join(os.path.dirname(__file__), '..',
                                           'controllers/config/{}.json'.format(
                                               self.robot_model.default_controller_config))
            self.controller_config = load_controller_config(custom_fpath=controller_path)

        # Assert that the controller config is a dict file:
        #             NOTE: "type" must be one of: {JOINT_POSITION, JOINT_TORQUE, JOINT_VELOCITY,
        #                                           OSC_POSITION, OSC_POSE, IK_POSE}
        assert type(self.controller_config) == dict, \
            "Inputted controller config must be a dict! Instead, got type: {}".format(type(self.controller_config))
        assert self.controller_config["type"] == "LOCOMOTION_JOINT_TORQUE", \
            "Only JOINT_TORQUE controllers are currently supported for quadruped robots, please change the " \
            "controller to be a joint torque controller. Got controller type: {}".format(self.controller_config["type"])

        # Add to the controller dict additional relevant params:
        #   the robot name, mujoco sim, eef_name, joint_indexes, timestep (model) freq,
        #   policy (control) freq, and ndim (# joints)
        self.controller_config["robot_name"] = self.name
        self.controller_config["sim"] = self.sim
        self.controller_config["chassis_name"] = self.robot_model.robot_base
        self.controller_config["joint_indexes"] = {
            "joints": self.joint_indexes,
            "qpos": self._ref_joint_pos_indexes,
            "qvel": self._ref_joint_vel_indexes
                                              }
        self.controller_config["actuator_range"] = self.torque_limits
        self.controller_config["policy_freq"] = self.control_freq
        self.controller_config["ndim"] = len(self.robot_joints)

        # Instantiate the relevant controller
        self.controller = controller_factory(self.controller_config["type"], self.controller_config)

    def load_model(self):
        """
        Loads robot and optionally add grippers.
        """
        # First, run the superclass method to load the relevant model
        super().load_model()

        # Verify that the loaded model is of the correct type for this robot
        if self.robot_model.arm_type != "none":
            raise TypeError("Error loading robot model: Incompatible arm type specified for this robot. "
                            "Requested model arm type: {}, robot arm type: {}"
                            .format(self.robot_model.arm_type, type(self)))

    def reset(self, deterministic=False):
        """
        Sets initial pose of arm and grippers. Overrides gripper joint configuration if we're using a
        deterministic reset (e.g.: hard reset from xml file)

        Args:
            deterministic (bool): If true, will not randomize initializations within the sim
        """
        # First, run the superclass method to reset the position and controller
        super().reset(deterministic)

        # Update base pos / ori references in controller
        self.controller.update_base_pose(self.base_pos, self.base_ori)

        # Setup buffers to hold recent values
        self.recent_qpos = DeltaBuffer(dim=len(self.joint_indexes))
        self.recent_actions = DeltaBuffer(dim=self.action_dim)
        self.recent_torques = DeltaBuffer(dim=len(self.joint_indexes))

    def setup_references(self):
        """
        Sets up necessary reference for robots, grippers, and objects.

        Note that this should get called during every reset from the environment
        """
        # First, run the superclass method to setup references for joint-related values / indexes
        super().setup_references()

        # Now, add references to chassis
        # indices for chassisS in qpos, qvel
        self._ref_chassis_free_joint_index = self.sim.model.joint_name2id(self.robot_model.robot_base_free_joint)

        (chassis_free_joint_addr_start, chassis_free_joint_addr_end) = \
            self.sim.model.get_joint_qpos_addr(self.robot_model.robot_base_free_joint)
        self._ref_chassis_pos_indexes = [x for x in range(chassis_free_joint_addr_start,
                                                          chassis_free_joint_addr_end)]
        (chassis_free_joint_addr_start, chassis_free_joint_addr_end) = \
            self.sim.model.get_joint_qvel_addr(self.robot_model.robot_base_free_joint)
        #print("CHASSIS:" + str(self._ref_chassis_pos_indexes))
        self._ref_chassis_vel_indexes = [x for x in range(chassis_free_joint_addr_start,
                                                          chassis_free_joint_addr_end)]

        #print("CHASSIS:" + str(self._ref_chassis_vel_indexes))


    def control(self, action, policy_step=False):
        """
        Actuate the robot with the
        passed joint velocities and gripper control.

        Args:
            action (np.array): The control to apply to the robot. The first @self.robot_model.dof dimensions should be
                the desired normalized joint velocities and if the robot has a gripper, the next @self.gripper.dof
                dimensions should be actuation controls for the gripper.
            policy_step (bool): Whether a new policy step (action) is being taken

        Raises:
            AssertionError: [Invalid action dimension]
        """

        # clip actions into valid range
        assert len(action) == self.action_dim, \
            "environment got invalid action dimension -- expected {}, got {}".format(
                self.action_dim, len(action))

        # Update the controller goal if this is a new policy step
        if policy_step:
            self.controller.set_goal(action)

        # Now run the controller for a step
        torques = self.controller.run_controller()

        # Clip the torques
        low, high = self.torque_limits
        self.torques = np.clip(torques, low, high)

        # Apply joint torque control
        self.sim.data.ctrl[self._ref_joint_torq_actuator_indexes] = self.torques

        # If this is a policy step, also update buffers holding recent values of interest
        if policy_step:
            # Update proprioceptive values
            self.recent_qpos.push(self._joint_positions)
            self.recent_actions.push(action)
            self.recent_torques.push(self.torques)

    def visualize_gripper(self):
        """
        Visualizes the gripper site(s) if applicable.
        """
        pass

    def get_observations(self, di: OrderedDict):
        """
        Returns an OrderedDict containing robot observations [(name_string, np.array), ...].

        Important keys:

            `'robot-state'`: contains robot-centric information.

        Args:
            di (OrderedDict): Current set of observations from the environment

        Returns:
            OrderedDict: Augmented set of observations that include this robot's proprioceptive observations
        """

        # Get prefix from robot model to avoid naming clashes for multiple robots
        pf = self.robot_model.naming_prefix

        # proprioceptive features
        di[pf + "chassis_pos"] =np.array(
            [self.sim.data.qpos[x] for x in self._ref_chassis_pos_indexes]
        )

        di[pf + "chassis_vel"] = np.array(
            [self.sim.data.qvel[x] for x in self._ref_chassis_vel_indexes]
        )

        di[pf + "joint_pos"] = np.array(
            [self.sim.data.qpos[x] for x in self._ref_joint_pos_indexes]
        )
        di[pf + "joint_vel"] = np.array(
            [self.sim.data.qvel[x] for x in self._ref_joint_vel_indexes]
        )

        robot_states = [
            di[pf + "chassis_pos"],
            di[pf + "chassis_vel"],
            np.sin(di[pf + "joint_pos"]),
            np.cos(di[pf + "joint_pos"]),
            di[pf + "joint_vel"],
        ]

        # Add in eef pos / qpos
        '''di[pf + "eef_pos"] = np.array(self.sim.data.site_xpos[self.eef_site_id])
        di[pf + "eef_quat"] = T.convert_quat(
            self.sim.data.get_body_xquat(self.robot_model.eef_name), to="xyzw"
        )
        robot_states.extend([di[pf + "eef_pos"], di[pf + "eef_quat"]])
        '''

        di[pf + "robot-state"] = np.concatenate(robot_states)
        return di

    @property
    def base_dof(self):
        return self.robot_model.base_dof

    @property
    def action_limits(self):
        """
        Action lower/upper limits per dimension.

        Returns:
            2-tuple:

                - (np.array) minimum (low) action values
                - (np.array) maximum (high) action values
        """
        # Action limits based on controller limits
        low_c, high_c = self.controller.control_limits
        return low_c, high_c

    @property
    def torque_limits(self):
        """
        Torque lower/upper limits per dimension.

        Returns:
            2-tuple:

                - (np.array) minimum (low) torque values
                - (np.array) maximum (high) torque values
        """
        # Torque limit values pulled from relevant robot.xml file
        low = self.sim.model.actuator_ctrlrange[self._ref_joint_torq_actuator_indexes, 0]
        high = self.sim.model.actuator_ctrlrange[self._ref_joint_torq_actuator_indexes, 1]

        return low, high

    @property
    def action_dim(self):
        """
        Action space dimension for this robot (controller dimension + gripper dof)

        Returns:
            int: action dimension
        """
        return self.controller.control_dim

    @property
    def dof(self):
        """
        Returns:
            int: degrees of freedom of the robot (with grippers).
        """
        # Get the dof of the base robot model
        dof = super().dof
        return dof

    @property
    def js_energy(self):
        """
        Returns:
            np.array: the energy consumed by each joint between previous and current steps
        """
        # We assume in the motors torque is proportional to current (and voltage is constant)
        # In that case the amount of power scales proportional to the torque and the energy is the
        # time integral of that
        # Note that we use mean torque
        return np.abs((1.0 / self.control_freq) * self.recent_torques.average)

