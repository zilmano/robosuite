from collections import OrderedDict
import numpy as np

from robosuite.utils.transform_utils import convert_quat
from robosuite.utils.mjcf_utils import CustomMaterial

from robosuite.environments.robot_env import RobotEnv
from robosuite.robots import Quadruped

from robosuite.models.arenas import WalkingArena
from robosuite.models.tasks import LocomotionTask

from robosuite.models.objects import BoxObject, CapsuleObject, BallObject, CylinderObject

class Walk(RobotEnv):
    """
    This class corresponds to the lifting task for a single robot arm.

    Args:
        robots (str or list of str): Specification for specific robot arm(s) to be instantiated within this env
            (e.g: "Sawyer" would generate one arm; ["Panda", "Panda", "Sawyer"] would generate three robot arms)
            Note: Must be a single single-arm robot!

        controller_configs (str or list of dict): If set, contains relevant controller parameters for creating a
            custom controller. Else, uses the default controller for this specific task. Should either be single
            dict if same controller is to be used for all robots or else it should be a list of the same length as
            "robots" param

        initialization_noise (dict or list of dict): Dict containing the initialization noise parameters.
            The expected keys and corresponding value types are specified below:

            :`'magnitude'`: The scale factor of uni-variate random noise applied to each of a robot's given initial
                joint positions. Setting this value to `None` or 0.0 results in no noise being applied.
                If "gaussian" type of noise is applied then this magnitude scales the standard deviation applied,
                If "uniform" type of noise is applied then this magnitude sets the bounds of the sampling range
            :`'type'`: Type of noise to apply. Can either specify "gaussian" or "uniform"

            Should either be single dict if same noise value is to be used for all robots or else it should be a
            list of the same length as "robots" param

            :Note: Specifying "default" will automatically use the default noise settings.
                Specifying None will automatically create the required dict with "magnitude" set to 0.0.

        use_camera_obs (bool): if True, every observation includes rendered image(s)

        use_object_obs (bool): if True, include object (cube) information in
            the observation.

        reward_scale (None or float): Scales the normalized reward function by the amount specified.
            If None, environment reward remains unnormalized

        reward_shaping (bool): if True, use dense rewards.

        use_indicator_object (bool): if True, sets up an indicator object that
            is useful for debugging.

        has_renderer (bool): If true, render the simulation state in
            a viewer instead of headless mode.

        has_offscreen_renderer (bool): True if using off-screen rendering

        render_camera (str): Name of camera to render if `has_renderer` is True. Setting this value to 'None'
            will result in the default angle being applied, which is useful as it can be dragged / panned by
            the user using the mouse

        render_collision_mesh (bool): True if rendering collision meshes in camera. False otherwise.

        render_visual_mesh (bool): True if rendering visual meshes in camera. False otherwise.

        control_freq (float): how many control signals to receive in every second. This sets the amount of
            simulation time that passes between every action input.

        horizon (int): Every episode lasts for exactly @horizon timesteps.

        ignore_done (bool): True if never terminating the environment (ignore @horizon).

        hard_reset (bool): If True, re-loads model, sim, and render object upon a reset call, else,
            only calls sim.reset and resets all robosuite-internal variables

        camera_names (str or list of str): name of camera to be rendered. Should either be single str if
            same name is to be used for all cameras' rendering or else it should be a list of cameras to render.

            :Note: At least one camera must be specified if @use_camera_obs is True.

            :Note: To render all robots' cameras of a certain type (e.g.: "robotview" or "eye_in_hand"), use the
                convention "all-{name}" (e.g.: "all-robotview") to automatically render all camera images from each
                robot's camera list).

        camera_heights (int or list of int): height of camera frame. Should either be single int if
            same height is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_widths (int or list of int): width of camera frame. Should either be single int if
            same width is to be used for all cameras' frames or else it should be a list of the same length as
            "camera names" param.

        camera_depths (bool or list of bool): True if rendering RGB-D, and RGB otherwise. Should either be single
            bool if same depth setting is to be used for all cameras or else it should be a list of the same length as
            "camera names" param.

    Raises:
        AssertionError: [Invalid number of robots specified]
    """

    def __init__(
        self,
        robots,
        controller_configs=None,
        use_camera_obs=True,
        use_object_obs=True,
        init_robot_pose = (0, 0, 0.45),
        init_robot_ori = (0, 0, 0),
        reward_scale=1.0,
        reward_shaping=False,
        use_indicator_object=False,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        control_freq=10,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="frontview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
    ):
        # First, verify that only one robot is being inputted
        self._check_robot_configuration(robots)

        #init robot chassis pose and orientation
        self.init_robot_pose = np.array(init_robot_pose)
        self.init_robot_ori = np.array(init_robot_ori)

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        super().__init__(
            robots=robots,
            controller_configs=controller_configs,
            gripper_types=None,
            gripper_visualizations=False,
            initialization_noise=None,
            use_camera_obs=use_camera_obs,
            use_indicator_object=use_indicator_object,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.25 is provided if the cube is lifted

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube

        The sparse reward only consists of the lifting component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.

        # sparse completion reward
        if self._check_success():
            reward = 2.25

        # use a shaping reward
        elif self.reward_shaping:
            chassis_body_id = self.sim.model.body_name2id(self.robots[0].robot_model.robot_base)
            # x_travel_dist = self.sim.data.body_xpos[chassis_body_id][0]
            body_pos = self.sim.data.body_xpos[chassis_body_id]
            print("body pos type:" + str(type(body_pos)) + " body pos:" + str(body_pos))
            reward = -1*np.linalg.norm(body_pos-np.array([0., 0., 0.43]))
            print(reward)
            

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Verify the correct robot has been loaded
        assert isinstance(self.robots[0], Quadruped), \
            "Error: Expected one single-armed robot! Got {} type instead.".format(type(self.robots[0]))

        # Adjust base pose accordingly
        print("init robo pose:" + str(self.init_robot_pose))
        self.robots[0].robot_model.set_base_xpos(self.init_robot_pose)
        self.robots[0].robot_model.set_base_ori(self.init_robot_ori)


        # load model for table top workspace
        # OLEG TODO: allow specifiying sizes for waliking arena floor, and maybe add random generated obs later, terrain type etc....
        self.mujoco_arena = WalkingArena()

        '''tex_attrib = {
            "type": "cube",
        }
        mat_attrib = {
            "texrepeat": "1 1",
            "specular": "0.4",
            "shininess": "0.1",
        }

        redwood = CustomMaterial(
            texture="WoodRed",
            tex_name="redwood",
            mat_name="redwood_mat",
            tex_attrib=tex_attrib,
            mat_attrib=mat_attrib,
        )'''

        '''cube = BoxObject(
            name="cube",
            size_min=[0.6, 0.6, 0.6],  # [0.015, 0.015, 0.015],
            size_max=[0.6, 0.6, 0.6],  # [0.018, 0.018, 0.018])
            rgba=[1, 0, 0, 1],
            material=redwood,
        )

        ball = BallObject(
            name="ball",
            size_min=[1],  # [0.015, 0.015, 0.015],
            size_max=[1],  # [0.018, 0.018, 0.018])
            rgba=[1, 1, 0, 1],
        )

        capsule = CapsuleObject(
            name="capsule",
            size_min=[0.5,0.3],  # [0.015, 0.015, 0.015],
            size_max=[0.5,0.3],  # [0.018, 0.018, 0.018])
            rgba=[0.5, 0.6, 0, 0.9],
        )

        cylinder = CylinderObject(
            name="cylinder",
            size_min=[1, 1],  # [0.015, 0.015, 0.015],
            size_max=[1, 1],  # [0.018, 0.018, 0.018])
            rgba=[0.7, 0.8, 1, 1],

        )'''


        '''self.mujoco_objects = OrderedDict([("cube", cube),
                                           ("ball", ball),
                                           ("capsule", capsule),
                                           ("cylinder", cylinder)])'''
        self.mujoco_objects = None
        if self.use_indicator_object:
            self.mujoco_arena.add_pos_indicator()

        # Arena always gets set to zero origin
        self.mujoco_arena.set_origin([0, 0, 0])

        # task includes arena, robot, and objects of interest
        self.model = LocomotionTask(
            mujoco_arena=self.mujoco_arena, 
            mujoco_robots=[self.robots[0].robot_model, ],
            mujoco_objects=self.mujoco_objects,
            visual_objects=None, 
            initializer=None
        )

        self.model.place_objects()


    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._get_reference()

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:
            #OLEG TODO: add randomized start z axis postion for the robot?
            pass


    def _get_observation(self):
        """
        Returns an OrderedDict containing observations [(name_string, np.array), ...].

        Important keys:

            `'robot-state'`: contains robot-centric information.

            `'object-state'`: requires @self.use_object_obs to be True. Contains object-centric information.

            `'image'`: requires @self.use_camera_obs to be True. Contains a rendered frame from the simulation.

            `'depth'`: requires @self.use_camera_obs and @self.camera_depth to be True.
            Contains a rendered depth map from the simulation

        Returns:
            OrderedDict: Observations from the environment
        """
        di = super()._get_observation()

        return di

    def _check_success(self):
        """
        Check if cube has been lifted.

        Returns:
            bool: True if cube has been lifted
        """
        chassis_body_id = self.sim.model.body_name2id(self.robots[0].robot_model.robot_base)
        #x_travel_dist = self.sim.data.body_xpos[chassis_body_id][0]
        body_pos = self.sim.data.body_xpos[chassis_body_id]
        '''
        print(type(body_pos))
        print(body_pos)

        print("")
        node = self.model.worldbody.find("./body[@name='{}']".format(
            self.robots[0].robot_model.robot_base)
        )
        from robosuite.utils.mjcf_utils import array_to_string
        print(f"xml pose:{node.get('pos')}")'''
        return np.allclose(body_pos, np.array([0., 0., 0.43]))

    def _visualization(self):
        """
        Do any needed visualization here. Overrides superclass implementations.
        """

        pass

    def _check_robot_configuration(self, robots):
        """
        Sanity check to make sure the inputted robots and configuration is acceptable

        Args:
            robots (str or list of str): Robots to instantiate within this env
        """
        if type(robots) is list:
            assert len(robots) == 1, "Error: Only one robot should be inputted for this task!"


