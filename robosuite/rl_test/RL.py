import gym
import numpy as np
import argparse
import pprint
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from stable_baselines3 import TD3
from stable_baselines3 import DDPG
from stable_baselines3 import SAC
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback



import robosuite as suite
#from robosuite.wrappers.clipper_gym_wrapper import ClipperGymWrapper
from robosuite.wrappers.gym_wrapper import GymWrapper
from robosuite import load_controller_config

log_dir = "./logs"

class Logger(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, log_dir, verbose=0):
        super(Logger, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.log_dir = log_dir
        self.num_episodes = 0
        self.batch_rewards = []
        

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        #print("step:" + str(self.num_episodes))
        
        return True

    def _on_rollout_end(self) -> None:

        """
        This event is triggered before updating the policy.
        """
        #self.num_episodes += 1
        #print("num_eps:" + str(self.num_episodes))
        #if (self.num_episodes % 10) == 0:
            #x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            #mean_reward = np.mean(y[-10:])
            #self.batch_rewards.append(mean_reward)
            #print("-----------")
            #print("mean eps return: {}".format(mean_reward))
            

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        #vals = np.array(self.batch_rewards)
        #np.savetxt("{}/rewards.csv".format(self.log_dir), vals, delimiter=",")



def make(has_renderer=False, init_robot_pose=(-1.37550484e-02,  5.21560077e-03,  8.78072546e-02), renderer=None, 
         horizon=100):
    controller_conf = load_controller_config(default_controller="LOCOMOTION_JOINT_TORQUE")
    os.makedirs(log_dir, exist_ok=True)
   
    rbs_env = suite.make(
        env_name="Walk",
        robots="Laikago",
        controller_configs=controller_conf,
        has_renderer=has_renderer,
        render_camera=renderer,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=10,
        init_robot_pose=init_robot_pose,
        init_robot_ori = (0, 0, 0),
        reward_shaping=True,
        horizon=horizon
    )

    env = GymWrapper(
            rbs_env,
            keys=[
                "robot0_chassis_pos",
                "robot0_chassis_vel",
                "robot0_joint_pos",
                "robot0_joint_vel"
            ]
    )

    if not has_renderer:
        env = Monitor(env, log_dir)

    print("box:")
    print(env.action_space)
    print("shape -1")
    print(env.action_space.shape[-1])
    print("shape")
    print(env.action_space.shape)
    print("high")
    print(env.action_space.high)
    print("low")
    print(env.action_space.low)

    print("box:")
    print(env.observation_space)
    print("shape -1")
    print(env.observation_space.shape[-1])
    print("shape")
    print(env.observation_space.shape)
    print("high")
    print(env.observation_space.high)
    print("low")
    print(env.observation_space.low)

    return env

def train_TD3(env):
    
    print(f"action space shape -1:{env.action_space.shape[-1]}")

    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.02 * np.ones(n_actions))

    model = TD3(MlpPolicy, env, learning_rate=0.0003, buffer_size=100000, action_noise=action_noise, batch_size=128, learning_starts=128, verbose=1)
    model.learn(total_timesteps=2000000, log_interval=10)

    model.save("TD3_pkl")

def train_DDPG(env):
    
    print(f"action space shape -1:{env.action_space.shape[-1]}")

    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.02 * np.ones(n_actions))

    model = DDPG('MlpPolicy', env, learning_rate=0.0003, learning_starts=5, train_freq=10, n_episodes_rollout=-1, buffer_size=100000, action_noise=action_noise,
                 batch_size=128, verbose=2,)
    model.learn(total_timesteps=1000000, log_interval=1)

    model.save("DDPG_pkl")

def train_SAC(env, title = "Stand Up Task Learning Curve"):
    print(f"action space shape -1:{env.action_space.shape[-1]}")

    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    callback = Logger(log_dir=log_dir)
    timesteps = 20000
    model = SAC('MlpPolicy', env, learning_rate=0.001, learning_starts=10000,  ent_coef= 'auto_1.1', train_freq=1, n_episodes_rollout=-1, target_entropy=-21, buffer_size=1000000, action_noise=None,
                batch_size=64, verbose=1, policy_kwargs=dict(net_arch=[64, 64]))
    model.learn(total_timesteps=timesteps, callback=callback)

    model.save("SAC_pkl")
    plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, title)
    plt.savefig("{}/learn_curve.png".format(log_dir))
    plt.show()


def run(env, algname,filename):
    if algname == "TD3":
        model = TD3.load(f"{algname}_pkl")
    elif algname == "SAC":
        if filename:
            model = SAC.load(f"{filename}")
        else:
            model = SAC.load(f"{algname}_pkl")
    elif algname == "DDPG":
        model = DDPG.load(f"{algname}_pkl")
    else:
        raise "Wrong algorithm name provided."

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            break


def run_manual(env):
    joint_indexes = {}
    for joint_name in env.env.robots[0].robot_model.actuators["torq"]:
        joint_indexes[joint_name] = env.env.sim.model.actuator_name2id(joint_name)
    import pprint
    pprint.pprint(joint_indexes)
    for i in range(0, 50):
        action = np.zeros(env.robots[0].dof)
        obs, rewards, done, info = env.step(action)
        env.render()
    print()
    print("setting shoulders...")
    for i in range(0, 50):
            #Quadruped::Control::actuator indexes [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
            action = np.array([0.1, 0, 0, #FR
                               -0.1, 0,0, #FL
                               0.1, 0, 0, #RR
                               -0.1, 0, 0]) #RL
            obs, rewards, done, info = env.step(action)
            env.render()
    '''for i in range(0, 200):
            #Quadruped::Control::actuator indexes [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
            action = np.array([-0.5, -i/200*1, 1, #FR
                               0.5, -i/200*1  ,1, #FL
                               -0.5, i/200*1, 1, #RR
                               0.5, i/200*1, 1]) #RL
            obs, rewards, done, info = env.step(action)
            env.render()'''
    print("starting rising...\n\n")
    a = {
        'FR_HIP':0,
        'FR_THIGH':0,
        'FR_CALF':0,
        'FL_HIP': 0,
        'FL_THIGH': 0,
        'FL_CALF':0,
        'RR_HIP':0,
        'RR_THIGH':0,
        'RR_CALF':0,
        'RL_HIP':0,
        'RL_THIGH':0,
        'RL_CALF':0,
    }
    
    d_F = 70
    d_RR = 40
    d_RL = 0
    for i in range(0,200):
            #Quadruped::Control::actuator indexes [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
            a['FR_HIP'] = 0.1
            a['FL_HIP'] = -0.1
            a['RR_HIP'] =0.06
            a['RL_HIP'] = -0.1
            '''if (i > x and i < 100+x):
                a['FR_THIGH'] = -1.8*(i-x)/200
                a['FL_THIGH'] = -1.8*(i-x)/200
                a['FR_CALF'] = max((i-x)/200, 0.5)
                a['FL_CALF'] = max((i-x)/200, 0.5)'''
           
            if (i > d_RR < 80):
                a['RR_THIGH'] = (-i+d_RR)/200
            elif i > 80:
                a['RR_THIGH'] = 0.2

            if (i > d_RL < 50):
                a['RL_THIGH'] = -i/200
            elif i > 50: 
                a['RL_THIGH'] = 0.2

            if (i > d_RR and i < 70):
                #a['FR_CALF'] = (i/200)
                #a['FL_CALF'] = (i/200)
                a['RR_CALF'] = (i/200)
            elif i > 70:
                a['RR_CALF'] = 0.35
            if (i > d_RL and i < 40):
                a['RL_CALF'] = (i/200)
            elif i > 30:
                a['RL_CALF'] = 0.15

            
            action = np.array([ a['FR_HIP'], a['FR_THIGH'], a['FR_CALF'],   # FR
                                a['FL_HIP'], a['FL_THIGH'], a['FL_CALF'],   # FL
                                a['RR_HIP'], a['RR_THIGH'], a['RR_CALF'],   # RR
                                a['RL_HIP'], a['RL_THIGH'], a['RL_CALF']])  # RL
            obs, rewards, done, info = env.step(action)
            env.render()
            pprint.pprint(a)
            print()


def run_test(env):
    obs = env.reset()
    env.env.sim.data.qvel[env.env.robots[0]._ref_joint_vel_indexes] = 0
    print("obs:")
    print(obs)
    for i in range(0, 50):
        # Actuator indexes: [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]
        action = np.array([0.0, 0.0, 0.0,    # FR
                           -0.0, 0.0, 0.0,   # FL
                           0.0, 0.0, 0.0,    # RR
                           -0.0, 0.0, 0.0])  # RL
        obs, rewards, done, info = env.step(action)
        torques = env.env.sim.data.qfrc_applied[:]
        env.render()
        #print(f"torques {torques}")
        #print(f"torque norm {np.linalg.norm(torques)}")
        print("---------------------")
        print("step")
        print(f"body pos: {obs[:7]}")
        print(obs)

    '''quat = [ -5.17724136e-02,  3.14450437e-05, -4.02432036e-04, 9.98658828e-01]

    from robosuite.utils.transform_utils import mat2euler, quat2mat
    mat = quat2mat(quat)
    euler = mat2euler(mat)
    print(euler)'''
       


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-train', dest='train', default=False, action="store_true",
                        help='train or run trained', required=False)
    parser.add_argument('-alg', dest='alg', default="TD3",
                        help='select training alg', required=False)
    parser.add_argument('-filename', dest='filename', default="",
                        help='set replay filename', required=False)
    parser.add_argument('-horizon', dest='horizon', default=30, type=int,
                        help='set horizon', required=False)
    parser.add_argument('-fullview', dest='fullview', default=False, action="store_true",
                        help='view type', required=False)


    arg = parser.parse_args()
    render = not arg.train
    view = "frontview"
    if arg.fullview == True:
        view = None
    env = make(render, (0, 0, 0.5), view, arg.horizon)
    if arg.train:
        if arg.alg == "TD3":
            train_TD3(env)
        elif arg.alg == "DDPG":
            train_DDPG(env)
        elif arg.alg == "SAC":
            train_SAC(env)
        else:
            raise f"Unknown algorithm provided with the -alg flag:'{arg.alg}'"
    else:
        if arg.alg == "manual":
            run_manual(env)
        elif arg.alg == "test":
            run_test(env)
        else:
            run(env, arg.alg, arg.filename)
