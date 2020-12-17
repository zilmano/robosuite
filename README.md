# Adding Quadruped locomotion to robosuite 

[**[Homepage]**](https://robosuite.ai/) &ensp; [**[White Paper]**](https://arxiv.org/abs/2009.12293) &ensp; [**[Documentations]**](https://robosuite.ai/docs/overview.html) &ensp; [**[ARISE Initiative]**](https://github.com/ARISE-Initiative)

-------

This **robosuite** fork contains the robosuite locomotion project for CS391 Robot Learning.

* Added Support for quadruped locomotion simulation
* Added two quadruped robots A1 and Laikago
* Added tasks, Walk, StandUp, and ClimbStairs (reward function not be set at all for ClimbStairs, for the two other ones reward shaping requires tuning)

Since this is a robosuite fork, set-up the code and install just like the origial repository, please see here for details:
https://robosuite.ai/docs/installation.html

In order to use the new robots, you can put the robot names "Laikago" or "A1", in the "robots" argument of the suite.make function.
The robots "A1" and "Laikago" will work only with the environment "Walk","StandUp" and "ClimbStairs". The other environments are intended for armed robots.
Here is a minimalistic example:

```
import numpy as np
import robosuite as suite

# create environment instance
env = suite.make(
    env_name="Walk", # try with other tasks like "StandUp" or "ClimbStairs"
    robots="A1",  # try with "Laikago"
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# reset the environment
env.reset()

for i in range(1000):
    action = np.random.randn(env.robots[0].dof) # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment
    env.render()  # render on display
```


The RL part was done with the script './robosutie/test_rl/RL.py'. Nothing fancy there, but in order to train an agent with max horizon of 100 the SAC algorithm do:

```python RL.py -train -alg SAC -horizon 100```

To modify the environment and/or robot change the 'env_name' and 'robots' variables in the suite.make function inside RL.py. 

To run a execution and visualization of the policy learned with the previous command do:
```python RL.py -alg SAC -horizon```

To run an execution of some other save model do:
```python RL.py -alg SAC  -filename <path to pickle zip file> -horizon <X>```

The Reinforcement Learning was done using the stable-baselines3 library, so you'll need to install it first in order to run the script above:
https://stable-baselines3.readthedocs.io/en/master/guide/install.html
