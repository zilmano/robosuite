#!/bin/bash

source ../bin/activate
export LD_LIBRARY_PATH=/u/ozilman/.mujoco/mujoco200/bin:/u/ozilman/lib/osmesa6-dev/lib/x86_64-linux-gnu/7:/u/ozilman/lib/x86_64-linux-gnu

python3 RL.py -train -alg SAC -horizon 100
