import sys

sys.path.insert(0, "/home/sean/py_environments/openai_baslines/lib/python3.5/site-packages")
sys.path.insert(0, "/home/sean/gym")
sys.path.insert(0, 
"/home/sean/TransferFunctionLearning/Inviwo")

import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from inviwo_env import InviwoEnv

env = InviwoEnv()
env = DummyVecEnv([lambda: env]) 

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(info)
    env.render()