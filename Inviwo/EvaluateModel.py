import sys

sys.path.insert(0, "/home/sean/py_environments/openai_baslines/lib/python3.5/site-packages")
sys.path.insert(0, "/home/sean/gym")
sys.path.insert(0, 
"/home/sean/TransferFunctionLearning/Inviwo")

import numpy as np
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from inviwo_env import InviwoEnv

def main(model_name, num_steps):
    # Evaluate a completely random model
    env = InviwoEnv()
    env = DummyVecEnv([lambda: env])
    if model_name == "random":
        model = PPO2(MlpPolicy, env, verbose=1)
    else:
        model = PPO2.load(model_name, env)
    rewards = evaluate(model, num_steps, env)
    print(model_name, rewards)

def evaluate(model, steps, env):
    episode_rewards = []
    obs = env.reset()
    for _ in range(steps):
        action, _states = model.predict(obs)

        #Actions are not clipped during running - only training
        action = np.clip(action, 0, 1)
        
        obs, reward, reset, info = env.step(action)
        
        env.render()

        if reset:
            episode_rewards.append(reward)
            obs = env.reset()
    episode_rewards.append(reward)
    return episode_rewards

if __name__ == "__main__":
    MODEL_NAME = "ppo2_tf"
    NUM_STEPS = 100
    main(MODEL_NAME, NUM_STEPS)