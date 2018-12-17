import sys

sys.path.insert(0, "/home/sean/py_environments/openai_baslines/lib/python3.5/site-packages")
sys.path.insert(0, "/home/sean/gym")
sys.path.insert(0, 
"/home/sean/TransferFunctionLearning/Inviwo")

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from inviwo_env import InviwoEnv

def main(num_steps):
    env = InviwoEnv()
    env = DummyVecEnv([lambda: env])
    train_ppo2(env, num_steps)

# trains ppo2 for a given environment and number of steps
def train_ppo2(env, num_steps):
    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=num_steps)
    model.save("ppo2_tf")

if __name__ == "__main__":
    NUM_TRAINING_STEPS=1000
    main(NUM_TRAINING_STEPS)
    