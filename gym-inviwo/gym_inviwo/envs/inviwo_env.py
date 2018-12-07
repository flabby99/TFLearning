import subprocess

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy

class InviwoParams():
    def __init__(self, num_steps,
        workspace_file, python_script, inviwo_exe_location):
        self.workspace_file = workspace_file
        self.python_script = python_script
        self.inviwo_exe_location = inviwo_exe_location
        self.time_step = 0
        self.num_steps = num_steps

class InviwoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, input_image_width=128, 
                 input_image_height=128, channels=3,
                 ivw_params):
        
        # The input image and the transfer function
        self.observation_space = spaces.Dict({
            'input_image': spaces.Box(
                low=0, high=255, dtype=np.uint8, 
                shape=(input_image_height, 
                       input_image_width, 
                       channels)),
            'rgba_tf': spaces.Box(
                low=0, high=255, dtype=np.unit8, shape=(256, 4))
        })

        # tf can change each position by increasing, decreasing or no-op
        tf_change_actions = np.full(shape=(256*4,), fill_value=3)
        self.action_space = spaces.MultiDiscrete(tf_change_actions)
        self.ivw_params = ivw_params

    def step(self, action):
        self.take_action(action)
        self.time_step += 1

        # Perform inviwo rendering
        self.render_inviwo_frame()
        reward = self.get_reward()
        ob = self.get_state()
        reset = (self.num_steps is self.time_step)

        # Dict of debug information
        info = {}

        return ob, reward, reset, info

    def get_state(self):
        pass


    def get_reward(self):
        pass

    # Likely to run into speed issues with this.
    # Maybe should run tf from inviwo similar to old project?
    # Will likely be awkward though.
    def render_inviwo_frame(self):
        subprocess.run("~/inviwo_build_minimalqt/bin/inviwo -n -w ~/inviwo/data/workspaces/boron.inv -p ~/TransferFunctionLearning/Inviwo/InviwoSingleFrameRender.py",
        shell=True, timeout=10)

    def take_action(self, action):
        
    # Set the transfer function back to the default value
    def reset(self):
        self.time_step = 0

    # If inside Inviwo should not be needed otherwise,
    # show the png image from inviwo
    def render(self, mode='human', close=False):
