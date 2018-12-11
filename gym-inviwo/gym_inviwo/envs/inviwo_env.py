import subprocess
import sys

sys.path.insert(0, "/home/sean/py_environments/gym/lib/python3.5/site-packages")

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np 

import inviwopy
import ivw.utils as inviwo_utils
from inviwopy.glm import ivec2, vec4

import SaveTF

class InviwoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, input_image_width=128, 
                 input_image_height=128, channels=3):
        
        # The input image and the transfer function
        self.observation_space = spaces.Dict({
            'input_image': spaces.Box(
                low=0, high=255, dtype=np.uint8, 
                shape=(input_image_width,
                    input_image_height, 
                    channels)),
            'rgba_tf': spaces.Box(
                low=0, high=255, dtype=np.uint8, shape=(256, 4))
        })
        self.action_space = spaces.Box(
            low=0, high=255, dtype=np.uint8, shape=(256, 4))
        
        # Inviwo init
        network = inviwopy.app.network
        canvases = network.canvases
        for canvas in canvases:
            canvas.inputSize.dimensions.value = ivec2(
                input_image_width, input_image_height)
        self.ivw_tf = network.VolumeRaycaster.isotfComposite.transferFunction
        self.input_data = self.render_inviwo_frame()

        self.reset()

    def step(self, action):
        self.take_action(action)
        self.time_step += 1

        # Perform inviwo rendering
        self.im_data = self.render_inviwo_frame()
        reward = self.get_reward()
        reset = (self.num_steps is self.time_step)

        # Dict of debug information
        info = {}

        ob = (self.im_data, action)
        return ob, reward, reset, info

    def get_reward(self):
        return np.sum((self.im_data-self.input_data)**2)

    # Render a frame from Inviwo
    def render_inviwo_frame(self):
        inviwo_utils.update()
        network = inviwopy.app.network
        outport = network.VolumeRaycaster.getOutport("outport")
        return outport.getData()

    # Use the action to set the transfer function in Inviwo
    # Either by saving an XML or directly in Inviwo
    def take_action(self, action):
        self.ivw_tf.clear()
        for i, val in enumerate(action):
            vector = vec4(*val)
            self.ivw_tf.add(float(i) / len(action), vector)

    # Set the transfer function back to the default value
    def reset(self):
        self.time_step = 0
        self.ivw_tf.clear()

    # If inside Inviwo should not be needed otherwise,
    # show the png image from inviwo
    def render(self, mode='human', close=False):
        pass
