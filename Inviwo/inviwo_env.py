import subprocess
import sys
import collections

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np 

import inviwopy
import ivw.utils as inviwo_utils
from inviwopy.glm import ivec2, vec4
from inviwopy.data import TFPrimitiveData

import SaveTF

class InviwoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, input_image_width=128, 
                 input_image_height=128, channels=3,
                 num_steps=1000):
        
        # The input image and the transfer function
        """ More complex
        self.observation_space = spaces.Dict(
            collections.OrderedDict((
                ("Image", spaces.Box(
                    low=0, high=255, dtype=np.uint8, 
                    shape=(input_image_width,
                        input_image_height, 
                        channels)
                )),
                ("TF", spaces.Box(
                    low=0, high=255, dtype=np.uint8, shape=(256, 4)
                ))
            ))
        )
        """
        self.observation_space = spaces.Box(
                    low=0, high=255, dtype=np.uint8, 
                    shape=(input_image_width,
                        input_image_height, 
                        channels))
        """
        self.action_space = spaces.Box(
            low=0, high=255, dtype=np.uint8, shape=(256 *4,))
        """
        nvec = np.full(fill_value=256, shape=(256*4, ))
        self.action_space = spaces.MultiDiscrete(nvec)
        self.num_steps=1000
        
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
        info = {"action": action,
                "reward": reward,
                "time_step": self.time_step}

        if (self.time_step % 10 == 0):
            print("At time step {}, information is {}".format(
                self.time_step, info
            ))

        """
        ob = collections.OrderedDict((
            ("Image", self.im_data), 
            ("TF", action)))
        """
        ob = self.im_data
        return ob, reward, reset, info

    def get_reward(self):
        return -np.sum((self.im_data-self.input_data)**2)

    # Render a frame from Inviwo
    def render_inviwo_frame(self):
        inviwo_utils.update()
        network = inviwopy.app.network
        outport = network.VolumeRaycaster.getOutport("outport")
        return outport.getData().colorLayers[0].data[:, :, :3]

    # Use the action to set the transfer function in Inviwo
    # Either by saving an XML or directly in Inviwo
    def take_action(self, action):
        # It would seem the action is not being properly selected
        action_rounded = np.around(action)
        self.ivw_tf.clear()
        data_list = []
        # If properly using, this needs to be a list
        for i in range(256):
            start_idx = 4*i
            vec_list = action_rounded[start_idx:start_idx+4].copy().astype(np.float32) / float(255)
            vector = vec4(*vec_list)
            data_list.append(TFPrimitiveData((float(i) / 255), vector))
        self.ivw_tf.add(data_list)

    # Set the transfer function back to the default value and moves the camera
    def reset(self):
        self.time_step = 0
        self.ivw_tf.clear()
        out_im = np.zeros(
                shape=self.input_data.shape, dtype=np.uint8)
        out_tf = np.zeros(
                shape=self.input_data.shape, dtype=np.uint8)
        """
        return collections.OrderedDict((
            ("Image", out_im), 
            ("TF", out_tf)))
        """
        return out_im

    # If inside Inviwo should not be needed otherwise,
    # show the png image from inviwo
    def render(self, mode='human', close=False):
        pass
