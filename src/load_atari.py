'''
Copyright © 2024 The Johns Hopkins University Applied Physics Laboratory LLC
 
Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the “Software”), to 
deal in the Software without restriction, including without limitation the 
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT R OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import gymnasium as gym
import numpy as np
import torch
from ppo_mpi_base.ppo import PPO
from ppo_mpi_base.wrappers import GymnasiumToGymWrapper
from ppo_mpi_base.core import ActorCritic
from utils import ObservationToInfoWrapper

# generic CNN that we can use
class CNNNetwork(torch.nn.Module):
    def __init__(self, output_size, final_gain=1.0):
        super().__init__()

        # from ray's setup
        self.conv1 = torch.nn.Conv2d(4, 16, kernel_size=4, stride=2)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv4 = torch.nn.Conv2d(64, 128, kernel_size=4, stride=2)
        self.fc1 = torch.nn.Linear(128*9, 512)
        self.fc2 = torch.nn.Linear(512, output_size)

        torch.nn.init.orthogonal_(self.conv1.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.conv2.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.conv3.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.conv4.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        torch.nn.init.orthogonal_(self.fc2.weight, gain=final_gain)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.conv4(x))
        x = torch.flatten(x, 1) if len(x.shape)==4 else torch.flatten(x, 0)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# define our environment
def atari_env(name):

    if name=="breakout":
        env = gym.make("BreakoutNoFrameskip-v4")
    else:
        env = gym.make("MsPacmanNoFrameskip-v4")
    env = ObservationToInfoWrapper(env)
    env = gym.wrappers.atari_preprocessing.AtariPreprocessing(
        env, 
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=True
    )

    env = gym.wrappers.FrameStack(env, 4)

    # old gym interface
    env = GymnasiumToGymWrapper(env)

    return env


# define a policy network architecture
def policy_net_fn(obs, act):
    return CNNNetwork(act, 0.01)


# define a value network architecture
def value_net_fn(obs):
    return CNNNetwork(1, 1.0)


def get_atari_expert(env, name):
    ac = ActorCritic(env.observation_space, env.action_space, policy_net_fn, value_net_fn)
    ac.load_state_dict(torch.load("./experts/saved_model/"+name+"/model_latest.pt", weights_only=True))
    return ac