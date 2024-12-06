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
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

from sac_base.sac import SAC
from sac_base.wrappers import GymnasiumToGymWrapper
import gymnasium as gym
import torch

class YNet(torch.nn.Module):
	def __init__(self, obs, act, net_size):
		super().__init__()
		self.fc1 = torch.nn.Linear(obs, net_size)
		self.fc2 = torch.nn.Linear(net_size, net_size)
		self.mu_out = torch.nn.Linear(net_size, act)
		self.log_std_out = torch.nn.Linear(net_size, act)

	def forward(self, x):
		x = torch.nn.functional.relu(self.fc1(x))
		x = torch.nn.functional.relu(self.fc2(x))
		return self.mu_out(x), self.log_std_out(x)


NET_SIZE = 256

# obs -> (act, act)
def make_policy_net(obs, act):
    return YNet(obs, act, NET_SIZE)

# [obs + act] -> 1
def make_Q_net(obs, act):
    model = torch.nn.Sequential(
          torch.nn.Linear(obs+act, NET_SIZE),
          torch.nn.ReLU(),
          torch.nn.Linear(NET_SIZE, NET_SIZE),
          torch.nn.ReLU(),
          torch.nn.Linear(NET_SIZE, 1)
        )
    return model


def env_fn():
	env = gym.make("HalfCheetah-v4")
	# old gym interface
	env = GymnasiumToGymWrapper(env)
	return env


if __name__ == "__main__":
    SAC(
    	env_fn, 
    	make_policy_net, 
    	make_Q_net,
    	training_steps=1e6,
    	seed=0, 
        replay_size=int(1e6), 
        gamma=0.99, 
        polyak=0.995, 
        lr=1e-3,
        alpha=0.2, 
        alpha_is_tuned=True,
        batch_size=128, 
        start_steps=10000, 
        update_after=1000, 
        update_every=50, 
        num_test_episodes=10, 
        max_ep_len=1000, 
        save_freq=1,
        logs_dir="./output/logs/cheetah",
        save_dir="./output/models/cheetah",
        checkpoint_every=100000
    )