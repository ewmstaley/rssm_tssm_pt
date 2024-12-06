import gymnasium as gym
import cv2
import numpy as np
import time
import torch
import pickle
from sac_runs.train import make_policy_net, make_Q_net
from sac_base.wrappers import GymnasiumToGymWrapper
from sac_base.core import ActorCritic, SquashedGaussianActor
from pixel_wrapper import MujocoRenderObservationWrapper

STATE_STACK = 1
IMAGE_STATES = True

# make the env
env = gym.make("HalfCheetah-v4", render_mode="rgb_array")

original_state_size = env.observation_space.shape[0]
original_state_space = env.observation_space
if STATE_STACK > 1:
    env = gym.wrappers.FrameStack(env, STATE_STACK)
    env = gym.wrappers.FlattenObservation(env)
env = GymnasiumToGymWrapper(env)

# make network
device = torch.device("cuda")
ac = ActorCritic(original_state_space, env.action_space, make_policy_net, make_Q_net)
ac.load_state_dict(torch.load("./sac_runs/output/models/cheetah/model.pt"))
ac = ac.to(device)

# std dev multiplier
std_dev_factor = 1.0

# play
all_trajectories = []
rewards = []
for episode in range(50):
	s = env.reset()
	rtotal = 0
	done = False
	stepno = 0
	trajectory = []
	while not done and stepno < 100:
		x = env.render()
		x = cv2.resize(x, (64, 64))
		x = x.astype(np.float32) / 255.0
		x = np.transpose(x, (2,0,1))

		if STATE_STACK > 1:
			s_agent = s[-original_state_size:]
		else:
			s_agent = s

		_, _, dist = ac.pi(torch.from_numpy(s_agent).to(torch.float32).to(device), return_distribution=True)
		mu, std = dist.mean, dist.stddev
		if std_dev_factor > 0.0:
			dist = torch.distributions.normal.Normal(mu, std*std_dev_factor)
			a = dist.sample()
		else:
			a = mu
		a = torch.tanh(a)
		a = a.detach().cpu().numpy()

		ns, r, done, _ = env.step(a)
		stepno += 1
		rtotal += r

		trajectory.append({
			"state":(x if IMAGE_STATES else s),
			"action":a,
			"reward":r,
			"done":done,
			# "image":x
		})

		s = ns

		# cv2.imshow("env", x[:,:,::-1])
		# cv2.waitKey(1)

	all_trajectories.append(trajectory)
	print(episode)
	rewards.append(rtotal)

print(np.mean(rewards), np.std(rewards))

pickle.dump(all_trajectories, open( "traj_stochastic_one_images_64.p", "wb" ) )