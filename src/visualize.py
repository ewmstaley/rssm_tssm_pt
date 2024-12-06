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

import gymnasium as gym
import numpy as np
import torch
import os
import cv2
from box import Box
from load_atari import atari_env, get_atari_expert 
from ppo_mpi_base.core import ActorCritic
from autoencoder import CNNEncoder
from rssm import RSSM
from buffers import SubTrajectoryBuffer
from utils import collect_trajectory_from_expert
from torch.utils.tensorboard import SummaryWriter
from load_cheetah import cheetah_env, get_cheetah_expert 
from train_world_model import get_env_and_expert, get_networks

USE_TSSM = False

# get cheetah and expert
env, ac, num_actions = get_env_and_expert("cheetah")

# get a trajectory from the environment
states, actions, rgbs = collect_trajectory_from_expert(
    env=env,
    ac=ac,
    use_one_hot=False,
    max_len=100
)
batches_of_one = [[np.array([rgbs[k]]), np.array([actions[k]])] for k in range(len(rgbs))]
true_images = np.array(rgbs)[1:]

# for various world models, see how they can reconstruct this sequence
if USE_TSSM:
	dirs = ["./tssm_outboard/", "./tssm_inline/", "./tssm_inline_frozen/"]
else:
	dirs = ["./rssm_outboard/", "./rssm_inline/", "./rssm_inline_frozen/"]
dirs = [d+"cheetah/" for d in dirs]
ae_is_inline = [False, True, True]
predictions = []

for i in range(3):

    cfg = Box()
    cfg.K = 256
    cfg.WM_CLASS = "TSSM" if USE_TSSM else"RSSM"
    cfg.INLINE_AE = ae_is_inline[i]
    cfg.TRAIN_STEPS = 0
    cfg.EXTERNALLY_OPTIMIZE_AE = False

    AE, wm, device = get_networks(cfg, num_actions)
    wm.load_models(dirs[i])

    if not ae_is_inline[i]:
        AE.load_state_dict(torch.load(dirs[i]+"ae.pt"))

    # run on images and actions, only observing the first 10 states
    predicted_states = wm.rollout_with_examples(batches_of_one, observable_states=10, as_eval=True)
    predicted_states = np.array(predicted_states)

    if not cfg.INLINE_AE:
        x = np.transpose(predicted_states, (1,0,2))
        x = torch.tensor(x).to(torch.float32).to(device)
        AE.eval()
        recons = torch.sigmoid(AE.decode(x))
        predicted_states = recons.detach().cpu().numpy()
    else:
    	predicted_states = predicted_states.squeeze()

    predictions.append(predicted_states)


# make a strip image
strip = np.zeros((3,96*4,96*10)) # [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
for i in range(10):
	strip[:,0:96,96*i:96*(i+1)] = true_images[i*10]
	strip[:,96:96*2,96*i:96*(i+1)] = predictions[0][i*10]
	strip[:,96*2:96*3,96*i:96*(i+1)] = predictions[1][i*10]
	strip[:,96*3:96*4,96*i:96*(i+1)] = predictions[2][i*10]

strip = np.transpose(strip, (1,2,0))

# converts to a mat that cv2 is happy with
strip = cv2.resize(strip, (strip.shape[1], strip.shape[0]), interpolation=cv2.INTER_NEAREST)

cv2.putText(strip, "Truth", (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)
cv2.putText(strip, "Ext. AE", (5,96+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)
cv2.putText(strip, "Int. AE", (5,96*2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)
cv2.putText(strip, "Int/Ext AE", (5,96*3+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)

for i in range(1,10):
	cv2.putText(strip, "+"+str(i*10), (96*i + 25,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 1)

strip = strip[:,:,::-1]
cv2.imshow("display", strip)
cv2.imwrite('./visualization.png', strip*255)


# now we have four collections of rgb images size (99, 3, 96, 96)
# we want to stitch into single images size (99, 3, 192, 192)
full_imgs = np.zeros((99,3,192,192))
full_imgs[:,:,:96,:96] = true_images
full_imgs[:,:,:96,96:] = predictions[0]
full_imgs[:,:,96:,:96] = predictions[1]
full_imgs[:,:,96:,96:] = predictions[2]

for i in range(99):
	img = full_imgs[i]
	img = np.transpose(img, (1,2,0))
	img = cv2.resize(img, (192*3, 192*3), interpolation=cv2.INTER_NEAREST)

	clr = (0,0,0)

	cv2.putText(img, "Truth", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
	cv2.putText(img, "Ext. AE", (10+96*3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, clr, 2)
	cv2.putText(img, "Int. AE", (10,30+96*3), cv2.FONT_HERSHEY_SIMPLEX, 1, clr, 2)
	cv2.putText(img, "Int/Ext AE", (10+96*3,30+96*3), cv2.FONT_HERSHEY_SIMPLEX, 1, clr, 2)

	k = i-9
	if k < 0:
		fnum = f'{k:03}'
		clr = (0,0,0)
	else:
		fnum = f'+{k:02}' 
		clr = (0,0,255)
	cv2.putText(img, fnum, (192*3 - 70,30), cv2.FONT_HERSHEY_SIMPLEX, 1, clr, 2)

	cv2.imshow("display", img[:,:,::-1])

	if i in [0,98]:
		cv2.waitKey(1000)
	else:
		cv2.waitKey(150)