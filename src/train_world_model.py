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
import yaml
from autoencoder import CNNEncoder
from rssm import RSSM
from tssm import TSSM
from buffers import SubTrajectoryBuffer
from utils import display_image_grid, display_image_stacks, collect_trajectory_from_expert
from torch.utils.tensorboard import SummaryWriter
from load_cheetah import cheetah_env, get_cheetah_expert 
from load_atari import atari_env, get_atari_expert 


# =======================================================

cfg = Box()

cfg.INITIAL_AE_ONLY_STEPS = 1000
cfg.TRAIN_STEPS = 20000
cfg.INLINE_AE = False
cfg.EXTERNALLY_OPTIMIZE_AE = True
cfg.INITIAL_TRAJECTORIES = 50
cfg.TRAJECTORY_MAX_LEN = 100
cfg.WM_CLASS = "TSSM"
cfg.K = 256
cfg.TRAIN_TRAJ_LENGTH = 20 if cfg.INLINE_AE else 50
cfg.TRAIN_BATCH_SIZE = 8 if cfg.INLINE_AE else 32
cfg.AE_TRAIN_BATCH_SIZE = 64
cfg.ONE_HOT = False
cfg.ENV_NAME = "cheetah"

cfg.OUTPUT_DIR = "./tssm" if (cfg.WM_CLASS=="TSSM") else "./rssm"
if cfg.INLINE_AE:
    cfg.OUTPUT_DIR += "_inline"
    if cfg.EXTERNALLY_OPTIMIZE_AE:
        cfg.OUTPUT_DIR += "_frozen"
else:
    cfg.OUTPUT_DIR += "_outboard"
cfg.OUTPUT_DIR += "/"+cfg.ENV_NAME+"/"

# =======================================================

if not os.path.exists(cfg.OUTPUT_DIR):
    os.makedirs(cfg.OUTPUT_DIR)

with open(cfg.OUTPUT_DIR+'config.yml', 'w') as outfile:
    yaml.dump(cfg.to_dict(), outfile)

# =======================================================


def get_env_and_expert(envname):
    # only supports half cheetah, breakout, and pacman currently
    if envname == "cheetah":
        env = cheetah_env()
        num_actions = env.action_space.shape[0]
        ac = get_cheetah_expert(env)
    else:
        env = atari_env(envname)
        num_actions = env.action_space.n
        ac = get_atari_expert(env, envname)
    return env, ac, num_actions


# =======================================================

# move to config instead of individual arguments
def get_networks(cfg, num_actions):
    device = torch.device("cuda")

    # build CNN
    AE = CNNEncoder(96, cfg.K, device, channels=64).to(device)

    if not cfg.INLINE_AE:

        def preprocess_encode(x):
            with torch.no_grad():
                x = AE.encode(x)
            return x

    # build RSSM that works on latent vectors (compressed via AE)
    if cfg.INLINE_AE:
        senc = AE
        preproc = None
    else:
        senc = None
        preproc = preprocess_encode

    wmcls = RSSM if cfg.WM_CLASS == "RSSM" else TSSM
    wm = wmcls(peak_lr=0.0003, expected_num_opt_steps=cfg.TRAIN_STEPS, device=device,
        state_size=cfg.K, action_size=num_actions, latent_size=cfg.K, 
        supplied_encoder=senc, preprocessor=preproc, layer_width=cfg.K, 
        optimize_supplied=(not cfg.EXTERNALLY_OPTIMIZE_AE))

    return AE, wm, device


# =======================================================


def train():
    env, ac, num_actions = get_env_and_expert(cfg.ENV_NAME)
    AE, wm, device = get_networks(cfg, num_actions)

    if cfg.EXTERNALLY_OPTIMIZE_AE:
        opt_ae = torch.optim.Adam(AE.parameters(), lr=0.0001)

    summary_writer = SummaryWriter(log_dir=cfg.OUTPUT_DIR+"logs/")

    # buffer for trajectories
    trajectory_buffer = SubTrajectoryBuffer(sizes=[(3,96,96), (num_actions,)], max_size=50000)

    # prefill a bit
    for i in range(cfg.INITIAL_TRAJECTORIES):
        states, actions, rgbs = collect_trajectory_from_expert(
            env=env,
            ac=ac,
            use_one_hot=cfg.ONE_HOT,
            max_len=cfg.TRAJECTORY_MAX_LEN
        )
        samples = [(rgbs[k], actions[k]) for k in range(len(rgbs))]
        trajectory_buffer.add_samples(samples)
        print(i+1)

    # train
    steps = cfg.TRAIN_STEPS
    if cfg.EXTERNALLY_OPTIMIZE_AE:
        steps += cfg.INITIAL_AE_ONLY_STEPS
    for i in range(steps):

        if cfg.EXTERNALLY_OPTIMIZE_AE:
            # update the ae
            opt_ae.zero_grad()
            images, _ = trajectory_buffer.sample_buffers(cfg.AE_TRAIN_BATCH_SIZE)
            images = torch.tensor(images).to(torch.float32).to(device)

            recon = AE.decode(AE.encode(images))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(recon, images)
            loss.backward()
            aeloss = loss.item()
            summary_writer.add_scalar("autoencoder loss", aeloss, i)
            print("    AE Loss:", aeloss)
            torch.nn.utils.clip_grad_norm_(AE.parameters(), 1.0)
            opt_ae.step()

            # periodically test
            if (i+1)%10 == 0:
                with torch.no_grad():
                    images, _ = trajectory_buffer.sample_buffers(16)
                    images_pt = torch.tensor(images).to(torch.float32).to(device)
                    recon = torch.sigmoid(AE.decode(AE.encode(images_pt)))
                    recon = recon.cpu().numpy()
                    display_image_grid(images, cfg.OUTPUT_DIR+"original.png")
                    display_image_grid(recon, cfg.OUTPUT_DIR+"recon.png")

        if cfg.EXTERNALLY_OPTIMIZE_AE and i < cfg.INITIAL_AE_ONLY_STEPS:
            continue

        # get batch of trajectories
        batches = trajectory_buffer.sample_subtrajectories(
            subtraj_len=cfg.TRAIN_TRAJ_LENGTH, amount=cfg.TRAIN_BATCH_SIZE, start_from_zero=False
        )
        states = batches[0][0]

        # update wm
        mse_loss, kl_loss, predicted_latent_states = wm.update_from_examples(batches)
        print(i, mse_loss.item(), kl_loss.item())
        summary_writer.add_scalar("rssm state loss", mse_loss.item(), i)
        summary_writer.add_scalar("rssm kl loss", kl_loss.item(), i)

        # display reconstructed states
        if (i+1)%10 == 0:
            with torch.no_grad():
                predicted_latent_states = np.array(predicted_latent_states)
                if not cfg.INLINE_AE:
                    # need to reconstruct first
                    predicted_latent_states = np.transpose(predicted_latent_states, (1,0,2))
                    x = predicted_latent_states[:5, :10]
                    x = torch.tensor(x).to(torch.float32).to(device)
                    AE.eval()
                    recons = torch.sigmoid(AE.decode(x))
                    AE.train()
                    recons = recons.cpu().numpy()
                    n, d1, d2, d3 = recons.shape
                    recons = np.reshape(recons, (5, 10, d1, d2, d3))
                else:
                    # print(predicted_latent_states.shape)
                    predicted_latent_states = np.transpose(predicted_latent_states, (1,0,2,3,4))
                    recons = predicted_latent_states[:5, :10]
                display_image_stacks(recons, cfg.OUTPUT_DIR+"reconstructed_sequences.png")

        # periodically get more data
        if (i+1)%100 == 0:
            states, actions, rgbs = collect_trajectory_from_expert(
                env=env,
                ac=ac,
                use_one_hot=cfg.ONE_HOT,
                max_len=cfg.TRAJECTORY_MAX_LEN
            )
            samples = [(rgbs[k], actions[k]) for k in range(len(rgbs))]
            trajectory_buffer.add_samples(samples)


    if not cfg.INLINE_AE:
        torch.save(AE.state_dict(), cfg.OUTPUT_DIR+"ae.pt")
    wm.save_models(cfg.OUTPUT_DIR)



if __name__ == "__main__":
    train()