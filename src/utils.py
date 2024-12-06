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

import torch
import numpy as np
import scipy.signal
import cv2
import math
import gymnasium as gym
import copy

def totensors(dtype, device, *args):
    return [torch.tensor(x).to(dtype).to(device) for x in args]


def grad_penalty_loss(real, fake, discriminator):
    B = real.shape[0]
    interps = torch.rand(B, 1).to(fake.device)
    inbetween_samples = (real - fake)*interps + fake
    inbetween_samples = torch.autograd.Variable(inbetween_samples, requires_grad=True).to(fake.device)
    inbetween_pred_labels = discriminator(inbetween_samples)
    input_grads = torch.autograd.grad(
        outputs=inbetween_pred_labels, 
        inputs=inbetween_samples, 
        grad_outputs=torch.ones_like(inbetween_pred_labels),
        create_graph=True,
        retain_graph=True
    )[0]
    gradients_norm = torch.sqrt(torch.sum(input_grads ** 2, dim=1) + 1e-12)
    gp_loss = ((gradients_norm - 1) ** 2).mean()
    return gp_loss


def smooth(x, window=10):
    smoothed = np.convolve(x, np.ones(window), 'valid') / window
    return smoothed


def basic_rollout(env, policy, device, max_len=100):
    # plays out an episode and returns list of tuples like:
    # (s, a, r, ns, done)

    s = env.reset()

    # collections of all states, actions, imgs, etc
    tuples = []
    done = False
    eplen = 0
    rew = 0
    while not done:

        s_torch = torch.tensor(s).to(torch.float32).to(device)
        a = policy(s_torch).detach().cpu().numpy()
        
        ns, r, done, _ = env.step(np.clip(a, -1.0, 1.0))
        rew += r
        eplen += 1
        if eplen >= max_len:
            done = True

        # store
        tuples.append((s,a,r,ns,done))
        s = ns

    return tuples, rew



def collect_trajectory_from_expert(env, ac, use_one_hot=False, max_len=None):
    s = env.reset()
    done = False
    states = []
    actions = []
    rgbs = []
    if max_len is None:
        max_len = 9999999
    while not done and len(states)<max_len:
        states.append(s)
        s = torch.tensor(s).to(torch.float32)
        a = ac.act(s)

        # one hot
        if use_one_hot:
            num_actions = env.action_space.n
            one_hot = np.zeros((num_actions,))
            one_hot[a] = 1.0
            actions.append(one_hot)
        else:
            actions.append(a)

        s, r, done, info = env.step(a)
        rgbs.append(info["saved_observation"])
    
    # duplicate state 1 since we didnt get it on reset, and remove end
    rgbs = [rgbs[0]] + rgbs[:-1]

    # (210,160,3) -> (96,96,3)
    rgbs = [cv2.resize(x, (96,96)) for x in rgbs]

    # normalize
    rgbs = (np.array(rgbs).astype(np.float32)) / 255.0
    rgbs = np.transpose(rgbs, (0,3,1,2)) # -> (B,3,96,96)

    return states, actions, rgbs


def display_image_stacks(images, fout=None):
    # assume shape is (B, seq, ch, W, H)
    B, seq, ch, w, _ = images.shape

    # is this data already normalized 0 to 1?
    if np.max(images) > 1.0:
        images = images.astype(np.float32) / 255.0

    # put into (w, h, c) order
    images = np.transpose(images, (0,1,4,3,2))

    # make an array to put them in
    img = np.zeros((B*w, seq*w, 3))
    for b in range(B):
        for s in range(seq):
            frame = images[b, s, :, :, :]
            frame = np.transpose(frame, (1,0,2))
            img[b*w:(b+1)*w, s*w:(s+1)*w, :] = frame

    img = img[:,:,::-1]
    if fout is not None:
        cv2.imwrite(fout, img*255)
        cv2.imshow(fout, img)
    else:
        cv2.imshow("display", img)

    cv2.waitKey(1)



def display_image_grid(images, fout=None):
    # assume shape is (B, ch, W, H)
    B, ch, w, _ = images.shape

    # is this data already normalized 0 to 1?
    if np.max(images) > 1.0:
        images = images.astype(np.float32) / 255.0

    # put into (w, h, c) order
    images = np.transpose(images, (0,2,3,1))

    # make an array to put them in
    side = int(math.sqrt(B))
    img = np.zeros((side*w, side*w, 3))
    i = 0
    for x in range(side):
        for y in range(side):
            frame = images[i,:,:,:]
            # frame = np.transpose(frame, (1,0,2))
            img[x*w:(x+1)*w, y*w:(y+1)*w, :] = frame
            i+=1

    img = img[:,:,::-1]
    if fout is not None:
        cv2.imwrite(fout, img*255)
        cv2.imshow(fout, img)
    else:
        cv2.imshow("display", img)

    cv2.waitKey(1)



# puts a copy of the current observation into info, so we can retrieve it later
# for example, we could store an RGB obs here before grayscaling is applied.
class ObservationToInfoWrapper(gym.Wrapper):

    def __init__(self, env, key="saved_observation", use_render=False):
        super().__init__(env)
        self.key = key
        self.use_render = use_render

    def reset(self, seed=None, options=None):
        s, info = self.env.reset(seed=seed, options=options)
        if self.use_render:
            ren = self.env.render()
            info[self.key] = copy.deepcopy(ren)
        else:
            info[self.key] = copy.deepcopy(s)
        return s, info

    def step(self, action):
        s, r, t1, t2, info = self.env.step(action)
        if self.use_render:
            ren = self.env.render()
            info[self.key] = copy.deepcopy(ren)
        else:
            info[self.key] = copy.deepcopy(s)
        return s, r, t1, t2, info
