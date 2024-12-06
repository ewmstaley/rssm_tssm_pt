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
from itertools import chain
from utils import totensors, display_image_stacks


class Encoder(torch.nn.Module):
    def __init__(self, state_size, bottleneck, layer_width=256):
        super().__init__()
        self.enc1 = torch.nn.Linear(state_size, layer_width)
        self.enc2 = torch.nn.Linear(layer_width, bottleneck)
        self.dec1 = torch.nn.Linear(bottleneck, layer_width)
        self.dec2 = torch.nn.Linear(layer_width, state_size)

    def encode(self, o):
        return self.enc2(torch.nn.functional.relu(self.enc1(o)))

    def decode(self, z):
        return self.dec2(torch.nn.functional.relu(self.dec1(z)))


class RSSMNet(torch.nn.Module):
    def __init__(self, h_size, s_size, action_size, e_size, layer_width=256, posterior_on_h=True):
        super().__init__()

        # activation
        self.act = torch.nn.functional.elu

        # recurrent model (h, s, a) -> h'
        self.fc1 = torch.nn.Linear(h_size+s_size+action_size, layer_width)
        self.gru = torch.nn.GRUCell(layer_width, h_size)

        # transition model (h',) -> gaussian(s') , the "prior"
        self.tfc1 = torch.nn.Linear(h_size, layer_width)
        self.tfc2 = torch.nn.Linear(layer_width, layer_width)
        self.t_mean = torch.nn.Linear(layer_width, s_size)
        self.t_std = torch.nn.Linear(layer_width, s_size)

        # representation model (h', e) -> gaussian(s') , the "posterior"
        # e is an encoding of the true current state after the transition
        # if not posterior_on_h, this is disconnected: (e,) -> gaussian(s')
        if posterior_on_h:
            self.rfc1 = torch.nn.Linear(h_size+e_size, layer_width)
        else:
            self.rfc1 = torch.nn.Linear(e_size, layer_width)
        self.rfc2 = torch.nn.Linear(layer_width, layer_width)
        self.r_mean = torch.nn.Linear(layer_width, s_size)
        self.r_std = torch.nn.Linear(layer_width, s_size)
        self.posterior_on_h = posterior_on_h

    def make_normal(self, mean, std):
        std = torch.nn.functional.softplus(std) + 0.01
        return torch.distributions.Normal(mean, std)

    def forward(self, h, s, a, e=None):

        # recurrent
        x = torch.cat([h, s, a], dim=-1)
        x = self.act(self.fc1(x))
        h = self.gru(x, h)

        # transition
        s_prior = self.act(self.tfc1(h))
        s_prior = self.act(self.tfc2(s_prior))
        s_prior_mean = self.t_mean(s_prior)
        s_prior_std = self.t_std(s_prior)
        prior = self.make_normal(s_prior_mean, s_prior_std)

        # representation
        if e is not None:
            if self.posterior_on_h:
                post_in = torch.cat([h, e], dim=-1)
            else:
                post_in = e
            s_post = self.act(self.rfc1(post_in))
            s_post = self.act(self.rfc2(s_post))
            s_post_mean = self.r_mean(s_post)
            s_post_std = self.r_std(s_post)
            post = self.make_normal(s_post_mean, s_post_std)
        else:
            post = None

        return h, prior, post



class RSSM():
    '''
    Recurrent State-Space Model (RSSM)

    The core model operates on latent vector representations, and can be used a few different ways:

    By default (no preprocessor and no supplied encoder), the states are assumes to be vectors, i.e. proprioception.
    
    If the states are images or something else, they can be reduced to latent vectors externally, in which
    case the preprocessor method should be given. Gradients will not propogate into this method.
    In this case, an internal encoder will still be used to convert between the preprocessed states and 
    a representation for the RSSM to use.
    
    If you DO want backprop through this projection, you can supply a custom encoder model with supplied_encoder.
    This must implement encode() and decode().
    '''

    def __init__(self, 
            peak_lr,                # peak learning rate for models
            expected_num_opt_steps, # total number of updates, for lr decay. Large number to effectively not decay.
            device,                 # device reference
            state_size,             # size of flat states incoming and outgoing from this model
            action_size,            # dimensionality of the actions
            latent_size,            # latent size for internal representations (h, s, and e)
            supplied_encoder=None,  # override the default encoder. For example, could supply a CNN encoder and image states.
            preprocessor=None,      # external pre-processing of states method. For example, use a CNN externally.
            layer_width=None,       # hidden layer sizes. By default, same as latent_size.
            posterior_on_h=True,    # whether to condition the posterior on the hidden state
            optimize_supplied=True, # whether to optimize the supplied encoder or not
        ):

        super().__init__()

        # capture some specific sizes
        h_size = latent_size
        s_size = latent_size
        e_size = latent_size
        if layer_width is None:
            layer_width = latent_size

        # all the settings
        self.lr = peak_lr
        self.expected_num_opt_steps = expected_num_opt_steps
        self.state_size = state_size
        self.h_size = h_size
        self.s_size = s_size
        self.action_size = action_size
        self.e_size = e_size
        self.device = device

        # to project from some other representation to latent
        self.preprocessor = preprocessor

        if supplied_encoder is not None:
            self.encoder = supplied_encoder
            self.state_loss = "bce"
        else:
            self.encoder = Encoder(state_size, e_size, layer_width).to(device)
            self.state_loss = "mse"

        self.model = RSSMNet(h_size, s_size, action_size, e_size, layer_width, posterior_on_h).to(device)
        if optimize_supplied:
            self.opt = torch.optim.Adam(chain(self.model.parameters(), self.encoder.parameters()), lr=peak_lr)
        else:
            self.opt = torch.optim.Adam(self.model.parameters(), lr=peak_lr)
        self.sched = torch.optim.lr_scheduler.LinearLR(self.opt, start_factor=1.0, end_factor=0.001, total_iters=expected_num_opt_steps)


    def initialize_with_state(self, state_batch):
        # initialize the beginning of trajectories with a batch of shape (B,S)
        B = state_batch.shape[0]
        h = torch.zeros(B, self.h_size).to(state_batch.device)
        s = torch.zeros(B, self.s_size).to(state_batch.device)
        a = torch.zeros(B, self.action_size).to(state_batch.device)
        e = self.encoder.encode(state_batch)
        h, prior, post = self.model(h, s, a, e)
        s = post.rsample()
        return h, s


    def save_models(self, directory):
        torch.save(self.model.state_dict(), directory+"rssm.pt")
        torch.save(self.encoder.state_dict(), directory+"rssm_encoder.pt")


    def load_models(self, directory):
        self.model.load_state_dict(torch.load(directory+"rssm.pt"))
        self.encoder.load_state_dict(torch.load(directory+"rssm_encoder.pt"))


    def update_from_examples(self, examples):
        # given a sequence of batch tuples, update the model
        # i.e. [[(B,S),(B,A)], [(B,S),(B,A)], [(B,S),(B,A)], ...]
        
        batch_size = examples[0][0].shape[0]
        roll = len(examples)
        self.opt.zero_grad()
        state_loss, kl_loss, predicted_states = self.rollout_with_examples(examples, observable_states=roll)
        loss = kl_loss + state_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        self.opt.step()
        self.sched.step()
        return state_loss, kl_loss, predicted_states


    def rollout_with_examples(self, examples, observable_states=2, as_eval=False):
        # given a sequence of batch tuples, test the model
        # i.e. [[(B,S),(B,A)], [(B,S),(B,A)], [(B,S),(B,A)], ...]
        # observable_states: how many steps the model can observe the true state
        # returns: state_loss, kl_loss, predicted_states

        if as_eval:
            self.model.eval()
            self.encoder.eval()

        batch_size = examples[0][0].shape[0]
        roll = len(examples)
        kl_loss = 0
        state_loss = 0
        predicted_states = []

        # initialize
        # h is our deterministic hidden state, s is our stochastic hidden state
        # s is NOT the state, rather o is the observation
        o = examples[0][0]
        o = torch.tensor(o).to(torch.float32).to(self.device)
        if self.preprocessor is not None:
            with torch.no_grad():
                o = self.preprocessor(o)
        h, s = self.initialize_with_state(o)

        # rollout
        for k in range(roll-1):
            # get inputs and targets
            a = examples[k][1]
            o_next = examples[k+1][0]
            a, o_next = totensors(torch.float32, self.device, a, o_next)
            if self.preprocessor is not None:
                with torch.no_grad():
                    o_next = self.preprocessor(o_next)

            e = None
            if k < observable_states:
                e = self.encoder.encode(o_next)

            # step forward
            h, prior, post = self.model(h, s, a, e)
            if k < observable_states:
                s = post.rsample()
                pred_o = self.encoder.decode(s)
                o = o_next
            else:
                s = prior.rsample()
                pred_o = self.encoder.decode(s)

            # loss
            if k < observable_states:
                kl_loss += torch.mean(torch.distributions.kl.kl_divergence(prior, post))

            if self.state_loss == "bce":
                state_loss += torch.nn.functional.binary_cross_entropy_with_logits(pred_o, o_next)
                predicted_states.append(torch.sigmoid(pred_o).detach().cpu().numpy())
            else:
                state_loss += torch.nn.functional.mse_loss(pred_o, o_next)
                predicted_states.append(pred_o.detach().cpu().numpy())
            
        kl_loss = kl_loss / (roll-1)
        state_loss = state_loss / (roll-1)

        if as_eval:
            self.model.train()
            self.encoder.train()
            return predicted_states

        return state_loss, kl_loss, predicted_states