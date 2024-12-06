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
import math
from rssm import Encoder


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, embeddings=1000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, d_model, 2)* math.log(10000) / d_model)
        pos = torch.arange(0, embeddings).reshape(embeddings, 1)
        pos_embedding = torch.zeros((embeddings, d_model))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, t):
        return self.pos_embedding[t]


class TSSMNet(torch.nn.Module):
    def __init__(self, s_size, action_size, e_size, layer_width=256):
        super().__init__()

        # activation
        self.act = torch.nn.functional.elu

        # prediction of posterior directly from encoded observations
        self.postfc1 = torch.nn.Linear(e_size, layer_width)
        self.postfc2 = torch.nn.Linear(layer_width, layer_width)
        self.post_mean = torch.nn.Linear(layer_width, s_size)
        self.post_std = torch.nn.Linear(layer_width, s_size)

        # prediction of prior from transformer output vectors
        self.priorfc1 = torch.nn.Linear(layer_width, layer_width)
        self.prior_mean = torch.nn.Linear(layer_width, s_size)
        self.prior_std = torch.nn.Linear(layer_width, s_size)

        # process and position embedding
        self.process = torch.nn.Linear(s_size+action_size, layer_width)
        self.pe = PositionalEncoding(layer_width)

        # transformer
        template = torch.nn.TransformerEncoderLayer(
            d_model=layer_width, 
            nhead=8, 
            dim_feedforward=layer_width*4, 
            batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(template, 6)


    def make_normal(self, mean, std):
        std = torch.nn.functional.softplus(std) + 0.01
        return torch.distributions.Normal(mean, std)


    def encoding_to_posterior(self, e):
        x = self.act(self.postfc1(e))
        x = self.act(self.postfc2(x))
        mu = self.post_mean(x)
        std = self.post_std(x)
        dist = self.make_normal(mu, std)
        return dist


    def forward(self, s, a, positions):
        # s : size (B, seq, s_size)
        # a : size (B, seq, action_size)
        # positions : (B, seq)

        # get single input vectors and apply position encoding
        sa = torch.cat([s, a], axis=-1)
        sa = self.process(sa)
        positions = positions[:,:,None]
        pos = self.pe(positions)
        x = sa + pos.squeeze()

        # apply transformer
        mask = torch.nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=x.device)
        output_vectors = self.transformer(x, mask=mask, is_causal=True)

        # get output distributions
        pr = self.act(self.priorfc1(output_vectors))
        pr_mu = self.prior_mean(pr)
        pr_std = self.prior_std(pr)
        dist = self.make_normal(pr_mu, pr_std)
        return dist


class TSSM():
    '''
    Transformer State-Space Model (TSSM)

    RSSM with recurrent portion replaced by a transformer. This requires that the posterior is no longer
    conditioned on the previous hidden state, but otherwise is very similar to the RSSM.

    This does not currently support the same range of features as the RSSM. It assumes incoming observations
    are already flat vectors.
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
            layer_width=None,       # hidden layer sizes. By default, same as latent_size. Functions as d_model.
            optimize_supplied=True, # whether to optimize the supplied encoder or not
            **kwargs
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

        self.model = TSSMNet(s_size, action_size, e_size, layer_width).to(device)
        if optimize_supplied:
            self.opt = torch.optim.Adam(chain(self.model.parameters(), self.encoder.parameters()), lr=peak_lr)
        else:
            self.opt = torch.optim.Adam(self.model.parameters(), lr=peak_lr)        
        self.sched = torch.optim.lr_scheduler.LinearLR(self.opt, start_factor=1.0, end_factor=0.001, total_iters=expected_num_opt_steps)


    def initialize_with_state(self, state_batch):
        pass


    def save_models(self, directory):
        torch.save(self.model.state_dict(), directory+"tssm.pt")
        torch.save(self.encoder.state_dict(), directory+"tssm_encoder.pt")


    def load_models(self, directory):
        self.model.load_state_dict(torch.load(directory+"tssm.pt"))
        self.encoder.load_state_dict(torch.load(directory+"tssm_encoder.pt"))


    def update_from_examples(self, examples):
        # given a sequence of batch tuples, update the model
        # i.e. [[(B,S),(B,A)], [(B,S),(B,A)], [(B,S),(B,A)], ...]
        # this format is same as RSSM, so using it for compability

        batch_size = examples[0][0].shape[0]
        roll = len(examples)
        self.opt.zero_grad()
        state_loss, kl_loss, predicted_states = self.pass_with_examples(examples)
        loss = kl_loss + state_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 1.0)
        self.opt.step()
        self.sched.step()
        return state_loss, kl_loss, predicted_states


    def prepare_examples_for_forward(self, examples):

        # get obs sequence and action sequence
        obs_seq = [x[0] for x in examples]
        obs_seq = [torch.tensor(x).to(torch.float32).to(self.device) for x in obs_seq]
        if self.preprocessor is not None:
            with torch.no_grad():
                obs_seq = [self.preprocessor(o) for o in obs_seq]
        obs_seq = torch.stack(obs_seq, dim=1) # (B, seq, o_size)

        act_seq = [x[1] for x in examples]
        act_seq = [torch.tensor(x).to(torch.float32).to(self.device) for x in act_seq]
        act_seq = torch.stack(act_seq, dim=1) # (B, seq, a_size)

        batch_size = examples[0][0].shape[0]
        seq_size = len(examples)

        # get the position ids
        positions = torch.arange(seq_size).to(torch.long).to(obs_seq.device)
        positions = positions[None, :]
        positions = positions.repeat(batch_size, 1)

        return obs_seq, act_seq, batch_size, seq_size, positions


    def pass_with_examples(self, examples):
        # given a sequence of batch tuples, test the model
        # i.e. [[(B,S),(B,A)], [(B,S),(B,A)], [(B,S),(B,A)], ...]
        # returns: state_loss, kl_loss, predicted_states

        obs_seq, act_seq, batch_size, seq_size, positions = self.prepare_examples_for_forward(examples)
        obs_seq_unshifted = obs_seq[:,:-1]
        obs_seq_target = obs_seq[:,1:]

        # encode all the obs and get "posteriors"
        need_batch_sequence_reshape = False
        if len(obs_seq.shape)==5:
            # if using images, collapse first two dimensions to make convs happy
            _, _, d3, d4, d5 = obs_seq.shape
            obs_seq = torch.reshape(obs_seq, (batch_size*seq_size, d3, d4, d5))
            es = self.encoder.encode(obs_seq)
            # reconstruct the batch and sequence dimensions
            es = torch.reshape(es, (batch_size, seq_size, -1))
            need_batch_sequence_reshape = True
        else:
            es = self.encoder.encode(obs_seq)

        es_input = es[:,:-1,:] # first N-1
        es_target = es[:,1:,:] # last N-1

        post_input = self.model.encoding_to_posterior(es_input)
        post_target = self.model.encoding_to_posterior(es_target)

        predicted_states = []

        # pass through and get "priors"
        s_post = post_input.rsample()
        prior = self.model.forward(s_post, act_seq[:,:-1], positions[:,:-1])

        # kld between the distributions (SHIFTED!)
        kl_loss = torch.mean(torch.distributions.kl.kl_divergence(prior, post_target))

        # sample the posteriors and decode, this should equal the original obs (NOT SHIFTED!)
        # (MSE loss)
        recon = self.encoder.decode(s_post)
        if self.state_loss == "bce":
            recon = torch.reshape(recon, obs_seq_unshifted.shape)
            state_loss = torch.nn.functional.binary_cross_entropy_with_logits(recon, obs_seq_unshifted)
        else:
            state_loss = torch.nn.functional.mse_loss(recon, obs_seq_unshifted)

        # sample the priors and decode, these will be our predicted states
        # note this differs from RSSM, which returns posteriors here.
        s_prior = prior.rsample()
        pred_o = self.encoder.decode(s_prior)
        if self.state_loss == "bce":
            pred_o = torch.sigmoid(pred_o)
            if need_batch_sequence_reshape:
                _, d3, d4, d5 = pred_o.shape
                pred_o = torch.reshape(pred_o, (batch_size, seq_size-1, d3, d4, d5))
        predicted_states = pred_o.detach().cpu().numpy()
            
        return state_loss, kl_loss, predicted_states



    # for training, we use a single pass (above), but for testing it can be
    # helpful to actually roll forward. This breaks inference into a few steps
    # and is not compatible with training.
    def rollout_with_examples(self, examples, observable_states=10, **kwargs):
        # given a sequence of batch tuples, test the model
        # i.e. [[(B,S),(B,A)], [(B,S),(B,A)], [(B,S),(B,A)], ...]
        # observable_states: how many steps the model can observe the true state
        # returns: predicted_states

        self.model.eval()
        self.encoder.eval()

        obs_seq, act_seq, batch_size, seq_size, positions = self.prepare_examples_for_forward(examples)

        # encode all the obs and get "posteriors"
        need_batch_sequence_reshape = False
        if len(obs_seq.shape)==5:
            # if using images, collapse first two dimensions to make convs happy
            _, _, d3, d4, d5 = obs_seq.shape
            obs_seq = torch.reshape(obs_seq, (batch_size*seq_size, d3, d4, d5))
            es = self.encoder.encode(obs_seq)
            # reconstruct the batch and sequence dimensions
            es = torch.reshape(es, (batch_size, seq_size, -1))
            need_batch_sequence_reshape = True
        else:
            es = self.encoder.encode(obs_seq)
        post = self.model.encoding_to_posterior(es)
        s_post = post.rsample()

        predicted_states = []

        # roll
        sequence_s = s_post[:,:observable_states]
        sequence_a = act_seq[:,:observable_states]
        sequence_p = positions[:,:observable_states]

        for i in range(seq_size - observable_states):

            # run the latest sequence forward
            prior = self.model.forward(sequence_s, sequence_a, sequence_p)

            # get the last prior
            s_prior = prior.rsample()
            s_prior = s_prior[:,-1]

            # put the sequence dimension back
            s_prior = s_prior[:,:,None]
            s_prior = torch.transpose(s_prior, 1, 2)

            # add it to s sequences
            sequence_s = torch.cat([sequence_s, s_prior], dim=1)

            # if they exist, get next actions and positions
            sequence_a = act_seq[:,:observable_states+i+1]
            sequence_p = positions[:,:observable_states+i+1]

        # decode the s sequence to get all predicted observations
        pred_o = self.encoder.decode(sequence_s)
        if self.state_loss == "bce":
            pred_o = torch.sigmoid(pred_o)
            if need_batch_sequence_reshape:
                _, d3, d4, d5 = pred_o.shape
                pred_o = torch.reshape(pred_o, (batch_size, seq_size, d3, d4, d5))
        predicted_states = pred_o.detach().cpu().numpy()
        predicted_states = predicted_states[:,1:]
        print(predicted_states.shape)

        return predicted_states