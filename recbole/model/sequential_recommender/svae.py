# -*- coding: utf-8 -*-
# @Time    : 2021/11/17 11:46
# @Author  : Alejandro Ariza Casabona
# @Email   : alejandro.ariza14@ub.edu

"""
SVAE
################################################
Note:
    In the original code, batch size is always 1 (one user at a time)

Reference:
    Noveen Sachdeva et al. "Sequential Variational Autoencoders for Collaborative Filtering" in WSDM 2019.

Reference:
    https://github.com/noveens/svae_cf

"""

import torch
import numpy as np

from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from recbole.utils import InputType
from recbole.model.abstract_recommender import SequentialRecommender


class Encoder(nn.Module):
    def __init__(self, rnn_size, hidden_size):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(
            rnn_size, hidden_size
        )
        nn.init.xavier_normal_(self.linear1.weight)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class SVAE(SequentialRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(SVAE, self).__init__(config, dataset)

        self.item_embed_size = config['embedding_size']
        self.rnn_size = config['rnn_size']
        self.hidden_size = config['hidden_size']
        self.latent_size = config['latent_size']
        self.anneal_cap = config['anneal_cap']
        self.total_anneal_steps = config["total_anneal_steps"]
        self.update = 0

        self.encoder = Encoder(self.rnn_size, self.hidden_size)
        self.decoder = Decoder(self.latent_size, self.hidden_size, self.n_items)

        # Since we don't need padding, our vocab_size = "hyper_params['total_items']" and not "hyper_params['total_items'] + 1"
        self.item_embed = nn.Embedding(self.n_items, self.item_embed_size, padding_idx=0)

        self.gru = nn.GRU(
            self.item_embed_size, self.rnn_size,
            batch_first=True, num_layers=1
        )

        self.linear1 = nn.Linear(self.hidden_size, 2 * self.latent_size)
        nn.init.xavier_normal_(self.linear1.weight)

        self.tanh = nn.Tanh()
        self.loss_fn = self.vae_loss

    @staticmethod
    def multi_label_onehot(labels):
        pass

    def vae_loss(self, decoder_output, y_true_s, mu_q, logvar_q, anneal):
        # WARNING: This is only valid in the next-item recommendation scenario
        # onehot_y = torch.zeros_like(decoder_output).scatter_(-1, y_true_s.unsqueeze(1), value=1.0)

        # Calculate KL Divergence loss
        kld = (0.5 * (-logvar_q + logvar_q.exp() + mu_q ** 2 - 1)).sum(-1).mean()

        # Calculate Likelihood
        # dec_shape = decoder_output.shape  # [batch_size x seq_len x total_items] = [1 x seq_len x total_items]

        decoder_output = F.log_softmax(decoder_output, -1)

        # WARNING: In this framework, the sequential dataset contains a single target item
        # num_ones = 1.0  # float(y_true_s[0, 0].sum())

        # WARNING: Equal to F.nll_loss(decoder_output, y_true_s) in next-item prediction scenario
        likelihood = F.nll_loss(decoder_output, y_true_s)
        # likelihood = (-1.0 * onehot_y.view(dec_shape[0] * dec_shape[1], -1) *
        #               decoder_output.view(dec_shape[0] * dec_shape[1], -1)
        #               ).sum() / (float(dec_shape[0]) * num_ones)

        final = (anneal * kld) + likelihood

        return final

    def sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        temp_out = self.linear1(h_enc)

        mu = temp_out[:, :self.latent_size]
        log_sigma = temp_out[:, self.latent_size:]

        sigma = log_sigma.exp()
        std_z = torch.zeros_like(sigma).normal_(mean=0, std=1)

        self.z_mean = mu
        self.z_log_sigma = log_sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, x, x_lengths):
        in_shape = x.shape  # [bsz x seq_len] = [1 x seq_len]
        # x = x.view(-1)  # [seq_len]
        x = self.item_embed(x)  # [seq_len x embed_size]
        # x = x.view(in_shape[0], in_shape[1], -1)  # [1 x seq_len x embed_size]

        x, _ = self.gru(x)  # [1 x seq_len x rnn_size]
        x = x.reshape(in_shape[0] * in_shape[1], -1)  # [seq_len x rnn_size]

        x = self.encoder(x)  # [seq_len x hidden_size]
        x = self.sample_latent(x)  # [seq_len x latent_size]
        x = x.view(in_shape[0], in_shape[1], -1)

        # Adaptation to match RecBole+ data format (a sequence is decomposed into sub-sequences for next-item prediction)
        x = self.gather_indexes(x, x_lengths - 1)

        x = self.decoder(x)  # [seq_len x total_items]
        # x = x.view(in_shape[0], in_shape[1], -1)  # [1 x seq_len x total_items]

        return x, self.z_mean, self.z_log_sigma

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]

        self.update += 1
        if self.total_anneal_steps > 0:
            anneal = min(self.anneal_cap, 1. * self.update / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap

        seq_output, z_mean, z_log_sigma = self.forward(item_seq, item_seq_len)

        return self.loss_fn(seq_output, pos_items, z_mean, z_log_sigma, anneal)

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]

        seq_output, _, _ = self.forward(item_seq, item_seq_len)

        return seq_output.gather(1, test_item)

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output, _, _ = self.forward(item_seq, item_seq_len)
        return seq_output
