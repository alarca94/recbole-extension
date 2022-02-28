import six
import numpy as np
import random
import math
import torch
import collections

from torch import nn
from torch.nn import functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


class LogUniformSampler(object):
    def __init__(self, ntokens):

        self.N = ntokens
        self.prob = [0] * self.N

        self.generate_distribution()

    def generate_distribution(self):
        for i in range(self.N):
            self.prob[i] = (np.log(i+2) - np.log(i+1)) / np.log(self.N + 1)

    def probability(self, idx):
        return self.prob[idx]

    def expected_count(self, num_tries, samples):
        freq = list()
        for sample_idx in samples:
            freq.append(-(np.exp(num_tries * np.log(1-self.prob[sample_idx]))-1))
        return freq

    def accidental_match(self, labels, samples):
        sample_dict = dict()

        for idx in range(len(samples)):
            sample_dict[samples[idx]] = idx

        result = list()
        for idx in range(len(labels)):
            if labels[idx] in sample_dict:
                result.append((idx, sample_dict[labels[idx]]))

        return result

    def sample(self, size, labels):
        log_N = np.log(self.N)

        x = np.random.uniform(low=0.0, high=1.0, size=size)
        value = np.floor(np.exp(x * log_N)).astype(int) - 1
        samples = value.tolist()

        true_freq = self.expected_count(size, labels.tolist())
        sample_freq = self.expected_count(size, samples)

        return samples, true_freq, sample_freq

    def sample_unique(self, size, labels):
        # Slow. Not Recommended.
        log_N = np.log(self.N)
        samples = list()

        while (len(samples) < size):
            x = np.random.uniform(low=0.0, high=1.0, size=1)[0]
            value = np.floor(np.exp(x * log_N)).astype(int) - 1
            if value in samples:
                continue
            else:
                samples.append(value)

        true_freq = self.expected_count(size, labels.tolist())
        sample_freq = self.expected_count(size, samples)

        return samples, true_freq, sample_freq


class SampledSoftmax(nn.Module):
    def __init__(self, ntokens, nsampled, nhid, tied_weight):
        super(SampledSoftmax, self).__init__()

        # Parameters
        self.ntokens = ntokens
        self.nsampled = nsampled

        self.sampler = LogUniformSampler(self.ntokens)
        self.params = nn.Linear(nhid, ntokens)

        if tied_weight is not None:
            self.params.weight = tied_weight
        else:
            in_, out_ = self.params.weight.size()
            stdv = math.sqrt(3. / (in_ + out_))
            self.params.weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs, labels):
        if self.training:
            # sample ids according to word distribution - Unique
            sample_values = self.sampler.sample(self.nsampled, labels.data.cpu().numpy())
            return self.sampled(inputs, labels, sample_values, remove_accidental_match=True)
        else:
            return self.full(inputs, labels)

    def sampled(self, inputs, labels, sample_values, remove_accidental_match=False):

        batch_size, d = inputs.size()
        sample_ids, true_freq, sample_freq = sample_values

        sample_ids = torch.autograd.Variable(torch.LongTensor(sample_ids))
        true_freq = torch.autograd.Variable(torch.FloatTensor(true_freq))
        sample_freq = torch.autograd.Variable(torch.FloatTensor(sample_freq))

        # gather true labels - weights and frequencies
        true_weights = self.params.weight[labels, :]
        true_bias = self.params.bias[labels]

        # gather sample ids - weights and frequencies
        sample_weights = self.params.weight[sample_ids, :]
        sample_bias = self.params.bias[sample_ids]

        # calculate logits
        true_logits = torch.sum(torch.mul(inputs, true_weights), dim=1) + true_bias
        sample_logits = torch.matmul(inputs, torch.t(sample_weights)) + sample_bias
        # remove true labels from sample set
        if remove_accidental_match:
            acc_hits = self.sampler.accidental_match(labels.data.cpu().numpy(), sample_ids.data.cpu().numpy())
            acc_hits = list(zip(*acc_hits))
            sample_logits[acc_hits] = -1e37

        # perform correction
        true_logits = true_logits.sub(torch.log(true_freq))
        sample_logits = sample_logits.sub(torch.log(sample_freq))

        # return logits and new_labels
        logits = torch.cat((torch.unsqueeze(true_logits, dim=1), sample_logits), dim=1)
        new_targets = torch.autograd.Variable(torch.zeros(batch_size).long())
        return logits, new_targets

    def full(self, inputs):
        return self.params(inputs)


class ResidualBlock(nn.Module):
    r"""
    Residual block (b) in the paper
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None, causal=False):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation * 2)
        self.ln2 = nn.LayerNorm(out_channel, eps=1e-8)

        self.dilation = dilation
        self.kernel_size = kernel_size
        self.causal = causal

    def forward(self, x):  # x: [batch_size, seq_len, embed_size]
        if self.causal:
            x = self.conv_pad(x, self.dilation)  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        out = self.conv1(x).squeeze(2).permute(0, 2, 1)
        # [batch_size, seq_len+(self.kernel_size-1)*dilations-kernel_size+1, embed_size]
        out = F.relu(self.ln1(out))
        if self.causal:
            out = self.conv_pad(out, self.dilation * 2)
        out2 = self.conv2(out).squeeze(2).permute(0, 2, 1)
        out2 = F.relu(self.ln2(out2))
        return out2 + x

    def conv_pad(self, x, dilation):
        r""" Dropout-mask: To avoid the future information leakage problem, this paper proposed a masking-based dropout
        trick for the 1D dilated convolution to prevent the network from seeing the future items.
        Also the One-dimensional transformation is completed in this function.
        """
        inputs_pad = x.permute(0, 2, 1)
        inputs_pad = inputs_pad.unsqueeze(2)
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dilation, 0, 0, 0))
        inputs_pad = pad(inputs_pad)
        return inputs_pad


class GRec(SequentialRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(GRec, self).__init__(config, dataset)

        self.is_negsample = config['is_negsample']
        self.embedding_width = config['embedding_size']
        self.residual_channels = config['embedding_size']
        self.dilations = config['dilations']
        self.kernel_size = config['kernel_size']
        self.masked_lm_prob = config['masked_lm_prob']
        self.max_predictions_per_seq = config['max_predictions_per_seq']
        self.item_list = list(range(self.n_items))
        self.mask_token = self.n_items

        self.embedding_en = nn.Embedding(self.n_items, self.embedding_width, padding_idx=0)
        self.embedding_de = nn.Embedding(self.n_items, self.embedding_width, padding_idx=0)
        nn.init.trunc_normal_(self.embedding_en.weight, std=0.02)
        nn.init.trunc_normal_(self.embedding_de.weight, std=0.02)

        rb = [
            ResidualBlock(
                self.residual_channels, self.residual_channels,
                kernel_size=self.kernel_size, dilation=dilation, causal=False
            ) for dilation in self.dilations
        ]
        self.residual_blocks_en = nn.Sequential(*rb)

        self.adapter_conv = nn.Conv2d(self.residual_channels, self.residual_channels, kernel_size=(1, 1),
                                      padding='same', dilation=1)

        rb = [
            ResidualBlock(
                self.residual_channels, self.residual_channels,
                kernel_size=self.kernel_size, dilation=dilation, causal=True
            ) for dilation in self.dilations
        ]
        self.residual_blocks_de = nn.Sequential(*rb)

        # TODO: Not sure about this linear layer (theoretically this was a softmax layer). Also, isn't embedding weight
        # shared by input and output layers to reduce overfitting and memory?
        self.linear = nn.Linear(self.embedding_width, self.n_items)
        nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        nn.init.constant_(self.linear.bias, val=0.1)

        self.sampled_softmax_loss = SampledSoftmax()

    def forward(self, item_seq_en, item_seq_de):
        context_embedding_en = self.embedding_en(item_seq_en)
        context_embedding_de = self.embedding_de(item_seq_de)

        dilate_input_en = self.residual_blocks_en(context_embedding_en)
        dilate_input_de = dilate_input_en + context_embedding_de

        # get_adapter()
        dilate_input_de.unsqueeze_(1)
        dilate_input_de = self.adapter_conv(dilate_input_de)
        dilate_input_de.squeeze_(1)

        dilate_input_de = self.residual_blocks_de(dilate_input_de)
        return dilate_input_de

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        pos_items = interaction[self.POS_ITEM_ID]

        item_seq_en, masked_idxs, masked_labels, masked_lm_weights = self.create_masked_lm_predictions_from_batch(item_seq)
        logits = self.forward(item_seq_en, item_seq)



        return loss

    @staticmethod
    def assert_rank(tensor, expected_rank, name=None):
        """Raises an exception if the tensor rank is not of the expected rank.

        Args:
          tensor: A tf.Tensor to check the rank of.
          expected_rank: Python integer or list of integers, expected rank.
          name: Optional name of the tensor for the error message.

        Raises:
          ValueError: If the expected shape doesn't match the actual shape.
        """
        if name is None:
            name = tensor.name

        expected_rank_dict = {}
        if isinstance(expected_rank, six.integer_types):
            expected_rank_dict[expected_rank] = True
        else:
            for x in expected_rank:
                expected_rank_dict[x] = True

        actual_rank = tensor.shape.ndims
        if actual_rank not in expected_rank_dict:
            raise ValueError(
                "For the tensor `%s`, the actual rank "
                "`%d` (shape = %s) is not equal to the expected rank `%s`" %
                (name, actual_rank, str(tensor.shape), str(expected_rank)))

    def get_shape_list(self, tensor, expected_rank=None, name=None):
        """Returns a list of the shape of tensor, preferring static dimensions.

        Args:
          tensor: A tf.Tensor object to find the shape of.
          expected_rank: (optional) int. The expected rank of `tensor`. If this is
            specified and the `tensor` has a different rank, and exception will be
            thrown.
          name: Optional name of the tensor for the error message.

        Returns:
          A list of dimensions of the shape of tensor. All static dimensions will
          be returned as python integers, and dynamic dimensions will be returned
          as tf.Tensor scalars.
        """
        if name is None:
            name = tensor.name

        if expected_rank is not None:
            self.assert_rank(tensor, expected_rank, name)

        shape = tensor.shape.as_list()

        non_static_indexes = []
        for (index, dim) in enumerate(shape):
            if dim is None:
                non_static_indexes.append(index)

        if not non_static_indexes:
            return shape

        dyn_shape = tensor.shape
        for index in non_static_indexes:
            shape[index] = dyn_shape[index]
        return shape

    def gather_indexes(self, sequence_tensor, positions):
        """Gathers the vectors at the specific positions over a minibatch."""
        sequence_shape = self.get_shape_list(sequence_tensor, expected_rank=3)
        batch_size = sequence_shape[0]
        seq_length = sequence_shape[1]
        width = sequence_shape[2]

        flat_offsets = torch.reshape(
            torch.range(0, batch_size, dtype=torch.int32) * seq_length, [-1, 1])
        flat_positions = torch.reshape(positions + flat_offsets, [-1])
        flat_sequence_tensor = torch.reshape(sequence_tensor,
                                             [batch_size * seq_length, width])
        output_tensor = torch.gather(flat_sequence_tensor, flat_positions)
        return output_tensor

    def get_masked_lm_output(self, input_tensor, positions, label_ids, label_weights, trainable=True):
        """Get loss and log probs for the masked LM."""

        input_tensor = self.gather_indexes(input_tensor, positions)

        if self.is_negsample:
            logits_2D = input_tensor
            label_flat = torch.reshape(label_ids, [-1, 1])  # 1 is the number of positive example
            num_sampled = int(0.2 * self.n_items)  # sample 20% as negatives
            loss = self.sampled_softmax_loss(self.softmax_w, self.softmax_b, label_flat, logits_2D,
                                              num_sampled,
                                              self.n_items)
        else:
            sequence_shape = self.get_shape_list(positions)
            batch_size = sequence_shape[0]
            seq_length = sequence_shape[1]
            residual_channels = input_tensor.get_shape().as_list()[-1]
            input_tensor = tf.reshape(input_tensor, [-1, seq_length, residual_channels])

            logits = self.conv1d(tf.nn.relu(input_tensor), self.n_items, name='logits')
            logits_2D = tf.reshape(logits, [-1, self.n_items])
            label_flat = tf.reshape(label_ids, [-1])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_flat, logits=logits_2D)
        loss = tf.reduce_mean(loss)
        regularization = 0.001 * tf.reduce_mean([tf.nn.l2_loss(v) for v in tf.compat.v1.trainable_variables()])
        loss = loss + regularization

        return loss

    @staticmethod
    def create_masked_lm_predictions(tokens, masked_lm_prob,
                                     max_predictions_per_seq, vocab_words, rng, item_size):
        """Creates the predictions for the masked LM objective."""

        cand_indexes = []
        for (i, token) in enumerate(tokens):
            if token == "[CLS]" or token == "[SEP]":
                continue
            cand_indexes.append(i)

        rng.shuffle(cand_indexes)

        output_tokens = list(tokens)

        num_to_predict = min(max_predictions_per_seq,
                             max(1, int(round(len(tokens) * masked_lm_prob))))

        masked_lms = []
        covered_indexes = set()
        for index in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            if index in covered_indexes:
                continue
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 1.0:
                # masked_token = "[MASK]"
                masked_token = 0  # item_size is "[MASK]"   0 represents '<unk>'
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

        masked_lms = sorted(masked_lms, key=lambda x: x.index)

        masked_lm_positions = []
        masked_lm_labels = []
        for p in masked_lms:
            masked_lm_positions.append(p.index)
            masked_lm_labels.append(p.label)

        return output_tokens, masked_lm_positions, masked_lm_labels

    def create_masked_lm_predictions(self, item_batch, masked_lm_prob,
                                               max_predictions_per_seq, items, item_size):
        output_tokens_batch = []
        maskedpositions_batch = []
        maskedlabels_batch = []
        masked_lm_weights_batch = []

        item_batch_ = item_batch[:, 1:]  # remove start and end
        for line_list in range(item_batch_.shape[0]):
            output_tokens, masked_lm_positions, masked_lm_labels = self.create_masked_lm_predictions(
                item_batch_[line_list],
                masked_lm_prob,
                max_predictions_per_seq,
                items, item_size)
            # print(output_tokens)
            output_tokens.insert(0, item_batch[line_list][0])
            output_tokens_batch.append(output_tokens)
            maskedpositions_batch.append(masked_lm_positions)
            maskedlabels_batch.append(masked_lm_labels)
            masked_lm_weights = [1.0] * len(masked_lm_labels)
            # note you can not change here since it should be consistent with 'num_to_predict' in create_masked_lm_predictions
            num_to_predict = min(max_predictions_per_seq,
                                 max(1, int(round(len(item_batch_[line_list]) * masked_lm_prob))))

            while len(masked_lm_weights) < num_to_predict:
                masked_lm_weights.append(0.0)
            masked_lm_weights_batch.append(masked_lm_weights)

        return output_tokens_batch, maskedpositions_batch, maskedlabels_batch, masked_lm_weights_batch
