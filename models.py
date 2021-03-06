'''
Implements feedforward, residual and stochastic networks composed of a number of Blocks.

Each Block is of the type from the "wide residual networks" paper: https://github.com/szagoruyko/wide-residual-networks

We assume that the network architecture is:

-input layer

-initial convolution

-3 "groups" of blocks:
    each with own width/# filters per block
    the first two groups followed by max-pool downsampling (32x32 to 16x16, 16x16 to 8x8)

-avg pooling over the 8x8 feature maps

-final fully connected layer

-softmax output

'''

from __future__ import print_function
import argparse
import torch as t
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms


# The sequence of sizes of blocks within each group from Wide Resnet paper
# I'll clean this up / incorporate k and base_seq as parameters
k = 1

base_seq = [16, 32, 64]

size_seq = [k * s for s in base_seq]


class Block(nn.Module):
    ''' The block type with sequence of (conv, relu, batchnorm, dropout) operations as defined in wide resnet paper

    Can be:
        feedforward, x_t = f(x_{t-1})
        residual, x_t = f(x_{t-1}) + x_{t-1}
        stochastic-depth, x_t = f(x_{t-1}) + x_{t-1} with probability alpha
                              = x_{t-1} with probability 1-alpha
    '''
    def __init__(self, in_size, out_size, residual=False, stochastic=False, keep_prob=None):
        super(Block, self).__init__()
        # number of input filters
        self.in_size = in_size
        # number of output filters
        self.out_size = out_size
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.drop = nn.Dropout2d()
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_size)
        self.bn2 = nn.BatchNorm2d(out_size)
        self.residual = residual
        self.stochastic = stochastic

        # hacky way of init-ing keep prob to something random between .5 and 1.0 to break ties
        if keep_prob is None:
            keep_prob = 0.5 + np.random.random() * 0.5

        # Create a keep_prob hyperparameter for adapting the stochastic prob
        # This is not an instance of torch.Parameter, so shouldn't be returned by net.Parameters()
        # and should not be optimized in the train step

        # We first find the logit such that sigmoid(logit) = keep_prob so we can constrain to [0, 1]
        logit = np.log(keep_prob/(1 - keep_prob))
        self.keep_prob_logit = Variable(t.FloatTensor([logit]), requires_grad=True)

    def cuda(self):
        ''' We override the nn.Module cuda method to also cuda-ize our trainable hyperparameters
        '''

        self.keep_prob_logit = self.keep_prob_logit.cuda()
        super(Block, self).cuda()

    def maybe_expand(self, x):
        # replicate x a number of times if the output # filters is a multiple of input # filters
        # don't know if this is actually the correct way to do it
        assert self.out_size >= self.in_size, "No defined behavior for in_size > out_size"

        if self.out_size > self.in_size:
            assert self.out_size % self.in_size == 0, "Out_size must be a multiple of in_size to replicate x"
            x = t.cat(
                tuple([x for _ in range(self.out_size / self.in_size)]),
                1)

        return x

    def transform(self, x):
        ''' Computes the transformation of the input defined by the layer's params
        (This is the f(x) part of the resnet eqn x_t+1 = f(x_t) + x_t)

        :param x: the input to the layer
        :return: inner_result: the transformation of the input (conv, batch norm, relu etc) defined by layer
        '''


        # conv1 result
        inner_result = F.relu(self.bn1(x))
        inner_result = self.conv1(inner_result)

        # dropout on conv1 result
        inner_result = self.drop(inner_result)

        # conv2 result
        inner_result = self.conv2(F.relu(self.bn2(inner_result)))

        return inner_result


    def forward(self, x):
        ''' Compute the output of layer.
        This fn handles residual / stochastic behavior. It calls transform() to do the actual ops (conv, relu etc).

        :param x: the input to the current layer
        :return: the input to the next layer
        '''

        # drop the whole layer
        if self.stochastic and \
                self.training and \
                (np.random.random() < t.sigmoid(self.keep_prob_logit.data.cpu()).numpy()[0]):
            # above: we HAVE to get 'data' from the keep_prob Variable, put on CPU,
            # convert to Numpy, and get the only element in array, before we can compare.
            # WARNING: 1.0 < Var will return TRUE even if Var.data = 0.5
            return self.maybe_expand(x)

        inner_result = self.transform(x)

        # if stochastic and not training, treat it as a weighted residual connection
        if self.stochastic and not self.training:
            # HACK: this is ugly bc PyTorch doesn't support broadcasting yet
            return self.maybe_expand(x) + t.mul(
                inner_result,
                # unsqueeze keep_prob to 4d and expand to size of inner_result before elementwise multiply
                t.sigmoid(self.keep_prob_logit).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(inner_result)
            )

        # vanilla residual connection
        elif self.stochastic or self.residual:
            return self.maybe_expand(x) + inner_result

        # vanilla network (not resnet or stochastic)
        else:
            return inner_result

class Net(nn.Module):
    ''' A generic class for networks composed of Blocks
    '''
    def __init__(self, num_blocks=2, input_dim=1, size_sequence=size_seq, residual=False, stochastic=False):
        super(Net, self).__init__()

        self.num_blocks = num_blocks

        # We start by mapping to 16 feature maps
        current_size = 16

        # Initial convolution layer
        self.conv1 = nn.Conv2d(input_dim, current_size, kernel_size=3, padding=1)

        self.blocks = []

        self.trainable_hyperparams = []

        # Determines # blocks in each group (divide by 3, remainder allocated among earlier layers)
        divisible = int(np.floor(self.num_blocks / 3.0))
        remainder = self.num_blocks - 3 * divisible
        idx = 0

        # Three groups of blocks:
        for j in range(3):

            # Iterate over # blocks in this group:
            for i in range(divisible + (1 if j < remainder else 0)):
                # create block
                new_block = Block(in_size=current_size, out_size=size_sequence[j], residual=residual, stochastic=stochastic)

                self.trainable_hyperparams.append(new_block.keep_prob_logit)
                # get size
                current_size = size_sequence[j]

                # HACK: apparently pytorch requires that each sub-module be an attribute of the top-level module
                # for backpropogation to work properly
                # so after creating the block, as WELL as appending it to the "blocks" list, we also
                # set that block to be a uniquely named attribute of the Net module
                attr_name = "block_" + str(idx)
                setattr(self, attr_name, new_block)
                self.blocks.append(new_block)
                idx += 1

        self.fc = nn.Linear(size_sequence[-1], 10)

    def cuda(self):
        ''' We override the nn.Module cuda method to also cuda-ize our trainable hyperparameters
        '''

        for block in self.blocks:
            block.cuda()

        super(Net, self).cuda()

    def forward(self, x):

        # Initial conv and relu
        x = F.relu(self.conv1(x))

        # Again figure out # blocks for each group
        divisible = int(np.floor(self.num_blocks / 3.0))
        remainder = self.num_blocks - 3 * divisible
        idx = 0

        # Iterate over 3 groups
        for j in range(3):

            # Iterate over # blocks in group
            for i in range(divisible + (1 if j < remainder else 0)):
                x = self.blocks[idx].forward(x)
                idx += 1

            # Downsample after the first two groups
            if j < 2:
                x = F.relu(F.avg_pool2d(x, 2))

        # Final pooling, fc layer and softmax output
        x = F.relu(x)
        x = t.squeeze(F.avg_pool2d(x, x.size()[3]))
        x = F.relu(self.fc(x))
        return F.log_softmax(x)

class ResNet(Net):
    '''Convenience class for ResNet: a Net with residual blocks
    '''
    def __init__(self, num_blocks=5, input_dim=1):
        super(ResNet, self).__init__(num_blocks=num_blocks, residual=True, input_dim=input_dim)

class StochasticResNet(Net):
    '''Convenience class for StochasticNet: a Net with stochastic-depth blocks
    '''
    def __init__(self, num_blocks=40, input_dim=1):
        super(StochasticResNet, self).__init__(num_blocks=num_blocks, residual=True,
                                               stochastic=True, input_dim=input_dim)
