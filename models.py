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
k = 4

base_seq = [16, 32, 64]

size_seq = [k * s for s in base_seq]


class Block(nn.Module):
    ''' The block type defined in wide resnet paper
    '''
    def __init__(self, in_size, out_size, residual=False, stochastic=False, keep_prob=0.8):
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
        self.keep_prob = keep_prob

    def forward(self, x):

        # drop the whole layer
        if self.stochastic and self.training and (np.random.random() < self.keep_prob):
            return x

        # conv1 result
        inner_result = F.relu(self.bn1(x))
        inner_result = self.conv1(inner_result)


        # dropout on conv1 result
        inner_result = self.drop(inner_result)

        # conv2 result
        inner_result = self.conv2(F.relu(self.bn2(inner_result)))

        # replicate x a number of times if the output # filters is a multiple of input # filters
        # don't know if this is actually the correct way to do it
        assert self.out_size >= self.in_size, "No defined behavior for in_size > out_size"

        if self.out_size > self.in_size:
            assert self.out_size % self.in_size == 0, "Out size must be a multiple of in size to replicate x"
            x = t.cat(
                tuple([x for _ in range(self.out_size / self.in_size)]),
                1)

        # if stochastic and not training, treat it as a weighted residual connection
        if self.stochastic and not self.training:
            return x + self.keep_prob * inner_result

        # vanilla residual connection
        elif self.stochastic or self.residual:
            return x + inner_result

        # vanilla network (not resnet or stochastic)
        else:
            return x

class Net(nn.Module):
    ''' A generic class for networks composed of Blocks
    '''
    def __init__(self, num_blocks=2, input_dim=1, size_sequence=size_seq, residual=False, stochastic=False):
        super(Net, self).__init__()
        self.num_blocks = num_blocks - 1

        current_size = 16

        self.conv1 = nn.Conv2d(input_dim, current_size, kernel_size=3, padding=1)

        self.blocks = []

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
                # get size
                current_size = size_sequence[j]

                # HACK: apparently pytorch requires that each sub-module be an attribute of the top-level module
                # for backpropogation to work properly
                # so after creating the block, as WELL as adding it to the "blocks" list attribute, we also
                # add that block to a uniquely named attribute of the Net module
                attr_name = "block_" + str(idx)
                setattr(self, attr_name, new_block)
                self.blocks.append(new_block)
                idx += 1

        self.fc = nn.Linear(size_sequence[-1], 10)

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
                x = F.relu(F.max_pool2d(x, 2))

        # Final pooling, fc layer and softmax output
        x = F.relu(x)
        x = t.squeeze(F.avg_pool2d(x, 8))
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
