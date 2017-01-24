from __future__ import print_function
import argparse
import torch as t
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

k = 4

base_seq = [16, 32, 64]

size_seq = [k * s for s in base_seq]


class Block(nn.Module):
    def __init__(self, in_size, out_size, residual=False, stochastic=False, keep_prob=0.8):
        super(Block, self).__init__()
        self.in_size = in_size
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
        if self.stochastic and self.training and (np.random.random() < self.keep_prob):
            return x

        #conv1 result
        inner_result = F.relu(self.bn1(x))
        inner_result = self.conv1(inner_result)


        #dropped
        inner_result = self.drop(inner_result)

        #conv2 result
        inner_result = self.conv2(F.relu(self.bn2(inner_result)))

        if self.out_size > self.in_size:
            x = t.cat((x, t.zeros(x.size()[0], self.out_size - self.in_size, x.size()[2], x.size()[3])), 1)

        if self.stochastic and not self.training:
            return x + self.keep_prob * inner_result

        elif self.residual:
            return x + inner_result

        else:
            return x

class Net(nn.Module):
    def __init__(self, num_blocks=2, input_dim=1, size_sequence=size_seq, residual=False, stochastic=False):
        super(Net, self).__init__()
        self.num_blocks = num_blocks - 1

        current_size = 16

        self.conv1 = nn.Conv2d(input_dim, current_size, kernel_size=3, padding=1)

        self.blocks = []

        divisible = int(np.floor(self.num_blocks / 3.0))
        remainder = self.num_blocks - 3 * divisible
        idx = 0

        for j in range(3):
            for i in range(divisible + (1 if j < remainder else 0)):
                new_block = Block(in_size=current_size, out_size=size_sequence[j], residual=residual, stochastic=stochastic)
                current_size = size_sequence[j]
                attr_name = "block_" + str(idx)
                setattr(self, attr_name, new_block)
                self.blocks.append(new_block)
                idx += 1
        self.fc = nn.Linear(size_sequence[-1], 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #print(x)
        divisible = int(np.floor(self.num_blocks / 3.0))
        remainder = self.num_blocks - 3 * divisible
        idx = 0
        for j in range(3):
            for i in range(divisible + (1 if j < remainder else 0)):
                x = self.blocks[idx].forward(x)
                idx += 1
            if j < 2:
                x = F.relu(F.max_pool2d(x, 2))
        x = F.relu(x)
        #x = t.squeeze(t.mean(t.mean(x, 2), 3))
        #x = F.dropout(x, training=self.training)
        x = t.squeeze(F.avg_pool2d(x, 8))
        x = F.relu(self.fc(x))
        return F.log_softmax(x)

class ResNet(Net):
    def __init__(self, num_blocks=5, input_dim=1):
        super(ResNet, self).__init__(num_blocks=num_blocks, residual=True, input_dim=input_dim)

class StochasticResNet(Net):
    def __init__(self, num_blocks=40, input_dim=1):
        super(StochasticResNet, self).__init__(num_blocks=num_blocks, residual=True,
                                               stochastic=True, input_dim=input_dim)
