'''
Trains the network.

Mostly taken from PyTorch MNIST example at https://github.com/pytorch/examples/tree/master/mnist

Example usage:
python trainer.py --dataset=cifar10 --num_blocks=12 --epochs=100 --model=resnet --momentum=0.9
'''

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import models


# Training settings
parser = argparse.ArgumentParser(description='Adaptive Stochastic Depth trainer')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num_blocks', type=int, default=64, metavar='B',
                    help='number of residual blocks')
parser.add_argument('--dataset', type=str, default='mnist', metavar='DS',
                    help='which dataset to run on (default mnist)')
parser.add_argument('--model', type=str, default='net', metavar='M',
                    help='which model to use (default "net", a basic convnet)')
parser.add_argument('--overfit', action='store_true', default=False,
                    help='fits on test set to prove optimization succeeds')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testremote_projects/torch_adaptive_stochastic_depthing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 2)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=0.0001, metavar='WD',
                    help='weight decay (default: 0.0001')
parser.add_argument('--lr_decay_factor', type=float, default=0.95, metavar='LRDF',
                    help='learning rate decay factor (default: 0.95)')
parser.add_argument('--num_epochs_to_decay', type=int, default=1, metavar='NETD',
                    help='number of epochs before applying lr decay (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()



# Define dataset here

if args.dataset == 'cifar10':
    ds = datasets.CIFAR10
elif args.dataset == 'cifar100':
    ds = datasets.CIFAR100
elif args.dataset == 'lsun':
    ds = datasets.LSUN
elif args.dataset == 'mnist':
    ds = datasets.MNIST
else:
    raise Exception('Incorrect dataset name.')


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.dataset=='mnist':
    transform=transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,)),
                             ])
else:
    transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])
train_loader = torch.utils.data.DataLoader(
    ds('../' + args.dataset + '_data', train=True, download=True,
                   transform=transform
                   ), shuffle=True, batch_size=args.batch_size, **kwargs)
test_loader = torch.utils.data.DataLoader(
    ds('../' + args.dataset + '_data', train=False, transform=transform
                   ), shuffle=False, batch_size=args.batch_size, **kwargs)

input_dim = 1 if args.dataset=='mnist' else 3


if args.model == 'net':
    model = models.Net(input_dim=input_dim, num_blocks=args.num_blocks)
elif args.model == 'resnet':
    model = models.ResNet(input_dim=input_dim, num_blocks=args.num_blocks)
elif args.model == 'stochasticresnet':
    model = models.StochasticResNetResNet(input_dim=input_dim, num_blocks=args.num_blocks)
else:
    raise Exception('Incorrect model name.')

if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader) # loss function already averages over batch size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, args.epochs + 1):
    lr = args.lr
    train(epoch)
    test(epoch)
    if (epoch > 0) and (epoch % args.num_epochs_to_decay == 0):
        lr = lr * args.lr_decay_factor
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)

