# Adaptive stochastic depth

Built on top of the PyTorch MNIST example at https://github.com/pytorch/examples/tree/master/mnist

Only feedforward, resnet and stochastic depth implemented, with bugs remaining.

Adaptive stochastic depth (validation-set train step) still to be implemented.

Example usage:
python trainer.py --dataset=cifar10 --num_blocks=12 --epochs=100 --model=resnet --momentum=0.9
