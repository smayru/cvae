#!/usr/bin/env python

from __future__ import print_function
import argparse
import time
import numpy as np
import six
import net
import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import serializers

import data

parser = argparse.ArgumentParser(description='CVAE')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--epoch', '-e', default=100, type=int,
                    help='number of epochs to learn')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='learning minibatch size')
parser.add_argument('--unit_1', '-u1', type=int, default=500,
                    help='size of unit1')
parser.add_argument('--unit_2', '-u2', type=int, default=200,
                    help='size of unit2')

args = parser.parse_args()

batchsize = args.batchsize
n_epoch = args.epoch
print('GPU: {}'.format(args.gpu))
print('# Minibatch-size: {}'.format(args.batchsize))
print('# epoch: {}'.format(args.epoch))
print('')

# Prepare dataset
print('load MNIST dataset')
mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255


N = 60000
#N_test = y_test.size
x_train, x_test = np.split(mnist['data'],   [N])

x_train_train=x_train[:,0:28*28/2]
x_train_test=x_train[:,28*28/2::]

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

model = net.CVAE(args.unit_1,args.unit_2)


# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

# Learning loop
error=np.zeros([n_epoch])
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N)
    sum_loss = 0
    start = time.time()
    for i in six.moves.range(0, N, batchsize):
        x = chainer.Variable(xp.asarray(x_train_train[perm[i:i + batchsize]]))
        y = chainer.Variable(xp.asarray(x_train_test[perm[i:i + batchsize]]))

        optimizer.zero_grads()
        loss = model.get_loss(x,y)
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(x.data)
    print('loss',sum_loss)
    error[epoch-1]=sum_loss
    del loss

np.savetxt('train_error.csv',error,delimiter=',')
# Save the model and the optimizer
print('save the model')
serializers.save_npz('cvae.model', model)
print('save the optimizer')
serializers.save_npz('cvae.state', optimizer)

