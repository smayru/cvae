#!/usr/bin/env python
from __future__ import print_function
import argparse
import time
import net
import numpy as np
import six
import copy
import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import serializers
import matplotlib.pyplot as plt
import data

parser = argparse.ArgumentParser(description='CVAE')
parser.add_argument('--initmodel', '-m', default='cvae.model',
                    help='Initialize the model from given file')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--resume', '-r', default='cvae.state',
                    help='Resume the optimization from snapshot')
parser.add_argument('--batchsize', '-b', type=int, default=100,
                    help='learning minibatch size')
parser.add_argument('--character', '-ch', type=int, default=5,
                    help='chracter')
parser.add_argument('--number', '-numb', type=int, default=8,
                    help='number')
parser.add_argument('--unit_1', '-u1', type=int, default=500,
                    help='size of unit1')
parser.add_argument('--unit_2', '-u2', type=int, default=200,
                    help='size of unit2')
parser.add_argument('--setting', '-s', default='predict',
                    help='setting of plot')
parser.add_argument('--data_type', '-data_type', default='test',
                    help='trainig or test')


args = parser.parse_args()

batchsize = args.batchsize
number_1=args.character
number_2=args.number
s=args.setting

# Prepare dataset
print('load MNIST dataset')
mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255
mnist['target'] = mnist['target'].astype(np.int32)

N = 60000
x_train, x_test = np.split(mnist['data'],   [N])
y_train, y_test = np.split(mnist['target'], [N])


x_train_train=x_train[:,0:28*28/2]
x_train_test=x_train[:,28*28/2::]

x_test_train=x_test[:,0:28*28/2]
x_test_test=x_test[:,28*28/2::]


if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy
"""
class CVAE(chainer.Chain):

    def __init__(self):
        super(CVAE, self).__init__(
            #Encoder
            f_e1=L.Linear(392,args.unit_1),
            f_e2=L.Linear(392,args.unit_1),
            f_e3=L.Linear(args.unit_1*2,args.unit_2),
            f_e4=L.Linear(args.unit_1*2,args.unit_2),
            f_d1=L.Linear(args.unit_2,args.unit_1),
            f_d2=L.Linear(392,args.unit_1),
            f_d3=L.Linear(args.unit_1*2,392),
        )
    def __call__(self, x):
        return self.reconstruct(self,x)
    def encode(self,x,y):
        h1=F.tanh(self.f_e1(x))
        h2=F.tanh(self.f_e2(y))
        h3=F.concat([h1,h2])
        mu=self.f_e3(h3)
        ln_var=self.f_e4(h3)
        return mu, ln_var
    def decode(self,z,x):
        h4=F.tanh(self.f_d1(z))
        h5=F.tanh(self.f_d2(x))
        h6=F.concat([h4,h5])
        h7=self.f_d3(h6)

        return h7#F.sigmoid(h7)
    def reconstruct(self,x):
        r_mu=chainer.Variable(xp.zeros([args.unit_2]).astype(np.float32))
        r_ln_var=chainer.Variable(xp.zeros([args.unit_2]).astype(np.float32))
        r_z = F.gaussian(r_mu, r_ln_var)
        x=F.reshape(x,[1,392])
        r_z=F.reshape(r_z,[1,args.unit_2])
        r4=F.tanh(self.f_d1(r_z))
        r5=F.tanh(self.f_d2(x))
        r6=F.concat([r4,r5])
        r7=self.f_d3(r6)
        r8=r7#F.sigmoid(r7)
        return r8.data

    def get_loss(self,x,y,train=True):
        mu, ln_var = self.encode(x,y)
        batchsize = len(mu.data)
            # reconstruction loss
        rec_loss = 0
        z = F.gaussian(mu, ln_var)
        rec_loss += F.bernoulli_nll(y, self.decode(z,x))/ (batchsize)
        self.rec_loss = rec_loss
        self.loss = self.rec_loss + F.gaussian_kl_divergence(mu, ln_var) / batchsize
        return self.loss
"""
model = net.CVAE(args.unit_1,args.unit_2)



# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)

train_num=np.zeros([10])
test_num=np.zeros([10])
train_idx=[[],[],[],[],[],[],[],[],[],[]]
test_idx=[[],[],[],[],[],[],[],[],[],[]]

for i in range(y_train.shape[0]):
	train_idx[y_train[i]].append(i)
	train_num[y_train[i]]+=1
for i in range(y_test.shape[0]):
	test_num[y_test[i]]+=1
	test_idx[y_test[i]].append(i)

data_type=args.data_type


if data_type=='train':
	x=x_train_train[train_idx[number_1][number_2]]
	y=x_train_test[train_idx[number_1][number_2]]
elif data_type=='test':
	x=x_test_train[test_idx[number_1][number_2]]
	y=x_test_test[test_idx[number_1][number_2]]
tmp_x=copy.deepcopy(x)
x=chainer.Variable(x.astype(np.float32))
r_mu=chainer.Variable(xp.zeros([args.unit_2]).astype(np.float32))
r_ln_var=chainer.Variable(xp.zeros([args.unit_2]).astype(np.float32))
reconst_x=model.reconstruct(x,r_mu,r_ln_var,args.unit_2).reshape([392])
predict=np.r_[tmp_x,reconst_x].reshape(28,28)

problem=np.r_[tmp_x,np.zeros(392)].reshape(28,28)
answer=np.r_[tmp_x,y].reshape(28,28)
predict[np.where(predict>1)]=1
predict[np.where(predict<0)]=0

if s=='problem':
	z=problem
elif s=='answer':
	z=answer
elif s=='predict':
	z=predict
z = z[::-1,:]
plt.clf()             # flip vertical
plt.xlim(0,27)
plt.ylim(0,27)
plt.pcolor(z)
plt.gray()
plt.tick_params(labelbottom="off")
plt.tick_params(labelleft="off")
name=s+'_'+'type_'+data_type+'_char_'+str(number_1)+'_num_'+str(number_2)+'.png'
plt.savefig(name)


