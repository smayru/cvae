import six

import chainer
import chainer.functions as F
from chainer.functions.loss.vae import gaussian_kl_divergence
import chainer.links as L


class CVAE(chainer.Chain):
    def __init__(self,m,n):
        super(CVAE, self).__init__(
            #Encoder
            f_e1=L.Linear(392,m),
            f_e2=L.Linear(392,m),
            f_e3=L.Linear(m*2,n),
            f_e4=L.Linear(m*2,n),
            f_d1=L.Linear(n,m),
            f_d2=L.Linear(392,m),
            f_d3=L.Linear(m*2,392),
        )
    def __call__(self, x):
        return self.reconstruct(self,x)
    def encode(self,x,y):
        h1=F.relu(self.f_e1(x))
        h2=F.relu(self.f_e2(y))
        h3=F.concat([h1,h2])
        mu=self.f_e3(h3)
        ln_var=self.f_e4(h3)
        return mu, ln_var
    def decode(self,z,x):
        h4=F.relu(self.f_d1(z))
        h5=F.relu(self.f_d2(x))
        h6=F.concat([h4,h5])
        h7=self.f_d3(h6)

        return h7#F.sigmoid(h7)
    def reconstruct(self,x,r_mu,r_ln_var,n):
        r_z = F.gaussian(r_mu, r_ln_var)
        x=F.reshape(x,[1,392])
        r_z=F.reshape(r_z,[1,n])
        r4=F.relu(self.f_d1(r_z))
        r5=F.relu(self.f_d2(x))
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