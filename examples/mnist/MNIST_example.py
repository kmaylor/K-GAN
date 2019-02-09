#from wgan_gp import WGAN_GP as GAN
#from wgan import WGAN as GAN
#from dcgan import DCGAN as GAN
from ct_gan import CTGAN as GAN
import numpy as np
from keras.datasets import mnist

class MNISTGAN(object):
    def __init__(self):
            
        kernels = [4,4,4,4]
        strides = [2,2,2,1]

        self.img_rows = 28
        self.img_cols = 28
        self.channel = 1
        (self.x_train, _), (_, _) = mnist.load_data()
        self.x_train = self.x_train.reshape(-1, self.img_rows,\
        	self.img_cols, 1)/255*2-1

        self.GAN = GAN(img_rows=self.img_rows, img_cols=self.img_cols, 
                            load_dir=None, depth = 128, save_dir = 'Saved_Models/MNIST_example')
        self.GAN.strides = strides
        self.GAN.kernels = kernels

    def train(self):
        self.GAN.train(self.x_train, 'MNIST_sims_',train_steps=5000, save_interval=500,
                       verbose = 100, train_rate=(5,1))

t = MNISTGAN()
t.train()