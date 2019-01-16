from wgan_gp import WGAN_GP as GAN
#from wgan import WGAN as GAN
#from dcgan import DCGAN as GAN
import numpy as np
from keras.datasets import cifar10

class CIFARGAN(object):
    def __init__(self):
            
        kernels = [4,4,4,4]
        strides = [2,2,2,1]

        self.img_rows = 32
        self.img_cols = 32
        self.channel = 3
        (self.x_train, y_train), (_, _) = cifar10.load_data()
        self.x_train=self.x_train[np.where([y==7 for y in y_train])[0]] #generate horse images
        self.x_train = self.x_train.reshape(-1, self.img_rows,\
        	self.img_cols, 3)/255*2-1

        self.GAN = GAN(img_rows=self.img_rows, img_cols=self.img_cols, channel=self.channel, 
                            load_dir=None, depth=128, save_dir = 'Saved_Models/CIFAR_example')
        self.GAN.strides = strides
        self.GAN.kernels = kernels

    def train(self):
        self.GAN.train(self.x_train, 'CIFAR_sims_',train_steps=50000, save_interval=500,
                       verbose = 100, train_rate=(5,1))

t = CIFARGAN()
t.train()