from kgan import KGAN
import numpy as np
from keras.datasets import mnist

class MNISTGAN(object):
    def __init__(self):
            
        kernels = [4,4,4,4,2]
        strides = [2,2,2,1,1]

        self.img_rows = 28
        self.img_cols = 28
        self.channel = 1

        (self.x_train, _), (_, _) = mnist.load_data()
        self.x_train = self.x_train.reshape(-1, self.img_rows,\
        	self.img_cols, 1).astype(np.float32)

        self.KGAN = KGAN(img_rows=self.img_rows, img_cols=self.img_cols, 
                            load_dir='Saved_Models/example', save_dir = 'Saved_Models/example')
        self.KGAN.strides = strides
        self.KGAN.kernels = kernels

    def train(self):
        self.KGAN.train(self.x_train, 'MNIST_sims',train_steps=8000, save_interval=100, verbose = 10)

t = MNISTGAN()
t.train()