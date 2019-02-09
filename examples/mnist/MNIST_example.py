from gans.dcgan import DCGAN
from gans.wgan import WGAN
from gans.wgan_gp import WGAN_GP
from gans.ct_gan import CTGAN
import numpy as np
from keras.datasets import mnist
import os

kernels = [5,5,5]
strides = [2,2,2]

img_rows = 28
img_cols = 28
channel = 1
(x_train, _), (_, _) = mnist.load_data()
x_train = x_train.reshape(-1, img_rows,img_cols, 1)/255*2-1

current_dir = os.path.dirname(__file__)

mnistGAN = DCGAN((img_rows,img_cols,channel), 
        load_dir=None,
        kernels = kernels,
        strides = strides,
        min_depth = 32,
        save_dir = os.path.join(current_dir,'Saved_Model/DCGAN'))
        

mnistGAN.train(x_train,
            os.path.join(current_dir,'DCGAN_figures/mnist'),
            train_steps=5000,
            save_rate=1000,
            mesg_rate = 100,
            train_rate=(1,1),
            batch_size= 32)

mnistGAN = WGAN((img_rows,img_cols,channel), 
        load_dir=None,
        kernels = kernels,
        strides = strides,
        min_depth = 32,
        save_dir = os.path.join(current_dir,'Saved_Model/WGAN'))
        

mnistGAN.train(x_train,
            os.path.join(current_dir,'WGAN_figures/mnist'),
            train_steps=5000,
            save_rate=1000,
            mesg_rate = 100,
            train_rate=(1,1),
            batch_size= 32,
            )

mnistGAN = WGAN_GP((img_rows,img_cols,channel), 
        load_dir=None,
        kernels = kernels,
        strides = strides,
        min_depth = 32,
        save_dir = os.path.join(current_dir,'Saved_Model/WGAN_GP'))
        

mnistGAN.train(x_train,
            os.path.join(current_dir,'WGAN_GP_figures/mnist'),
            train_steps=5000,
            save_rate=1000,
            mesg_rate = 100,
            train_rate=(1,1),
            batch_size= 32,
            )

mnistGAN = CTGAN((img_rows,img_cols,channel), 
        load_dir=None,
        kernels = kernels,
        strides = strides,
        min_depth = 32,
        save_dir = os.path.join(current_dir,'Saved_Model/CTGAN'))
        

mnistGAN.train(x_train,
            os.path.join(current_dir,'CTGAN_figures/mnist'),
            train_steps=5000,
            save_rate=1000,
            mesg_rate = 100,
            train_rate=(1,1),
            batch_size= 32,
            )

