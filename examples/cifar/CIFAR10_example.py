from gans.dcgan import DCGAN
from gans.wgan import WGAN
from gans.wgan_gp import WGAN_GP
from gans.ct_gan import CTGAN
import numpy as np
import os
from keras.datasets import cifar10

kernels = [5,5,5,5]
strides = [2,2,2,1]

img_rows = 32
img_cols = 32
channel = 3
(x_train, y_train), (_, _) = cifar10.load_data()
x_train=x_train[np.where([y==7 for y in y_train])[0]] #generate horse images
x_train = x_train.reshape(-1, img_rows, img_cols, 3)/255*2-1

current_dir = os.path.dirname(__file__)

cifarGAN = DCGAN((img_rows,img_cols,channel), 
        load_dir=None,
        kernels = kernels,
        strides = strides,
        min_depth = 32,
        save_dir = os.path.join(current_dir,'Saved_Model/DCGAN'))
        

cifarGAN.train(x_train,
            os.path.join(current_dir,'DCGAN_figures/cifar'),
            train_steps=5000,
            save_rate=1000,
            mesg_rate = 100,
            train_rate=(1,1),
            batch_size= 32)

cifarGAN = WGAN((img_rows,img_cols,channel), 
        load_dir=None,
        kernels = kernels,
        strides = strides,
        min_depth = 32,
        save_dir = os.path.join(current_dir,'Saved_Model/WGAN'))
        

cifarGAN.train(x_train,
            os.path.join(current_dir,'WGAN_figures/cifar'),
            train_steps=5000,
            save_rate=1000,
            mesg_rate = 100,
            train_rate=(5,1),
            batch_size= 32,
            )

cifarGAN = WGAN_GP((img_rows,img_cols,channel), 
        load_dir=None,
        kernels = kernels,
        strides = strides,
        min_depth = 32,
        save_dir = os.path.join(current_dir,'Saved_Model/WGAN_GP'))
        

cifarGAN.train(x_train,
            os.path.join(current_dir,'WGAN_GP_figures/cifar'),
            train_steps=5000,
            save_rate=1000,
            mesg_rate = 100,
            train_rate=(5,1),
            batch_size= 32,
            )

cifarGAN = CTGAN((img_rows,img_cols,channel), 
        load_dir=None,
        kernels = kernels,
        strides = strides,
        min_depth = 32,
        save_dir = os.path.join(current_dir,'Saved_Model/CTGAN'))
        

cifarGAN.train(x_train,
            os.path.join(current_dir,'CTGAN_figures/cifar'),
            train_steps=5000,
            save_rate=1000,
            mesg_rate = 100,
            train_rate=(5,1),
            batch_size= 32,
            )
