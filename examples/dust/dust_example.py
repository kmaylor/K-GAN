from gans.wgan_gp import WGAN_GP as GAN
#from gans.wgan import WGAN as GAN
#from gans.dcgan import DCGAN as GAN
#from gans.ct_gan import CTGAN as GAN
import numpy as np
import h5py 
from skimage.transform import rescale
import matplotlib.pyplot as plt
import os

current_dir = os.path.dirname(__file__)

kernels = [5,5,5]
strides = [2,2,2]

img_rows = 128
img_cols = 128
channel = 1
        
data = os.path.join(current_dir,'Planck_dust_cuts_353GHz_norm_log_res256.h5')

with h5py.File(data, 'r') as hf:
            x_train=np.array([rescale(np.array(i),img_rows/256.,preserve_range=True,anti_aliasing=False)
             for i in hf.values()]).reshape(-1, img_rows,img_cols, 1).astype(np.float32)

dustGAN = GAN((img_rows,img_cols,channel),
                        strides=strides,
                        kernels=kernels,
                        load_dir=os.path.join(current_dir,'Saved_Model/WGAN_GP'),
                        min_depth=64,
                        save_dir = os.path.join(current_dir,'Saved_Model/WGAN_GP'))
        
dustGAN.train(x_train,
              os.path.join(current_dir,'WGAN_GP_figures/dust'),
              train_steps=50000,
              batch_size=32,
              save_rate=500,
              mesg_rate = 100, 
              train_rate=(5,1),
             )
