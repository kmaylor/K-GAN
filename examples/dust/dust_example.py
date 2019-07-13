# from gans.wgan_gp import WGAN_GP# as GAN
# from gans.wgan import WGAN# as GAN
#from gans.dcgan import DCGAN as GAN
# from gans.ct_gan import CTGAN# as GAN
# from gans.ggan import GGAN# as GAN
from gans.powergan import PowerGAN
import numpy as np
import h5py 
import matplotlib.pyplot as plt
import os
from power_spectrum_callback import PSDCallback

current_dir = os.path.dirname(__file__)

kernels = [5,5,5,5]
strides = [2,2,2,2]

img_rows = 256
img_cols = 256
channel = 1
        
data = os.path.join(current_dir,'Planck_dust_cuts_353GHz_norm_log_res256.h5')

with h5py.File(data, 'r') as hf:
            x_train=np.array([i for i in hf.values()]).reshape(-1, img_rows,img_cols, 1).astype(np.float32)

call_back = PSDCallback(x_train)



dustGAN = PowerGAN((img_rows,img_cols,channel),
                        strides=strides,
                        kernels=kernels,
                        load_dir=None,
                        min_depth=64,
                        latent_dim = 64,
                        gpus=2,
                        save_dir = os.path.join(current_dir,'Saved_Model/PowerGAN16_15'))
        
dustGAN.train(x_train,
              os.path.join(current_dir,'PowerGAN_figures/dust'),
              train_steps=40000,
              batch_size=16,
              save_rate=200,
              mesg_rate = 100, 
              train_rate=(1,1),
              noisy_label=0.0,
              call_back = call_back
             )

               
