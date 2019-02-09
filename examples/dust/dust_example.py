#from wgan_gp import WGAN_GP as GAN
#from wgan import WGAN as GAN
from dcgan import DCGAN as GAN
#from ct_gan import CTGAN as GAN
import numpy as np
import h5py 
from skimage.transform import rescale
import matplotlib.pyplot as plt

class DustGAN(object):
    def __init__(self):
            
        kernels = [5,5,5]
        strides = [2,2,2]

        self.img_rows = 128
        self.img_cols = 128
        self.channel = 1
        
        data = 'D:/Projects/Planck-Dust-GAN/Planck_dust_cuts_353GHz_norm_log_2.h5'
        with h5py.File(data, 'r') as hf:
            self.x_train=np.array([rescale(np.array(i),self.img_rows/900.,preserve_range=True,anti_aliasing=False)
             for i in hf.values()]).reshape(-1, self.img_rows, self.img_cols, 1).astype(np.float32)
        
        self.GAN = GAN(img_rows=self.img_rows,
                        img_cols=self.img_cols,
                        channel=self.channel,
                        strides=strides,
                        kernels=kernels,
                        load_dir=None,
                        min_depth=64,
                        save_dir = 'Saved_Models/Dust_example')
        
    def train(self):
        self.GAN.train(self.x_train, 'Dust_sims_',train_steps=50000,batch_size=64, save_interval=5,
                       verbose = 1, train_rate=(1,1))

t = DustGAN()

t.train()