#from wgan_gp import WGAN_GP as GAN
#from wgan import WGAN as GAN
from dcgan import DCGAN as GAN
import numpy as np
import h5py 

class DustGAN(object):
    def __init__(self):
            
        kernels = [4,4,4,4]
        strides = [2,2,2,1]

        self.img_rows = 32
        self.img_cols = 32
        self.channel = 1
        
        data = 'D:/Projects/Planck-Dust-GAN/Planck_dust_cuts_353GHz_norm_log_2.h5'
        with h5py.File(data, 'r') as hf:
            self.x_train=np.array([np.array(i)[:self.img_rows,:self.img_cols] for i in hf.values()]).reshape(-1, self.img_rows, self.img_cols, 1).astype(np.float32)

        self.GAN = GAN(img_rows=self.img_rows, img_cols=self.img_cols, channel=self.channel, 
                            load_dir='Saved_Models/Dust_example', depth=128,batch_size=32, save_dir = 'Saved_Models/Dust_example')
        self.GAN.strides = strides
        self.GAN.kernels = kernels

    def train(self):
        self.GAN.train(self.x_train, 'Dust_sims_',train_steps=50000, save_interval=500,
                       verbose = 100, train_rate=(5,1))

t = DustGAN()
t.train()