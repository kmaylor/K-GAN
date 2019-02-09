""" Utility functions to be used for tracking the progress, saving, loading, and plotting outputs for gans.
"""

import os
import pickle as pk
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model

# Suppress warning message from Keras when loading the discriminator. Keras does not like
# it when you change model.trainable and save without compiling. But you don't need to
# compile the discriminator again. Possible bug.
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class ProgressLogger(object):
    """ Class used to track the losses of the different models in a GAN. 
        It's methods create log files and plots for the given losses.
        # Need to implement methods for determining stopping criteria and calculating statistics of
        generator outputs. Also need to allow for more models than just discriminator and generator.
    
    # Arguments
        fileprefix: Path to directory in which logs and plots will be saved. If not specified will save to the
            working directory.
        nan_threshold: The allowed number of consecutive times the total loss for all models is allowed to be
            NaN before stopping training.
        mesg_rate: The number of steps between printing losses for each model.
        save_rate: The number of steps between saving models.
    """

    def __init__(self,
                 fileprefix = os.path.dirname(__file__),
                 nan_threshold = 100,
                 mesg_rate = 100,
                 save_rate = 500):
        
        self.mesg_rate = mesg_rate
        self.save_rate = save_rate
        self.nan_loss_counter = 0
        self.nan_threshold = nan_threshold
        self.loss_acc = {'Discriminator':[],'Adversarial':[]}
        #Create file for loss statistics
        self.fileprefix = fileprefix
        self.step = 0
        with open(self.fileprefix+'_losses.txt','wb') as f:
            pk.dump(self.loss_acc,f)

    def update(self, gan, d_loss, a_loss, x_samples = None, num_samples = 16):
        """ Updates a ProgressLogger instance with given loss values and GAN
        
        # Arguments
            gan: Instance of the GAN being trained.
            d_loss: Discriminator loss.
            a_loss: Adversarial loss.
            x_samples = Samples of training data. Used to generate a plot comparing len(x_samples) real 
                and len(x_samples) generated images.
            num_samples: The number of generated images to plot if x_samples == None. Specifying x_samples = None
                and num_samples = 0 results in no plot being generated.
        """
        
        self.loss_acc['Discriminator'].append(d_loss)
        self.loss_acc['Adversarial'].append(a_loss)
        self.step += 1
        if self.step%self.mesg_rate == 0: self.log_mesg()
        total_loss = np.sum(d_loss+a_loss)
        if np.isnan(total_loss):
                self.nan_loss_count+=1
                if self.nan_loss_count >= self.nan_threshold:
                    raise ValueError('Loss has been NaN for %i training steps'%(self.nan_threshold))
        else:
            self.nan_loss_count=0
        if len(self.loss_acc['Discriminator'])%self.save_rate == 0:
            self.save_losses()
            save_state(gan)
            if (x_samples is not None) or (num_samples != 0):
                self.plot_losses()
                plot_samples(gan, self.fileprefix+'_samples_'+str(self.step), x_samples = x_samples, num_samples = num_samples)

    def log_mesg(self):
        log_mesg = "%d: [D loss: %f]" % (self.step, self.loss_acc['Discriminator'][-1])
        log_mesg = "%s  [A loss: %f]" % (log_mesg, self.loss_acc['Adversarial'][-1])
        print(log_mesg)

    def save_losses(self):
        with open(self.fileprefix+'_losses.txt','rb') as f:
            old=pk.load(f)
            for k in old.keys():
                old[k].extend(self.loss_acc[k])
        with open(self.fileprefix+'_losses.txt','wb') as f:
            pk.dump(old,f)
        self.loss_acc={'Discriminator':[],'Adversarial':[]}

    def plot_losses(self):
        with open(self.fileprefix+'_losses.txt','rb') as f:
            stats=pk.load(f)
            d_loss = stats['Discriminator']
            a_loss = stats['Adversarial']
        plt.figure(figsize=(10,5))
        plt.subplot(2,1,1)
        plt.plot(d_loss,'-')
        plt.title('Discriminator Loss')
        plt.subplot(2,1,2)
        plt.plot(a_loss,'-')
        plt.title('Generator Loss')
        plt.tight_layout()
        plt.savefig(self.fileprefix+'_loss_plots.png')
            
            
            
            
def plot_samples(gan, filename, x_samples = None, num_samples = 8):
    """ Function for creating plots of generated and real samples.
        
        # Arguments
            gan: Instance of the GAN being trained.
            filename: Name of .png file.
            x_samples = Samples of training data. Used to generate a plot comparing len(x_samples) real 
                and len(x_samples) generated images.
            num_samples: The number of generated images to plot if x_samples == None. Specifying x_samples = None
                and num_samples = 0 results in no plot being generated.
    """
    if x_samples is not None:
        samples = len(x_samples)
        set1 = x_samples
        labels1 = ['Real']*samples
    else:
        samples = int(num_samples/2)
        noise = np.random.normal(loc=0., scale=1., size=[samples, gan.latent_dim])
        set1 = gan.models['generator'].predict(noise)
        labels1 = ['Generated']*samples
    noise = np.random.normal(loc=0., scale=1., size=[samples, gan.latent_dim])
    set2 = gan.models['generator'].predict(noise)
    labels2 = ['Generated']*samples
    images = np.concatenate((set2, set1))
    labels = labels2 + labels1
    plt.figure(figsize=(10,10))
    for i in range(images.shape[0]):
        plt.subplot(int((samples*2)**.5), int((samples*2)**.5), i+1)
        image = images[i, :, :, :]
        if gan.channel==1:
            image = np.reshape(image, [gan.img_rows, gan.img_cols])
        plt.imshow(((image+1)*255/2).astype('int'))
        plt.axis('off')
        plt.title(labels[i])
    plt.tight_layout()
    plt.savefig(filename+'.png')
    plt.close('all')

        
def save_state(gan):
    if not os.path.exists(str(gan.save_dir)): os.makedirs(str(gan.save_dir))
    for k in ['discriminator','generator']:
        gan.models[k].save(gan.save_dir+'/'+k+'.h5')
           
                        
def load_state(gan):
    for k in ['discriminator','generator']:
        gan.models[k] = load_model(str(gan.load_dir)+'/'+k+'.h5')

        
def get_model_memory_usage(self,batch_size, model):
    ''' Get memory usage of a model during training.
    
    # Arguments
        batch_size: Number of samples to be used in a single forward pass during training.
        model: A model instance.
    '''
    model = getattr(self,model)
    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes   