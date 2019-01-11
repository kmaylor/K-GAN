import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle as pk
from os.path import exists
from os import makedirs
import types
from keras import __version__
print("Using Keras version = "+__version__)
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Cropping2D, UpSampling2D
from keras.layers import LeakyReLU, Dropout, Lambda, ReLU
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.backend import log, count_params, int_shape
from keras.initializers import TruncatedNormal, Zeros
from keras.utils import multi_gpu_model
import tensorflow as tf
from keras import backend as K
import os


class WGAN(object):
    def __init__(self, img_rows, img_cols,
                    channel=1,
                    load_dir =None,
                    save_dir = 'Saved_Models/testing',
                    kernels = [4,4,4,4,2],
                    strides = [2,2,2,1,1],
                    depth = 64,
                    depth_scale = None,
                    input_dim = 100,
                    clip_value = 0.01,
                    gpus=1,
                    ):
        
        self.clip_value = clip_value
        self.gpus = gpus
        self.save_dir = save_dir
        self.load_dir = load_dir
        self.input_dim = input_dim
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.DM = None  # discriminator model
        self.AM = None  # adversarial model
        self.kernels = kernels
        self.strides = strides
        self.depth = depth
        
        if depth_scale == None:
            self.depth_scale = lambda:((2*np.ones(len(self.kernels)))**np.arange(len(self.kernels))).astype('int')
        
        if load_dir != None:
            try:
                print('Loading Previous State')
                self.load_state()
            except IOError:
                print('State '+str(load_dir)+' not found, beginning with fresh state.')

    def discriminator(self):
        '''Create the discriminator if it does not already exist'''
        if self.D:
            return self.D
        
        #initialize weights from normal distribution with 1-sigma cutoff
        initial = TruncatedNormal(0,0.02)
        bias_initial = Zeros()

        # depth*scale_depth give the number of features for each layer
        depth = self.depth
        if isinstance(self.depth_scale,types.FunctionType):
            depth_scale = self.depth_scale()
        else:
            depth_scale = self.depth_scale
        
        input_shape = (self.img_rows, self.img_cols, self.channel)

        # Discriminator is sequential model
        self.D = Sequential(name='Discriminator')
        # First layer of discriminator i a convolution, minimum of two convolutional layers
        self.D.add(Conv2D(depth*depth_scale[0], self.kernels[0], strides=self.strides[0], \
                    input_shape=input_shape, padding='same', kernel_initializer=initial, bias_initializer=bias_initial, name = 'Conv2D_1'))
        
        # Iterate over layers defined by the number of kernels and strides
        for i,ks in enumerate(zip(self.kernels[1:],self.strides[1:])):
            self.D.add(LeakyReLU(alpha=0.2, name = 'LRelu_D%i'%(i+1)))
            self.D.add(BatchNormalization(momentum=0.9, name = 'BN_D%i'%(i+1)))
            self.D.add(Conv2D(depth*depth_scale[i+1], ks[0], strides=ks[1], padding='same', \
                        kernel_initializer=initial,bias_initializer=bias_initial, name = 'Conv2D_D%i'%(i+2)))
        
        self.D.add(LeakyReLU(alpha=0.2, name = 'LRelu_D%i'%(i+2)))
            
        self.D.add(BatchNormalization(momentum=0.9, name = 'BN_D%i'%(i+2)))
        
        # Flatten final features and calculate the probability of the input belonging to the same 
        # as the training set
        self.D.add(Flatten(name = 'Flatten'))
        self.D.add(Dense(1, kernel_initializer=initial,bias_initializer=bias_initial, name = 'Dense_D'))
        self.D.summary()
        return self.D

    def generator(self):
        '''Create the generator if it does not already exist'''
        if self.G:
            return self.G
        
        #initialize weights from normal distribution with 1-sigma cutoff
        initial = TruncatedNormal(0,0.02)
        bias_initial = Zeros()
        
        # depth/2*scale_depth give the number of features for each layer
        depth = self.depth
        if isinstance(self.depth_scale,types.FunctionType):
            depth_scale = self.depth_scale()[::-1]
        else:
            depth_scale = self.depth_scale[::-1]
        
        #Get the size of the initial features from the size of the final feature layer in 
        #the discriminator
        dim1 = self.discriminator().get_layer('Flatten').input_shape[1]
        dim2 = self.discriminator().get_layer('Flatten').input_shape[2]
        
        # Generator is sequential model
        self.G = Sequential(name='Generator')
        # First layer of generator is densely connected
        self.G.add(Dense(dim1*dim2*depth*depth_scale[0], input_dim=self.input_dim,
                        kernel_initializer=initial,bias_initializer=bias_initial, name = 'Dense_G'))
        self.G.add(LeakyReLU(alpha=.2, name = 'LRelu_G1'))
        self.G.add(Reshape((dim1, dim2, depth*depth_scale[0]),name='Reshape'))

        # Iterate over layers defined by the number of kernels and strides
        #Use larger kernels with larger feature maps
        for i,ks in enumerate(zip(self.kernels,self.strides)):
            self.G.add(BatchNormalization(momentum=0.9, name = 'BN_G%i'%(i+1)))
            if i < len(self.kernels)-1:
                self.G.add(UpSampling2D(ks[1],name='UpSample_%i'%(i+1), interpolation='bilinear'))
                self.G.add(Conv2D(depth*depth_scale[i+1], ks[0], strides = 1, padding='same',
                        kernel_initializer=initial,bias_initializer=bias_initial, name = 'Conv2D_G%i'%(i+1)))
                self.G.add(LeakyReLU(alpha=.2, name = 'LRelu_G%i'%(i+2)))
            else:
                self.G.add(UpSampling2D(self.strides[-1],name='UpSample_%i'%(i+1), interpolation='bilinear'))
                self.G.add(Conv2D(self.channel, self.kernels[-1], strides = 1, padding='same',
                        kernel_initializer=initial,bias_initializer=bias_initial, name = 'Conv2D_G%i'%(i+1)))
                self.G.add(Activation('tanh', name = 'Tanh'))
        
        # If the output of the last layer is larger than the input for the discriminator crop
        # the image
        crop_r = int((self.G.get_layer('Tanh').output_shape[1]-self.img_rows)/2)
        crop_c = int((self.G.get_layer('Tanh').output_shape[2]-self.img_cols)/2)
        self.G.add(Cropping2D(cropping=((crop_c,crop_c),(crop_r,crop_r)), name = 'Crop2D'))
        self.G.summary()
        return self.G

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)
    
    def discriminator_model(self):
        '''Build and compile the discriminator model from the discriminator. This allows for 
        easy saving and reloading of models'''
        if self.DM:
            return self.DM        

        # Compile the discriminator model on the number of specified gpus
        if self.gpus <=1:
            self.DM = Sequential(name = 'Discriminator Model')
            self.DM.add(self.discriminator())
            optimizer = RMSprop(lr=0.00005)
            self.DM.compile(loss=self.wasserstein_loss, optimizer=optimizer)
        else:
            with tf.device("/cpu:0"):
                self.DM = Sequential(name = 'Discriminator_Model')
                self.DM.add(self.discriminator())
            self.DM = multi_gpu_model(self.DM,gpus=self.gpus)
            optimizer = RMSprop(lr=0.00005)
            self.DM.compile(loss=self.wasserstein_loss, optimizer=optimizer)

        self.DM.summary()
        return self.DM

    def adversarial_model(self):
        '''Build and compile the adversarial model from the discriminator and generator. Stacks the 
        discriminator on the generator'''
        if self.AM:
            return self.AM

        # Compile the gemerator model on the number of specified gpus
        if self.gpus <=1:
            self.AM = Sequential(name = 'Adversarial Model')
            self.AM.add(self.generator())
            discriminator =self.discriminator()
            # Only use the discriminator to evaluate the generator's output
            discriminator.trainable=False
            self.AM.add(discriminator)
            optimizer = RMSprop(lr=0.00005)
            self.AM.compile(loss=self.wasserstein_loss, optimizer=optimizer)
        else:
            with tf.device("/cpu:0"):
                self.AM = Sequential(name = 'Adversarial_Model')
                self.AM.add(self.generator())
                discriminator =self.discriminator()
                # Only use the discriminator to evaluate the generator's output
                discriminator.trainable=False
                self.AM.add(discriminator)
            self.AM = multi_gpu_model(self.AM,gpus=self.gpus)
            optimizer = RMSprop(lr=0.00005)
            self.AM.compile(loss=self.wasserstein_loss, optimizer=optimizer)

        self.AM.summary()
        # Set the discriminator back to trainable so the discriminator is in the correct state when reloading
        # a model
        discriminator.trainable=True
        return self.AM

    
    
    def train(self, x_train, filename, train_rate=(1,2),
                    train_steps=2000, batch_size=32,
                    save_interval=100, verbose = 10,
                    samples=16):
        '''Trains the generator and discriminator on x_train for batches = train_steps
        filename: path to where to save the sample images
        train_rate: iterable containing number of times to train the discriminator and the generator in that order
        train_steps: number of batches to train on before stopping
        batch_size: number of samples to train discriminator and generator on each step
        save_interval: number of steps afterwhich the models and samples will be saved
        verbose: number of steps afterwhich the discriminator and adversarial model loss and accuracy will be d
                 printed
        samples: number of images in output plot
        '''
        self.G=self.generator()
        self.DM=self.discriminator_model()
        self.AM=self.adversarial_model()
        loss_acc={'Discriminator':[],'Adversarial':[]}
        with open(filename+'stats.txt','wb') as f:
            pk.dump(loss_acc,f)
        
        print('Training Beginning')
        y_real = -np.ones((batch_size, 1))
        y_fake = np.ones((batch_size, 1))
            
        for i in range(train_steps):
            if i%1000 == 0:
                rate = 100
            else:
                rate = train_rate[0]
            for k in range(train_rate[0]):
                # First train the discriminator with correct labels
                # Randomly select batch from training samples
                images_real = x_train[np.random.randint(0,
                    x_train.shape[0], size=batch_size), :, :, :]
                
                # Generate fake images from generator
                noise = np.random.normal(loc=0., scale=1., size=[batch_size, self.input_dim])
                images_fake = self.G.predict(noise)
                
                d_loss_real = self.DM.train_on_batch(images_real, y_real)
                d_loss_fake = self.DM.train_on_batch(images_fake,y_fake)
                d_loss = np.add(d_loss_fake,d_loss_real)*0.5
                
                for l in self.DM.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)
            # Now train the adversarial network
            # Create new fake images labels as if they are from the training set
            
            for j in range(train_rate[1]):
                y_lie = -np.ones([batch_size, 1])
                noise = np.random.normal(loc=0., scale=1., size=[batch_size, self.input_dim])
                a_loss = self.AM.train_on_batch(noise, y_lie)
            
            # Generate log messages
            if np.isnan(np.sum(d_loss+a_loss)):
                print('Loss is nan')
                break
            loss_acc['Discriminator'].append(d_loss)
            loss_acc['Adversarial'].append(a_loss)
            log_mesg = "%d: [D loss: %f]" % (i, d_loss)
            log_mesg = "%s  [A loss: %f]" % (log_mesg, a_loss)
            if i%verbose==0:
                print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.save_state()
                    fn = filename+"_%d.png" % (i+1)
                    self.plot_images(fake=images_fake, real=images_real,seed=None, filename=fn, samples=samples)
                    with open(filename+'stats.txt','rb') as f:
                        old=pk.load(f)
                        for k in old.keys():
                            old[k].extend(loss_acc[k])
                    with open(filename+'stats.txt','wb') as f:
                        pk.dump(old,f)
                    loss_acc={'Discriminator':[],'Adversarial':[]}
                    self.plot_stats(filename)

    def plot_stats(self, filename):
        with open(filename+'stats.txt','rb') as f:
            stats=pk.load(f)
            d_loss = stats['Discriminator']
            a_loss=stats['Adversarial']
        plt.figure(figsize=(10,5))
        plt.plot(d_loss,'-',label='Discriminator')
        plt.plot(a_loss,'--',label='Adversarial')
        plt.title('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename+'stats_plots')


    def plot_images(self, fake=None, real=None, seed=None, filename=None, samples=16):
        '''plot samples from the generator or the training data
        fake: List of images from the generator overridden if seed != None
        real: List of images from the training set
        seed: Fix value to always produce the same images from the generator
        filename: Path to where to save images
        samples: Number of images to plot'''
        if seed!=None:
            np.random.seed(seed)
            noise = np.random.normal(loc=0., scale=1., size=[16, self.input_dim])
            fake = self.G.predict(noise)
        if real is not None:
            images = np.concatenate((fake[:int(samples/2)],real[:int(samples/2)]))
        else:
            images = fake
        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(int(samples**.5), int(samples**.5), i+1)
            image = images[i, :, :, :]
            if self.channel==1:
                image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(((image+1)*255/2).astype('int'))
            plt.axis('off')
        plt.tight_layout()
        if filename!=None:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()
            
    def save_state(self):
        '''Save only the discriminator and generator DM and AM are re-compiled, allows you to change learning rates
        during training'''
        if not exists(str(self.save_dir)): makedirs(str(self.save_dir))
        model_type = ['D', 'G']
        for m in model_type:
            model = getattr(self, m)
            model.save(str(self.save_dir)+'/'+m+'_model.h5')
            # serialize model to JSON
            #with open(m+".json", "w") as f: f.write(model.to_json())
            # serialize weights to HDF5
            #model.save_weights(m+"_weights.h5")
                        
    def load_state(self):
        '''Load only the discriminator and generator, DM and AM are re-compiled, allows you to change learning rates
        during training '''
        model_type = ['D', 'G']
        for m in model_type:
            setattr(self,m,load_model(str(self.load_dir)+'/'+m+'_model.h5'))
            # load json and create model
            #with open(m+'.json', 'r') as f: setattr(self,m,model_from_json(f.read()))
            # load weights into new model
            #getattr(self, m).load_weights(m+"_weights.h5", by_name=True)

    def get_model_memory_usage(self,batch_size, model):
        ''' Get memory usage of a model during training.
            model = [D,G,DM,AM]
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

        trainable_count = np.sum([count_params(p) for p in set(model.trainable_weights)])
        non_trainable_count = np.sum([count_params(p) for p in set(model.non_trainable_weights)])

        total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
        gbytes = np.round(total_memory / (1024.0 ** 3), 3)
        return gbytes



    
            
        
