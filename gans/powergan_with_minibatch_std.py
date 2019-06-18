import sys
sys.path.append('..')
import numpy as np
import os
import types
from keras import __version__
print("Using Keras version = "+__version__)
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Reshape, Input, Concatenate
from keras.layers import Conv2D, Cropping2D, UpSampling2D, Conv1D
from keras.layers import LeakyReLU, Dropout, Lambda, ReLU
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from keras import backend as K
import tensorflow as tf
from utils.utils import *
from tensorflow_power_spectrum import PowerSpectrum

ps = PowerSpectrum(image_size=256)

class PowerGAN(object):
    """ Class for quickly building a DCGAN model (Radford et al. https://arxiv.org/pdf/1511.06434.pdf)
    
    # Arguments
        img_rows: Number of rows in an image from the training set.
        img_cols: Number of columns in an image from the training set.
        channel: Number of channels in an image from the training set, 1 for grayscale, 3 for rgb.
        kernels: A list of ints containing the kernel size to be used for each convolution in
            the discriminator, assumed symmetric. The reversed list is used for the generator kernels.
        strides: A list of ints containing the stride or scaling to be used in each convolution.
        min_depth: The number of features created at the first convolution in the discriminator, or
            the number of features created at the second to last convolution in the generator. 
        depth_scale: Function or tuple of ints to deterimine the number of features created at each convolution
            after the first convolution in the discriminator. Default is to increase by a factor of 2 after each
            convolution.
        latent_dim: Number of dimensions of the latent vector. Drawn from normal disctribution.
        load_dir: Directory containing saved models used as starting point.
        save_dir: Directory to save GAN models in.
        gpus: Number of gpus to use for training.
        discriminator_optimizer: Optimizer to use for discriminator training. Default is 
            Adam(lr=0.0002,beta_1=0.5, decay=0)
        generator_optimizer: Optimizer to use for generator training. Default is 
            Adam(lr=0.0002,beta_1=0.5, decay=0)
    """
    def __init__(self,
                    img_dims,
                    kernels,
                    strides,
                    min_depth,
                    depth_scale = None,
                    latent_dim = 64,
                    load_dir = None,
                    save_dir = 'Saved_Models/dcgan',
                    gpus = 1,
                    discriminator_optimizer = Adam(lr=0.0002,beta_1=0.5, decay=0),
                    generator_optimizer = Adam(lr=0.0002,beta_1=0.5, decay=0)
                    ):
        
        self.img_rows, self.img_cols, self.channel = img_dims
        self.gpus = gpus
        self.save_dir = save_dir
        self.load_dir = load_dir
        self.latent_dim = latent_dim
        self.kernels = kernels
        self.strides = strides
        self.depth = min_depth
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_optimizer = generator_optimizer
        # If depth_scale not provided increase the number of features after each convolution by factor of 2.
        if depth_scale == None:
            self.depth_scale = lambda:((2*np.ones(len(self.kernels)))**np.arange(len(self.kernels))).astype('int')
        
        self.models = dict.fromkeys(['discriminator','power_discriminator',
                                     'generator','discriminator_model','adversarial_model'])

        if load_dir != None:
            print('Loading Previous State')
            load_state(self,models=['discriminator','power_discriminator','generator'])
        else:
            self.models['discriminator'] = self.build_discriminator()   # discriminator
            self.models['power_discriminator'] = self.build_power_discriminator()   # discriminator
            self.models['generator'] = self.build_generator()   # generator

        self.models['discriminator_model'] = self.build_discriminator_model()  # discriminator model
        self.models['adversarial_model'] = self.build_adversarial_model()  # adversarial model

    def batch_std(self,x):
            shape = K.shape(x)
            dims = [shape[i] for i in range(len(x.shape)-1)]+[1]
            s = K.std(x,keepdims=True, axis=np.arange(1,len(x.shape)-1))
            s = tf.reduce_mean(x,keep_dims=True)
            s = K.tile(s, dims)
            return K.concatenate([x, s], axis=-1)
    
    
    def build_discriminator(self):
        '''Create the discriminator'''

        def discriminator_block(x, depth, kernel_size, stride):
            x = Conv2D(depth, kernel_size, strides=stride, padding='same')(x)
            x = BatchNormalization(momentum=0.9)(x)
            return LeakyReLU(alpha=0.2)(x)
        
        # depth*scale_depth give the number of features for each layer
        depth = self.depth
        if isinstance(self.depth_scale,types.FunctionType):
            depth_scale = self.depth_scale()
        else:
            depth_scale = self.depth_scale
        
        input_shape = (self.img_rows, self.img_cols, self.channel)

        # Discriminator is sequential model
        input_img = Input(input_shape)
        # First layer of discriminator is a convolution, minimum of two convolutional layers
        x = Conv2D(depth*depth_scale[0], self.kernels[0], strides=self.strides[0])(input_img)
        x = LeakyReLU(alpha=0.2)(x)

        # Iterate over layers defined by the number of kernels and strides
        for i,ks in enumerate(zip(self.kernels[1:],self.strides[1:])):
            if i==len(self.kernels[1:])-1:
                x = Lambda(self.batch_std)(x)
            x = discriminator_block(x,depth*depth_scale[i+1],ks[0],ks[1])
        
        # Flatten final features and calculate the probability of the input belonging to the same 
        # as the training set
        x = Flatten(name = 'Flatten')(x)
        x = Dense(1)(x)
        D = Model(inputs=input_img,outputs=x, name='Image_Discriminator')

        D.summary()
        return D
    
    
    def build_power_discriminator(self):
        '''Create the discriminator'''

        def discriminator_block(x, depth, kernel_size, stride):
            x = Conv1D(depth, kernel_size, strides=stride, padding='same')(x)
            x = BatchNormalization(momentum=0.9)(x)
            return LeakyReLU(alpha=0.2)(x)

        input_shape = (int(self.img_rows/2),1)
        input_power = Input(input_shape)
        # First layer of discriminator is a convolution, minimum of two convolutional layers
        x = Conv1D(32, 5, strides=2)(input_power)
        x = LeakyReLU(alpha=0.2)(x)
        x = discriminator_block(x,64,2,2)
        x = discriminator_block(x,128,2,1)
        x = Lambda(self.batch_std)(x)
        x = discriminator_block(x,128,2,1)
        x = Flatten()(x)
        x = Dense(128)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(1)(x)
        D = Model(inputs=input_power,outputs=x, name='Power_Discriminator')
        D.summary()
        return D


    def build_generator(self):
        '''Create the generator'''
        
        def generator_block(G,depth,kernel,stride):
            G.add(UpSampling2D(stride,name='UpSample_%i'%(i+1), interpolation='bilinear'))
            G.add(Conv2D(depth, kernel, strides = 1, padding='same',
                    kernel_initializer='he_normal',bias_initializer='zeros', name = 'Conv2D_G%i'%(i+1)))
            G.add(BatchNormalization(momentum=0.9, name = 'BN_G%i'%(i+2)))
            G.add(LeakyReLU(alpha=0.2, name = 'LRelu_G%i'%(i+2)))

        # depth/2*scale_depth give the number of features for each layer
        depth = self.depth
        if isinstance(self.depth_scale,types.FunctionType):
            depth_scale = self.depth_scale()[::-1]
        else:
            depth_scale = self.depth_scale[::-1]
        
        #Get the size of the initial features from the size of the final feature layer in 
        #the discriminator
        dim1 = self.models['discriminator'].get_layer('Flatten').input_shape[1]
        dim2 = self.models['discriminator'].get_layer('Flatten').input_shape[2]
        
        # Generator is sequential model
        G = Sequential(name='Generator')

        # First layer of generator is densely connected
        G.add(Dense(dim1*dim2*depth*depth_scale[0], input_dim=self.latent_dim,
                        kernel_initializer='he_normal',bias_initializer='zeros', name = 'Dense_G'))
        
        G.add(Reshape((dim1, dim2, depth*depth_scale[0]),name='Reshape'))
        G.add(BatchNormalization(momentum=0.9, name = 'BNorm_G1'))
        G.add(LeakyReLU(alpha=0.2, name = 'LRelu_G1'))

        # Iterate over layers defined by the number of kernels and strides
        #Use larger kernels with larger feature maps
        for i,ks in enumerate(zip(self.kernels[:-1],self.strides[:-1])):
            generator_block(G,depth*depth_scale[i+1],ks[0],ks[1])
        
        G.add(UpSampling2D(self.strides[-1],name='UpSample_%i'%(i+2), interpolation='bilinear'))
        G.add(Conv2D(self.channel, self.kernels[-1], strides = 1, padding='same',
                kernel_initializer='he_normal',bias_initializer='zeros', name = 'Conv2D_G%i'%(i+2)))
        G.add(Activation('tanh', name = 'Tanh'))
        
        # If the output of the last layer is larger than the input for the discriminator crop
        # the image
        crop_r = int((G.get_layer('Tanh').output_shape[1]-self.img_rows)/2)
        crop_c = int((G.get_layer('Tanh').output_shape[2]-self.img_cols)/2)
        G.add(Cropping2D(cropping=((crop_c,crop_c),(crop_r,crop_r)), name = 'Crop2D'))
        G.summary()
        return G
    
    def binary_crossentropy_custom(self,y_true,y_pred):
        return K.mean(K.binary_crossentropy(y_true, y_pred, from_logits=True), axis=-1)

    def build_discriminator_model(self):
        '''Build and compile the discriminator model from the discriminator.'''
        input_shape = (self.img_rows, self.img_cols, self.channel)
        input_img = Input(shape=input_shape)
        x = self.models['discriminator'](input_img)
        p=Reshape((self.img_rows, self.img_cols))(input_img)
        p = Lambda(lambda v: ps.power1D(v))(p)
        p = Reshape((int(self.img_rows/2),1))(p)
        p = self.models['power_discriminator'](p)
        # Compile the discriminator model on the number of specified gpus
        if self.gpus <=1:
            DM = Model(inputs=input_img,
                        outputs=[x,p], name='Discriminator_Model')
        else:
            with tf.device("/cpu:0"):
                DM = Model(inputs=input_img,
                        outputs=[x,p], name='Discriminator_Model')
            DM = multi_gpu_model(DM,gpus=self.gpus)
        DM.compile(loss=self.binary_crossentropy_custom, optimizer=self.discriminator_optimizer)

        DM.summary()
        return DM


    def build_adversarial_model(self):
        '''Build and compile the adversarial model from the discriminator and generator. Stacks the 
        discriminator on the generator'''

        # Only use the discriminator to evaluate the generator's output, don't want to update weights
        self.models['discriminator'].trainable = False
        self.models['power_discriminator'].trainable = False
            
        input_shape = (self.latent_dim,)
        input_noise = Input(shape=input_shape)
        gen = self.models['generator'](input_noise)
        x = self.models['discriminator'](gen)
        p=Reshape((self.img_rows, self.img_cols))(gen)
        p = Lambda(lambda v: ps.power1D(v))(p)
        p = Reshape((int(self.img_rows/2),1))(p)
        p = self.models['power_discriminator'](p)
        # Compile the generator model on the number of specified gpus
        if self.gpus <=1:
            AM = Model(inputs=input_img,
                        outputs=[x,p], name='Adversarial_Model')
        else:
            with tf.device("/cpu:0"):
                AM = Model(inputs=input_noise,
                        outputs=[x,p], name='Adversarial_Model')
            AM = multi_gpu_model(AM,gpus=self.gpus)
        AM.compile(loss=self.binary_crossentropy_custom, optimizer=self.generator_optimizer)

        # Set the discriminator back to trainable so the discriminator is in the correct state when reloading
        # a model
        self.models['discriminator'].trainable = True
        self.models['power_discriminator'].trainable = True
        AM.summary()
        return AM
    
    
    def train(self,
                x_train,
                fileprefix,
                train_rate=(1,2),
                train_steps=2000,
                batch_size=32,
                save_rate=100,
                mesg_rate = 10,
                samples=16,
                noisy_label=0.01,
                nan_threshold = 100,
                call_back = None):
        '''Trains the generator and discriminator.
        
        # Arguments
            x_train: Data to train models on.
            fileprefix: Path to where to save the sample images and log files.
            train_rate: Iterable containing number of times to train the discriminator
                and the generator in that order.
            train_steps: Number of batches to train on before stopping.
            batch_size: Number of samples to train discriminator and generator on each step.
            save_rate: Number of steps afterwhich the models and samples will be saved.
            mesg_rate: Number of steps afterwhich the models loss will be printed.
            samples: Number of images in output plot.
            noisy_label: Rate at which to swap training labels for the generator.
            nan_threshold: Number of allowed consecutive times the total loss for all models
                can be NaN before stopping training.
            call_back: A function that will be called at the same rate as mesg_rate and takes the
                current DCGAN instance as input.
        '''
        logger = ProgressLogger(fileprefix, mesg_rate = mesg_rate,
                                save_rate = save_rate, nan_threshold = nan_threshold,
                                 call_back=call_back,models_to_save=['discriminator','power_discriminator','generator'])
        
        print('Training Beginning')
        
            
        for i in range(train_steps):
            for k in range(train_rate[0]):
                # First train the discriminator with correct labels
                # Randomly select batch from training samples

                y_real = np.random.binomial(1,1-noisy_label,size=[batch_size,
                 1])
                y_fake = np.random.binomial(1,noisy_label,size=[batch_size, 1])
                
                images_real = x_train[np.random.randint(0,
                    x_train.shape[0], size=batch_size), :, :, :]
                
                # Generate fake images from generator
                noise = np.random.normal(loc=0., scale=1., size=[batch_size, self.latent_dim])
                images_fake = self.models['generator'].predict(noise)
                
                d_loss_real = self.models['discriminator_model'].train_on_batch(images_real, [y_real,y_real])
                d_loss_fake = self.models['discriminator_model'].train_on_batch(images_fake,[y_fake,y_fake])
            # Now train the adversarial network
            # Create new fake images labels as if they are from the training set
            #weight_clipper(self.models['discriminator_model'])
            for j in range(train_rate[1]):
                y_lie = np.ones([batch_size, 1])
                noise = np.random.normal(loc=0., scale=1., size=[batch_size, self.latent_dim])
                a_loss = self.models['adversarial_model'].train_on_batch(noise, [y_lie,y_lie])
            
            #Log losses and generator plots
            tracked = {'Discriminator Real loss':d_loss_real[0]/2,
                       'Discriminator Generated loss':d_loss_fake[0]/2,
                       'Average Discriminator loss': (d_loss_real[0]/2+d_loss_fake[0]/2)/2,
                       'Generator loss': a_loss[0]}
            logger.update(self, tracked, x_samples = images_real[:8])
            
            


        
