print('Importing necessary packages and modules')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle as pk
from os.path import exists
from os import makedirs
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, Cropping2D, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.backend import log, count_params
from keras.initializers import TruncatedNormal
class KGAN(object):
    def __init__(self, img_rows, img_cols, channel=1,
                    load_dir =None,
                    save_dir = '',
                    kernels = [4,4,4,4,2],
                    strides = [2,2,2,1,1],
                    depth = 64,
                    depth_scale = None,
                    ):
        self.save_dir = save_dir
        self.load_dir = load_dir
        self.input_dim = 64
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.DM = None  # discriminator model
        self.AM = None  # adversarial model
        self.kernels = [15,10,5]
        self.strides = [8,4,2]
        self.depth = 64
        if depth_scale == None:
            self.depth_scale = lambda:((2*np.ones(len(self.kernels)))**np.arange(len(self.kernels))).astype('int')
        if load_dir != None:
            try:
            	print('Loading Previous State')
            	self.load_state(load_dir)
            except IOError:
                print('State '+load_dir+' not found, begin with fresh state?')

    def discriminator(self):
        if self.D:
            return self.D
        initial = TruncatedNormal(0,0.02)
        self.D = Sequential(name='Discriminator')
        depth = self.depth
        depth_scale = self.depth_scale()
        input_shape = (self.img_rows, self.img_cols, self.channel)
        
        self.D.add(Conv2D(depth*depth_scale[0], self.kernels[0], strides=self.strides[0], \
                    input_shape=input_shape, padding='same', kernel_initializer=initial, name = 'Conv2D_1'))
        
        for i,ks in enumerate(zip(self.kernels[1:],self.strides[1:])):
            self.D.add(LeakyReLU(alpha=0.2, name = 'LRelu_D%i'%(i+1)))
            self.D.add(BatchNormalization(momentum=0.9, name = 'BN_D%i'%(i+1)))
            self.D.add(Conv2D(depth*depth_scale[i+1], ks[0], strides=ks[1], padding='same', \
            	        kernel_initializer=initial, name = 'Conv2D_%i'%(i+2)))
        
        self.D.add(LeakyReLU(alpha=0.2, name = 'LRelu_D%i'%(i+2)))
        self.D.add(BatchNormalization(momentum=0.9, name = 'BN_D%i'%(i+2)))
        
        #self.D.add(Dropout(dropout))
        # Out: 1-dim probability
        self.D.add(Flatten(name = 'Flatten'))
        self.D.add(Dense(1, kernel_initializer=initial, name = 'Dense_D'))
        self.D.add(Activation('sigmoid', name = 'Sigmoid'))
        self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G
        initial = TruncatedNormal(0,0.02)
        self.G = Sequential(name='Generator')
        depth = int(self.depth/2)
        depth_scale = self.depth_scale()[::-1]
        dim1 = self.discriminator().get_layer('Flatten').input_shape[1]
        dim2 = self.discriminator().get_layer('Flatten').input_shape[2]
        

        self.G.add(Dense(dim1*dim2*depth*depth_scale[0], input_dim=self.input_dim,
                        kernel_initializer=initial, name = 'Dense_G'))
        self.G.add(LeakyReLU(alpha=0.2, name = 'LRelu_G_1'))
        self.G.add(Reshape((dim1, dim2, depth*depth_scale[0]),name='Reshape'))
        
        for i,ks in enumerate(zip(self.kernels,self.strides)):
        	self.G.add(BatchNormalization(momentum=0.9, name = 'BN_G%i'%(i+1)))
        	if i < len(self.kernels)-1:
        		#self.G.add(UpSampling2D())
        		self.G.add(Conv2DTranspose(depth*depth_scale[i+1], ks[0], strides = ks[1], padding='same',
        	            kernel_initializer=initial, name = 'ConvTr2D_%i'%(i+1)))
        		self.G.add(LeakyReLU(alpha=0.2, name = 'LRelu_G%i'%(i+2)))
        	else:
        		self.G.add(Conv2DTranspose(1, self.kernels[-1], strides = self.strides[-1], padding='same',
        	            kernel_initializer=initial, name = 'ConvTr2D_%i'%(i+1)))
        		self.G.add(Activation('tanh', name = 'Tanh'))
        
        crop_r = int((self.G.get_layer('Tanh').output_shape[1]-self.img_rows)/2)
        crop_c = int((self.G.get_layer('Tanh').output_shape[2]-self.img_cols)/2)
        self.G.add(Cropping2D(cropping=((crop_c,crop_c),(crop_r,crop_r)), name = 'Crop2D'))
        self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = Adam(lr=0.0002,beta_1=0.5, decay=0)
        self.DM = Sequential(name = 'Discriminator Model')
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['binary_accuracy'])
        self.DM.summary()
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = Adam(lr=0.0002,beta_1=0.5, decay=0)
        self.AM = Sequential(name = 'Adversarial Model')
        self.AM.add(self.generator())
        discriminator =self.discriminator()
        discriminator.trainable=False
        self.AM.add(discriminator)
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['binary_accuracy'])
        self.AM.summary()
        discriminator.trainable=True
        return self.AM

    def save_state(self):
        if not exists(self.save_dir): makedirs(self.save_dir)
        model_type = ['D', 'G']
        for m in model_type:
            model = getattr(self, m)
            model.save(self.save_dir+'/'+m+'_model.h5')
    		# serialize model to JSON
    		#with open(m+".json", "w") as f: f.write(model.to_json())
    		# serialize weights to HDF5
    		#model.save_weights(m+"_weights.h5")		
                        
    def load_state(self):
    	model_type = ['D', 'G']
    	for m in model_type:
    		setattr(self,m,load_model(self.load_dir+'/'+m+'_model.h5'))
            # load json and create model
            #with open(m+'.json', 'r') as f: setattr(self,m,model_from_json(f.read()))
            # load weights into new model
            #getattr(self, m).load_weights(m+"_weights.h5", by_name=True)

    def get_model_memory_usage(batch_size, model):
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
    
    def train(self, x_train, filename, train_rate=(1,2),
                    train_steps=2000, batch_size=32,
                    save_interval=100, verbose = 10,
                    samples=16):

        self.G=self.generator()
        self.DM=self.discriminator_model()
        self.AM=self.adversarial_model()

        print('Training Beginning')
        
        for i in range(train_steps):
            # First train the discriminator with correct labels
            # Randomly select batch from training samples
            images_real = x_train[np.random.randint(0,
                x_train.shape[0], size=batch_size), :, :, :]
            # Generate fake images from generator
            noise = np.random.normal(loc=0., scale=1., size=[batch_size, self.input_dim])
            images_fake = self.G.predict(noise)
            # Train true and false sets with correct labels and train discriminator
            #d_loss=np.zeros(train_rate[0])
            for k in range(train_rate[0]):
                y = np.random.binomial(1,.9,size=[batch_size, 1])
                d_loss_real = self.DM.train_on_batch(images_real, y)
                y =np.random.binomial(1,.1,size=[batch_size, 1])
                d_loss_fake = self.DM.train_on_batch(images_fake,y)
                #d_loss += np.add(d_loss_fake,d_loss_real)/2/train_rate[0]
                d_loss = np.add(d_loss_fake,d_loss_real)/2
            # Now train the adversarial network
            # Create new fake images labels as if they are from the training set
            #a_loss=np.zeros(train_rate[1])
            for j in range(train_rate[1]):
            	y = np.ones([batch_size, 1])
            	#a_loss += np.array(self.adversarial.train_on_batch(noise, y))/train_rate[1]
            	a_loss = np.array(self.AM.train_on_batch(noise, y))
            # Generate log messages
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            if i%verbose==0:
                print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.save_state()
                    fn = filename+"_%d.png" % (i+1)
                    self.plot_images(images_real, images_fake, filename=fn, samples=samples)

    def plot_images(self, real, fake, filename=None, samples=16):

        images = np.concatenate((fake[:int(samples/2)],real[:int(samples/2)]))
        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(int(samples**.5), int(samples**.5), i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='viridis')
            plt.axis('off')
        plt.tight_layout()
        if filename!=None:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()



    
            
    
