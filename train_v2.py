import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.sparse.linalg import eigsh

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer

from src import graphconv
from src import gcw
from src import processtools as pt
from src import healpyfunctional as hpf

import healpy as hp

from pygsp.graphs import SphereHealpix
from pygsp import filters

from tqdm import tqdm

print(tf.config.get_visible_devices())



class GCNHealpy_uNetlike(Model, gcw.GCW):
    """
    Graph convolutional NN models for the healpy pixelization scheme. 
    Precalculates the polynomial approximation of the graph laplacian for graph convolutional layers.
    """
    def __init__(self,
                 nside,
                 indices,
                 channels=1,
                 use_polyK=False,
                 verbose=True):
        """
        :param nside: nside of the input maps
        :param indices: indices of the input maps
        :param channels: number of input channels
        :param use_polyK: Bool. Optional. If True, will precalculate P(L) and use P(L) in graph convolution
                          layers. Might lead to performance gains.
        """
        super(GCNHealpy_uNetlike, self).__init__(name='')
        self.nside = nside
        self.indices = indices
        self.channels = channels
        self.use_polyK = use_polyK
        self.verbose = verbose
        self.polydict = {}
        self.Ldict = {}
    
    def l2(self, weight_decay):
        return tf.keras.regularizers.L2(l2=weight_decay)
        
    def model(self, weight_decay, sdrate, include_top=True, num_classes=3):
        """
        :param weight_decay: l2 regularization penalty to apply on the convolution kernels
        :param sdrate: spatial dropout rate to apply after the convolution layers
        :param include_top: if true, will include the globalavereagepooling and densely connected layers
        :param num_classes: number of outputs of the final densely connected layer.
        """
        inputs = tf.keras.layers.Input(shape=(len(self.indices), self.channels), name="input_maps")
        x1 = self.Conv(nside=self.nside, indices=self.indices, n_neighbors=8, poly_type='chebyshev',
                       K=4, Fout=16, activation='relu', use_bn=True, 
                       kernel_initializer='he_normal', kernel_regularizer=self.l2(weight_decay))(inputs)
        x1 = self.Conv(nside=self.nside, indices=self.indices, n_neighbors=8, poly_type='chebyshev',
                       K=4, Fout=16, activation='relu', use_bn=True, 
                       kernel_initializer='he_normal', kernel_regularizer=self.l2(weight_decay))(x1)
        x1 = hpf.HealpyPseudoConv(p=1, Fout=16, activation='relu', initializer='he_normal',
                                  kernel_regularizer=self.l2(weight_decay), nside=self.nside, 
                                  indices=self.indices)(x1)
        x1 = tf.keras.layers.SpatialDropout1D(sdrate)(x1)        
        x1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, 
                                                epsilon=0.001, center=False, 
                                                scale=False)(x1) 

        
        x2 = self.Conv(nside=self.nside, indices=self.indices, n_neighbors=20, poly_type='chebyshev',
                       K=8, Fout=32, activation='relu', use_bn=True, 
                       kernel_initializer='he_normal', kernel_regularizer=self.l2(weight_decay))(inputs)
        x2 = hpf.HealpyPseudoConv(p=1, Fout=32, activation='relu', initializer='he_normal',
                                  kernel_regularizer=self.l2(weight_decay), nside=self.nside, 
                                  indices=self.indices)(x2)
        x2 = tf.keras.layers.SpatialDropout1D(sdrate)(x2)
        x2 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, 
                                                epsilon=0.001, center=False, 
                                                scale=False)(x2)
        
        x3 = self.Conv(nside=self.nside, indices=self.indices, n_neighbors=20, poly_type='chebyshev',
                       K=12, Fout=16, activation='relu', use_bn=True, 
                       kernel_initializer='he_normal', kernel_regularizer=self.l2(weight_decay))(inputs)
        x3 = hpf.HealpyPseudoConv(p=1, Fout=16, activation='relu', initializer='he_normal',
                                  kernel_regularizer=self.l2(weight_decay), nside=self.nside, 
                                  indices=self.indices)(x3)
        x3 = tf.keras.layers.SpatialDropout1D(sdrate)(x3)
        x3 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, 
                                                epsilon=0.001, center=False, 
                                                scale=False)(x3) 

        
        nside_out1 = hpf.HealpyPseudoConv(p=1, Fout=16, activation='relu', 
                                          initializer='he_normal',
                                          kernel_regularizer=self.l2(weight_decay),
                                          nside=self.nside, indices=self.indices).nside_out
        indices_out1 = hpf.HealpyPseudoConv(p=1, Fout=16, activation='relu', 
                                            initializer='he_normal',
                                            kernel_regularizer=self.l2(weight_decay),
                                            nside=self.nside, indices=self.indices).indices_out
        
        x = tf.keras.layers.Concatenate(axis=-1)([x1,x2,x3]) #output of 'conv+P 1', nside = 64, F=64

        
        x1 = self.Conv(nside=nside_out1, indices=indices_out1, n_neighbors=20, poly_type='chebyshev',
                       K=4, Fout=32, activation='relu', use_bn=True, 
                       kernel_initializer='he_normal', kernel_regularizer=self.l2(weight_decay))(x)
        x1 = self.Conv(nside=nside_out1, indices=indices_out1, n_neighbors=8, poly_type='chebyshev',
                       K=8, Fout=32, activation='relu', use_bn=True, 
                       kernel_initializer='he_normal', kernel_regularizer=self.l2(weight_decay))(x1)
        x1 = hpf.HealpyPseudoConv(p=1, Fout=32, activation='relu', 
                                  initializer='he_normal', kernel_regularizer=self.l2(weight_decay), 
                                  nside=nside_out1, indices=indices_out1)(x1)
        x1 = tf.keras.layers.SpatialDropout1D(sdrate)(x1)
        x1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, 
                                                epsilon=0.001, center=False, 
                                                scale=False)(x1) 

        
        x2 = self.Conv(nside=nside_out1, indices=indices_out1, n_neighbors=20, poly_type='chebyshev',
                       K=12, Fout=32, activation='relu', use_bn=True, 
                       kernel_initializer='he_normal', kernel_regularizer=self.l2(weight_decay))(x)
        x2 = hpf.HealpyPseudoConv(p=1, Fout=32, activation='relu', 
                                  initializer='he_normal', kernel_regularizer=self.l2(weight_decay), 
                                  nside=nside_out1, indices=indices_out1)(x2)
        x2 = tf.keras.layers.SpatialDropout1D(sdrate)(x2)
        x2 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, 
                                                epsilon=0.001, center=False, 
                                                scale=False)(x2) 

        
        x3 = tf.keras.layers.Concatenate(axis=-1)([x1,x2])
        
        xres = hpf.HealpyPool(nside=nside_out1, indices = indices_out1, p=1, pool_type='AVG')(x)
        xres = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, 
                                                  epsilon=0.001, center=False, 
                                                  scale=False)(xres)        

        nside_out2 = hpf.HealpyPool(nside=nside_out1, indices = indices_out1, 
                                    p=1, pool_type='AVG').nside_out
        indices_out2 = hpf.HealpyPool(nside=nside_out1, indices = indices_out1, 
                                      p=1, pool_type='AVG').indices_out
        
        x = tf.keras.layers.Add()([x3,xres]) #output of 'conv+P 2', nside=32, F=128
        
        x1 = self.Conv(nside=nside_out2, indices=indices_out2, n_neighbors=8, poly_type='chebyshev',
                      K=8, Fout=128, activation='relu', use_bn=True, 
                      kernel_initializer='he_normal', kernel_regularizer=self.l2(weight_decay))(x)
        x = self.Conv(nside=nside_out2, indices=indices_out2, n_neighbors=20, poly_type='chebyshev',
                      K=12, Fout=128, activation='relu', use_bn=True, 
                      kernel_initializer='he_normal', kernel_regularizer=self.l2(weight_decay))(x1)
        
        x1 = self.Conv(nside=nside_out2, indices=indices_out2, n_neighbors=8, poly_type='chebyshev',
                      K=4, Fout=128, activation='relu', use_bn=True, 
                      kernel_initializer='he_normal', kernel_regularizer=self.l2(weight_decay))(x)
        x1 = self.Conv(nside=nside_out2, indices=indices_out2, n_neighbors=20, poly_type='chebyshev',
                      K=8, Fout=128, activation='relu', use_bn=True, 
                      kernel_initializer='he_normal', kernel_regularizer=self.l2(weight_decay))(x1)
        
        x = tf.keras.layers.Add()([x,x1]) #output of 'conv 3', nside = 32, F=128
        
        xup = hpf.HealpyPseudoConv_Transpose(nside=nside_out2, indices=indices_out2, 
                                             p=1, Fout=64, 
                                             kernel_initializer='he_normal')(x)
        xup = tf.keras.layers.ReLU()(xup)
        xup = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, 
                                                 epsilon=0.001, center=False, 
                                                 scale=False)(xup)
        
        nside_up1 = hpf.HealpyPseudoConv_Transpose(nside=nside_out2, indices=indices_out2, 
                                                   p=1, Fout=64, 
                                                   kernel_initializer='he_normal').nside_out
        indices_up1 = hpf.HealpyPseudoConv_Transpose(nside=nside_out2, indices=indices_out2, 
                                                   p=1, Fout=64, 
                                                   kernel_initializer='he_normal').indices_out
        xup = self.Conv(nside=nside_up1, indices=indices_up1, n_neighbors=20, poly_type='chebyshev',
                        K=8, Fout=64, activation='relu', use_bn=True, 
                        kernel_initializer='he_normal', kernel_regularizer=self.l2(weight_decay))(xup)
        xup = hpf.HealpyPseudoConv(p=1, Fout=128, activation='relu', 
                                   initializer='he_normal', kernel_regularizer=self.l2(weight_decay), 
                                   nside=nside_out1, indices=indices_out1)(xup)
        xup = tf.keras.layers.SpatialDropout1D(sdrate)(xup)
        xup = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, 
                                                 epsilon=0.001, center=False, 
                                                 scale=False)(xup) #output of 'u/d 1', nside=32, F=128
        
        
        for i in range(2):
            x1 = self.SeparableConv(nside=nside_out2,
                                    indices=indices_out2,
                                    n_neighbors=8,
                                    poly_type='chebyshev',
                                    K=6,
                                    Fout=128,
                                    depth_multiplier=2,
                                    pointwise_initializer='he_normal',
                                    depthwise_initializer='he_normal',
                                    pointwise_regularizer=self.l2(weight_decay),
                                    depthwise_regularizer=self.l2(weight_decay))(x)
            x1 = tf.keras.layers.ReLU()(x1)
            x1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, 
                                                    epsilon=0.001, center=False, 
                                                    scale=False)(x1)

            
            x1 = self.SeparableConv(nside=nside_out2,
                                    indices=indices_out2,
                                    n_neighbors=20,
                                    poly_type='chebyshev',
                                    K=10,
                                    Fout=128,
                                    depth_multiplier=2,
                                    pointwise_initializer='he_normal',
                                    depthwise_initializer='he_normal',
                                    pointwise_regularizer=self.l2(weight_decay),
                                    depthwise_regularizer=self.l2(weight_decay))(x1)
            x1 = tf.keras.layers.ReLU()(x1)
            x1 = tf.keras.layers.SpatialDropout1D(sdrate)(x1)            
            x1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, 
                                                    epsilon=0.001, center=False, 
                                                    scale=False)(x1)         
        
            x = tf.keras.layers.Add()([x, x1]) #output of 'sepconv 1', nside=32, F=128
            
        x = tf.keras.layers.Add()([x, xup])
        
        x = self.DepthwiseConv(nside=nside_out2, indices=indices_out2, n_neighbors=20, poly_type='chebyshev',
                      K=8, depth_multiplier=2, activation='relu', use_bn=True, 
                      kernel_initializer='he_normal', kernel_regularizer=self.l2(weight_decay))(x)
        x = hpf.HealpyPool(nside=nside_out2, indices=indices_out2, p=1, pool_type='AVG')(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, 
                                               epsilon=0.001, center=False, 
                                               scale=False)(x)
        x = tf.keras.layers.SpatialDropout1D(sdrate)(x) #output of 'dconv + P 1', nside=16, F=256
        
        nside_out3 = hpf.HealpyPool(nside=nside_out2, indices=indices_out2, p=1, pool_type='AVG').nside_out
        indices_out3 = hpf.HealpyPool(nside=nside_out2, indices=indices_out2, p=1, pool_type='AVG').indices_out
        
        xup2 = hpf.HealpyPseudoConv_Transpose(nside=nside_out3, indices=indices_out3, 
                                             p=1, Fout=128, 
                                             kernel_initializer='he_normal')(x)
        xup2 = tf.keras.layers.ReLU()(xup2)
        xup2 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, 
                                                 epsilon=0.001, center=False, 
                                                 scale=False)(xup2)
        #xup2 = tf.keras.layers.Add()([xup, xup2])
        
        nside_up2 = hpf.HealpyPseudoConv_Transpose(nside=nside_out3, indices=indices_out3, 
                                                   p=1, Fout=128, 
                                                   kernel_initializer='he_normal').nside_out
        indices_up2 = hpf.HealpyPseudoConv_Transpose(nside=nside_out3, indices=indices_out3, 
                                                   p=1, Fout=128, 
                                                   kernel_initializer='he_normal').indices_out
        xup2 = self.Conv(nside=nside_up2, indices=indices_up2, n_neighbors=20, poly_type='chebyshev',
                        K=8, Fout=128, activation='relu', use_bn=True, 
                        kernel_initializer='he_normal', kernel_regularizer=self.l2(weight_decay))(xup2)
        xup2 = hpf.HealpyPseudoConv(p=1, Fout=256, activation='relu', 
                                   initializer='he_normal', kernel_regularizer=self.l2(weight_decay), 
                                   nside=nside_up2, indices=indices_up2)(xup2)
        xup2 = tf.keras.layers.SpatialDropout1D(sdrate)(xup2)
        xup2 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, 
                                                 epsilon=0.001, center=False, 
                                                 scale=False)(xup2) #output of 'u/d 2', nside=16, F=256
        
        for i in range(2):
            x1 = self.SeparableConv(nside=nside_out3,
                                    indices=indices_out3,
                                    n_neighbors=20,
                                    poly_type='chebyshev',
                                    K=4,
                                    Fout=256,
                                    depth_multiplier=1,
                                    pointwise_initializer='he_normal',
                                    depthwise_initializer='he_normal',
                                    pointwise_regularizer=self.l2(weight_decay),
                                    depthwise_regularizer=self.l2(weight_decay))(x)
            x1 = tf.keras.layers.ReLU()(x1)            
            x1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, 
                                                    epsilon=0.001, center=False, 
                                                    scale=False)(x1)

            
            x1 = self.SeparableConv(nside=nside_out3,
                                    indices=indices_out3,
                                    n_neighbors=8,
                                    poly_type='chebyshev',
                                    K=8,
                                    Fout=256,
                                    depth_multiplier=1,
                                    pointwise_initializer='he_normal',
                                    depthwise_initializer='he_normal',
                                    pointwise_regularizer=self.l2(weight_decay),
                                    depthwise_regularizer=self.l2(weight_decay))(x1)
            x1 = tf.keras.layers.ReLU()(x1)   
            x1 = tf.keras.layers.SpatialDropout1D(sdrate)(x1)            
            x1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, 
                                                    epsilon=0.001, center=False, 
                                                    scale=False)(x1)
         
        
            x = tf.keras.layers.Add()([x, x1])  #output of 'sepconv 2', nside=16, F=256
            
        x = tf.keras.layers.Add()([x, xup2])
        
        x = self.Conv(nside=nside_out3, indices=indices_out3, n_neighbors=8, poly_type='chebyshev',
                      K=8, Fout=512, activation='relu', use_bn=True, 
                      kernel_regularizer=self.l2(weight_decay))(x)
        x = hpf.HealpyPool(nside=nside_out3, indices=indices_out3, p=1, pool_type='AVG')(x)
        x = tf.keras.layers.SpatialDropout1D(sdrate)(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, 
                                               epsilon=0.001, center=False, 
                                               scale=False)(x) #output of 'conv+P 3', nside=8, F=512
        
        nside_out4 = hpf.HealpyPool(nside=nside_out3, indices=indices_out3, p=1, pool_type='AVG').nside_out
        indices_out4 = hpf.HealpyPool(nside=nside_out3, indices=indices_out3, p=1, pool_type='AVG').indices_out
        
        xup3 = hpf.HealpyPseudoConv_Transpose(nside=nside_out4, indices=indices_out4, 
                                             p=1, Fout=256, 
                                             kernel_initializer='he_normal')(x)
        xup3 = tf.keras.layers.ReLU()(xup3)
        xup3 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, 
                                                 epsilon=0.001, center=False, 
                                                 scale=False)(xup3)
        #xup3 = tf.keras.layers.Add()([xup2, xup3])
        
        nside_up3 = hpf.HealpyPseudoConv_Transpose(nside=nside_out4, indices=indices_out4, 
                                                   p=1, Fout=256, 
                                                   kernel_initializer='he_normal').nside_out
        indices_up3 = hpf.HealpyPseudoConv_Transpose(nside=nside_out4, indices=indices_out4, 
                                                   p=1, Fout=256, 
                                                   kernel_initializer='he_normal').indices_out
        xup3 = self.Conv(nside=nside_up3, indices=indices_up3, n_neighbors=20, poly_type='chebyshev',
                        K=8, Fout=256, activation='relu', use_bn=True, 
                        kernel_initializer='he_normal', kernel_regularizer=self.l2(weight_decay))(xup3)
        xup3 = hpf.HealpyPseudoConv(p=1, Fout=512, activation='relu', 
                                   initializer='he_normal', kernel_regularizer=self.l2(weight_decay), 
                                   nside=nside_up3, indices=indices_up3)(xup3)
        xup3 = tf.keras.layers.SpatialDropout1D(sdrate)(xup3)
        xup3 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, 
                                                 epsilon=0.001, center=False, 
                                                 scale=False)(xup3) #output of 'u/d 3', nside=8, F=512        
        
        for i in range(2):
            x1 = self.SeparableConv(nside=nside_out4,
                                    indices=indices_out4,
                                    n_neighbors=8,
                                    poly_type='chebyshev',
                                    K=8,
                                    Fout=512,
                                    depth_multiplier=1,
                                    pointwise_initializer='he_normal',
                                    depthwise_initializer='he_normal',
                                    pointwise_regularizer=self.l2(weight_decay),
                                    depthwise_regularizer=self.l2(weight_decay))(x)
            x1 = tf.keras.layers.ReLU()(x1)
            x1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, 
                                                    epsilon=0.001, center=False, 
                                                    scale=False)(x1)

            
            x1 = self.SeparableConv(nside=nside_out4,
                                    indices=indices_out4,
                                    n_neighbors=8,
                                    poly_type='chebyshev',
                                    K=8,
                                    Fout=512,
                                    depth_multiplier=1,
                                    pointwise_initializer='he_normal',
                                    depthwise_initializer='he_normal',
                                    pointwise_regularizer=self.l2(weight_decay),
                                    depthwise_regularizer=self.l2(weight_decay))(x1)
            x1 = tf.keras.layers.ReLU()(x1) 
            x1 = tf.keras.layers.SpatialDropout1D(sdrate)(x1)            
            x1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, 
                                                    epsilon=0.001, center=False, 
                                                    scale=False)(x1)
           
        
            x = tf.keras.layers.Add()([x, x1])  #output of 'sep conv 3', nside=8, F=512
        
            
        x = tf.keras.layers.Add()([x, xup3])
        
        x = self.SeparableConv(nside=nside_out4,
                               indices=indices_out4,
                               n_neighbors=8,
                               poly_type='chebyshev',
                               K=8,
                               Fout=512,
                               depth_multiplier=1,
                               pointwise_initializer='he_normal',
                               depthwise_initializer='he_normal',
                               pointwise_regularizer=self.l2(weight_decay),
                               depthwise_regularizer=self.l2(weight_decay))(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, 
                                               epsilon=0.001, center=False, 
                                               scale=False)(x)

        
        
        if include_top == True:
            outputs = tf.keras.layers.GlobalAveragePooling1D()(x)
            outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(outputs)
                               
        
        return Model(inputs = inputs, outputs = outputs)
    
#load data
print('Loading data...')
a_lm_triv = np.load('data/realizations_L_infty_lmax_250_num_1000.npy').astype(np.complex128)
a_lm_torus1400 = np.load('data/realizations_L_1400_lmax_250_num_1000.npy').astype(np.complex128)
a_lm_torus2800 = np.load('data/realizations_L_2800_lmax_250_num_1000.npy').astype(np.complex128)
print('Data loading complete.')

#input indices and masking:
print('Preparing the mask and calculating relevant map indices')
nside = 128
npix = hp.nside2npix(nside=nside)
indices = np.arange(npix)
mask=hp.read_map('data/masks/COM_Mask_CMB-common-Mask-Int_2048_R3.fits')
print('Mask preparation done.')

#unmasked pixels:
unmasked_pix = pt.get_indices(mask=mask, nside=nside, target_nside=nside)
#aggresive masking:
worst_case_pix = pt.get_indices(mask=mask, nside=nside, target_nside=8)
#adaptive masking:
adaptive_case_pix = pt.get_indices(mask=mask, nside=nside, target_nside=nside//2)
print('Relevant map indices are calculated.')

print('Defining TensorFlow distribution strategy.')
strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.NcclAllReduce(num_packs=2))

#tf.keras.backend.clear_session()
#defining batch sizes
print('Creating TensorFlow datasets.')
BATCH_SIZE_PER_REPLICA = 5
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
#create datasets:
train_data, test_data, x_eval, y_eval, x_alm_train, y_train = pt.create_dataset(a_lm_torus1400, 
                                                                         a_lm_torus2800, 
                                                                         a_lm_triv, 
                                                                         relevant_pix=adaptive_case_pix,
                                                                         global_batch_size=GLOBAL_BATCH_SIZE,
                                                                         trainperc=0.8,
                                                                         evalperc=0.05,
                                                                         strategy=strategy)
print('Dataset creation completed.')

tf.keras.backend.clear_session()
print('Creating the model.')
with strategy.scope():
    GCN_v2 = GCNHealpy_uNetlike(nside=nside, 
                                indices=adaptive_case_pix, 
                                channels=1,
                                use_polyK=False)
    model = GCN_v2.model(weight_decay=1e-4, 
                         sdrate=0.05, 
                         include_top=True,
                         num_classes=3)
print('Model creation complete.')

model.summary(110)

import os
import csv

checkpoint_path = "runs_2/training_3_class/SGDopt_xception_v3_L_precalc_adaptive_mask/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

csv_logger = tf.keras.callbacks.CSVLogger('training_log.csv')
callbacklist = []


callbacks = tf.keras.callbacks.CallbackList(
    callbacklist, add_history=True, model=model)


def lr_decay(lr_init, epoch, num_batches, decay=0.998):
    steps = epoch * BUFFER_SIZE//GLOBAL_BATCH_SIZE + num_batches
    if epoch < 20:
        return lr_init
    else:
        return lr_init* (decay)**(-20*BUFFER_SIZE//GLOBAL_BATCH_SIZE)*(decay)**(steps)
    
with strategy.scope():
  # Set reduction to `none` so we can do the reduction afterwards and divide by
  # global batch size.
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3,  #1e-3 with mom = 0.8 and decay = 0.998 is very stable
                                         momentum = 0.8)     #but seems to stagnate (or run out of data)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)


def train_step(inputs):
    samples, labels = inputs
    with tf.GradientTape() as tape:
        predictions = model(samples, training=True)
        loss = compute_loss(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_accuracy.update_state(labels, predictions)
    return loss 

def test_step(inputs):
    samples, labels = inputs

    predictions = model(samples, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss.update_state(t_loss)
    test_accuracy.update_state(labels, predictions)

# `run` replicates the provided computation and runs it
# with the distributed input.
@tf.function()
def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)

@tf.function()
def distributed_test_step(dataset_inputs):
    return strategy.run(test_step, args=(dataset_inputs,))

train_loss_xception_v2 = []
train_accuracy_xception_v2 = []
test_loss_xception_v2 = []
test_accuracy_xception_v2 = []

logs = {}

BUFFER_SIZE = len(x_alm_train)


EPOCHS = 100
AUG_EPOCH = 50 #at the end of 50th, 100th etc epochs, we will rotate the training data randomly
#for aug_epoch in range(AUG_EPOCHS):
#    print(f"Augmentation epoch {aug_epoch+1}/{AUG_EPOCHS}")
#    if aug_epoch >= 1:
#        print('Creating new training dataset...')
#        train_data = pt.rotate_train_data(alm=x_alm_train, y_train=y_train, relevant_pix=adaptive_case_pix, 
#                                          global_batch_size=GLOBAL_BATCH_SIZE, strategy=strategy)
#        print('Done.')
print("------------------------------------------------------")
print(f"Starting Training, Epochs:{EPOCHS}, Augmentation Epochs:{EPOCHS//AUG_EPOCH}")
print("------------------------------------------------------")
for epoch in range(EPOCHS):
    if epoch > 0:
        if (epoch)%AUG_EPOCH == 0:
            print(f"Augmentation Epoch {(epoch+1)//AUG_EPOCH}/{EPOCHS//AUG_EPOCH} ")
            print('Rotating training dataset...')
            train_data = pt.rotate_train_data(alm=x_alm_train, y_train=y_train, 
                                                  relevant_pix=adaptive_case_pix, 
                                                  global_batch_size=GLOBAL_BATCH_SIZE, 
                                                  strategy=strategy)
            print('Rotation complete.')
    # TRAIN LOOP
    print(f"Starting with Epoch {epoch + 1}/{EPOCHS}", flush=True)
    total_loss = 0.0
    num_batches = 0    
    with tqdm(train_data, total=BUFFER_SIZE//GLOBAL_BATCH_SIZE) as pbar:
        for x in pbar:
            optimizer.learning_rate = lr_decay(5e-4, epoch, num_batches, 0.9995)
            pbar.set_description(f"Epoch {epoch +1}/{EPOCHS}", refresh=True)
            total_loss += distributed_train_step(x)
            num_batches += 1
            pbar.set_postfix({'train_loss': total_loss.numpy()/num_batches,
                              'learning_rate': optimizer.learning_rate.numpy()}, refresh=True)
            
        train_loss = total_loss / num_batches

    # TEST LOOP
        for x in test_data:
            distributed_test_step(x)



    template = ("Epoch {}/{}, Training Loss: {:.5g}, Training Accuracy: {:.5g}, Test Loss: {:.5g}, "
                "Test Accuracy: {:.5g}")
    print (template.format(epoch+1, EPOCHS,train_loss.numpy(),
                             train_accuracy.result().numpy(), test_loss.result().numpy(),
                             test_accuracy.result().numpy()))
    train_loss_xception_v2.append(train_loss)
    train_accuracy_xception_v2.append(train_accuracy.result())
    test_loss_xception_v2.append(test_loss.result())
    test_accuracy_xception_v2.append(test_accuracy.result())
    logs[f"training_loss-epoch:{epoch+1}"] = train_loss.numpy()
    logs[f"training_accuracy-epoch:{epoch+1}"] = train_accuracy.result().numpy()
    logs[f"test_loss-epoch:{epoch+1}"] = test_loss.result().numpy()
    logs[f"test_accuracy-epoch:{epoch+1}"] = test_accuracy.result().numpy()
        
    if (epoch+1) % 2 == 0:
        checkpoint.save(checkpoint_prefix)
        checkpoint_template = ("training_3_class/SGDopt_xception_v3_L_precalc_adaptive_mask/cp-{:04d}.ckpt")
        print('Creating checkpoint...')
        print('Checkpoint saved. Filename:')
        print(checkpoint_template.format(epoch+1))
    
    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()
    
    training_logs = csv.writer(open("training_logs.csv", "w"))
    for key, val in logs.items():
        training_logs.writerow([key, val])

print("Training complete. Saving weights.")        
model.save_weights("runs_2/training_3_class/SGDopt_xception_v3_L_precalc_adaptive_mask/weights.h5")

print("Creating plots.")
epochs = np.arange(1,EPOCHS+1)
#epochs = np.arange(1, len(train_loss_xception_v2))
fig, axes = plt.subplots(2, figsize=(10, 10), sharex=True)
#fig.title('Metrics')
fig.subplots_adjust(hspace=0)

axes[0].set_ylabel("Loss", fontsize=14)
#axes[0].set_xlabel("Epoch", fontsize=14)
axes[0].plot(epochs, train_loss_xception_v2, label = 'Training')
axes[0].plot(epochs, test_loss_xception_v2, '--', label = 'Validation')
axes[0].grid(visible=True, axis='both')
axes[0].set_yscale('log')
axes[0].legend()
axes[0].set_title('Training Metrics (Heavy Masking)')

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(epochs, train_accuracy_xception_v2, label = 'Training')
axes[1].plot(epochs, test_accuracy_xception_v2, '--', label = 'Validation')
axes[1].grid(visible=True, axis='both')
#axes[1].set_yscale('log')
axes[1].legend()
plt.show()
fig.savefig('runs_2/test_metrics_on_separable_convs_v1.pdf')
print("Training metric plots saved to 'runs_2/test_metrics_on_separable_convs_v1.pdf'.")
import sys
sys.exit()


