import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.sparse.linalg import eigsh

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer

import healpy as hp

from pygsp.graphs import SphereHealpix
from pygsp import filters



class GraphConv(Layer):
    """
    Abstract graph convolutional layer.
    """
    def __init__(self,
                 K, 
                 Fout,
                 L=None,
                 poly_type='chebyshev',
                 polyK=None,
                 activation=None, 
                 use_bias=False, 
                 use_bn=False, 
                 kernel_initializer=None, 
                 kernel_regularizer=None,
                 **kwargs):
        """
        Initializes the graph convolutional layer, assuming the input has dimension (N, M, Fin)    
        :param K: highest order polynomial to use
        :param Fout: number of features output by the layer
        :param L: the graph laplacian, in tf.sparse format.
        :param poly_type: polynomials to use. 'chebyshev', 'legendre' or 'mono'
        :param polyK: Optional. tf.sparse_tensor of shape [K*M, M] where M is the number of nodes. 
                      given the graph laplacian L, polyK = concatenate([P_0(L), P_1(L), P_2(L), ..., P_K(L)]) 
                      where each P_k are polynomials (Chebyshev, Legendre, Monomial, etc.) to use to 
                      approximate the graph fourier transform. If supplied, will use P(L) instead of recursively
                      calculating P(L).input. One of L or polyK is needed
        :param activation: activation function to apply after the convolution (and batch norm and bias)
        :param use_bias: whether to use bias or not.
        :param use_bn: whether to use batch normalization after the convolution
        :param kernel_initializer: the initializer to use for kernel initialization
        :param kernel_regularizer: regularization function to apply to kernel weights
        :param kwargs: additional kwargs passed to add_weight
        """
        super(GraphConv, self).__init__(**kwargs)
        self.K = K
        self.Fout = Fout
        self.L = L
        if self.L:
            if poly_type not in ['chebyshev', 'legendre', 'mono']:
                raise NotImplementedError(f"The requested polynomial type {poly_type} is not supported."
                                          f"Choose from chebyshev, legendre or mono.")
        self.poly_type = poly_type
        self.polyK = polyK
        if self.L is None:
            if self.polyK is None:
                raise NotImplementedError(f"Either the graph laplacian L or polyK is needed."
                                          f"If L is supplied, please choose the polynomial type."
                                          f"poly_type is either 'chebyshev', 'legendre' or 'mono'.")
        if activation is None or callable(activation):
            self.activation = activation
        elif hasattr(tf.keras.activations, activation):
            self.activation = getattr(tf.keras.activations, activation)
        else:
            raise ValueError(f"Could not find activation <{activation}> in tf.keras.activations...")        
        
        self.use_bias = use_bias
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = tf.keras.layers.BatchNormalization(axis=-1, 
                                                         momentum=0.9, 
                                                         epsilon=1e-5, 
                                                         center=False, 
                                                         scale=False)
        
        if kernel_initializer is None or callable(kernel_initializer):
            self.kernel_initializer = kernel_initializer
        elif hasattr(tf.keras.initializers, kernel_initializer):
            self.kernel_initializer = getattr(tf.keras.initializers, kernel_initializer)
        else:
            raise ValueError(f"Could not find activation <{kernel_initializer}> in tf.keras.activations...")   
        self.kernel_regularizer = kernel_regularizer
        self.kwargs = kwargs

    def build(self, input_shape):
        """
        Build the weights of the layer.
        """
        Fin = int(input_shape[-1]) #number of batches need to be defined.
        self.kernel = self.add_weight("kernel",
                                      shape=[Fin*self.K, self.Fout],
                                      trainable=True,
                                      initializer=self.kernel_initializer, 
                                      regularizer=self.kernel_regularizer,
                                      **self.kwargs)
        if self.use_bias:
            self.bias = self.add_weight("bias", 
                                        shape=[1, 1, self.Fout],
                                        trainable=True)            
        if self.polyK is not None:
            if tf.keras.backend.floatx() == 'float16':
                self.polyK = tf.cast(self.polyK, tf.float16)
            if tf.keras.backend.floatx() == 'float32':
                self.polyK = tf.cast(self.polyK, tf.float32)
            if tf.keras.backend.floatx() == 'float64':
                self.polyK = tf.cast(self.polyK, tf.float64)
        if self.L is not None:
            if tf.keras.backend.floatx() == 'float16':
                self.L = tf.cast(self.L, tf.float16)
            if tf.keras.backend.floatx() == 'float32':
                self.L = tf.cast(self.L, tf.float32)
            if tf.keras.backend.floatx() == 'float64':
                self.L = tf.cast(self.L, tf.float64)            
            
    def call(self, input_tensor, training=True, *args, **kwargs):
        """
        Calls the layer on a input tensor
        :param input_tensor: input of the layer shape (batch, nodes, channels)
        :param args: further arguments
        :param training: wheter we are training or not
        :param kwargs: further keyword arguments
        :return: the output of the layer        
        """
        #get input shape
        N, M, Fin = input_tensor.get_shape()
        M, Fin = int(M), int(Fin)
        
        #polyK shape: K*M x M
        x0 = tf.transpose(input_tensor, perm=[1, 2, 0]) # M x Fin x N
        x0 = tf.reshape(x0, [M, -1]) #M x Fin*N
        if self.polyK:
            x = tf.sparse.sparse_dense_matmul(self.polyK, x0) #K*M x Fin*N
        elif self.L:
            stack = [x0]
            if self.poly_type == 'mono':
                for k in range(1, self.K):
                    x1 = tf.sparse.sparse_dense_matmul(self.L, x0) #P_1(L) = L
                    stack.append(x1)
                    x0 = x1
            else:    
                if self.K > 1:
                    x1 = tf.sparse.sparse_dense_matmul(self.L, x0)
                    stack.append(x1)
                for k in range(2, self.K):
                    if self.poly_type == 'chebyshev':
                        x2 = 2*tf.sparse.sparse_dense_matmul(self.L, x1) - x0
                    if self.poly_type == 'legendre':  
                        x2 = ((2*k-1)/k)*tf.sparse.sparse_dense_matmul(self.L, x1) - ((k-1)/k)*x0
                    stack.append(x2)
                    x0, x1 = x1, x2
            x = tf.stack(stack, axis=0)            
        x = tf.reshape(x, [self.K, M, Fin, -1])  # K x M x Fin x N
        #kernel shape: Fin*K x Fout
        x = tf.transpose(x, perm=[3, 1, 2, 0])  # N x M x Fin x K
        x = tf.reshape(x, [-1, Fin * (self.K)])  # N*M x Fin*K
        x = tf.matmul(x, self.kernel)  # N*M x Fout
        x = tf.reshape(x, [-1, M, self.Fout])  # N x M x Fout

        if self.use_bias:
            x = tf.add(x, self.bias)

        if self.activation is not None:
            x = self.activation(x)
        
        if self.use_bn:
            x = self.bn(x, training=training)
        
        return x

class GraphDepthwiseConv(Layer):
    """
    Abstract depthwise graph convolutional layer.
    """
    def __init__(self, 
                 K, 
                 depth_multiplier=1,
                 L=None,
                 poly_type='chebyshev',
                 polyK=None,
                 activation=None, 
                 use_bias=False, 
                 use_bn=False, 
                 kernel_initializer=None, 
                 kernel_regularizer=None, 
                 **kwargs):
        """
        Initializes the graph convolutional layer, assuming the input has dimension (N, M, Fin)    
        :param K: highest order polynomial to use
        :param polyK: tf.sparse_tensor of shape [K*M, M] where M is the number of nodes. 
                      given the graph laplacian L, polyK = concatenate([P_0(L), P_1(L), P_2(L), ..., P_K(L)]) where
                      each P_k are polynomials (Chebyshev, Legendre, Monomial, etc.) to use to approximate the graph fourier 
                      transform. 
        :param depthwise_multiplier: number of output features per input feature
        :param activation: activation function to apply after the convolution (and batch norm and bias)
        :param use_bias: whether to use bias or not.
        :param use_bn: whether to use batch normalization after the convolution
        :param kernel_initializer: the initializer to use for kernel initialization
        :param kernel_regularizer: regularization function to apply to kernel weights
        :param kwargs: additional kwargs passed to add_weight
        """
        super(GraphDepthwiseConv, self).__init__(**kwargs)
        self.K = K
        self.depth_multiplier = depth_multiplier
        self.L = L
        if self.L:
            if poly_type not in ['chebyshev', 'legendre', 'mono']:
                raise NotImplementedError(f"The requested polynomial type {poly_type} is not supported."
                                          f"Choose from chebyshev, legendre or mono.")
        self.poly_type = poly_type
        self.polyK = polyK
        if self.L is None:
            if self.polyK is None:
                raise NotImplementedError(f"Either the graph laplacian L or polyK is needed."
                                          f"If L is supplied, please choose the polynomial type."
                                          f"poly_type is either 'chebyshev', 'legendre' or 'mono'.")        
        if activation is None or callable(activation):
            self.activation = activation
        elif hasattr(tf.keras.activations, activation):
            self.activation = getattr(tf.keras.activations, activation)
        else:
            raise ValueError(f"Could not find activation <{activation}> in tf.keras.activations...")        

        self.use_bias = use_bias
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = tf.keras.layers.BatchNormalization(axis=-1, 
                                                         momentum=0.9, 
                                                         epsilon=1e-5, 
                                                         center=False, 
                                                         scale=False)
        
        
        if kernel_initializer is None or callable(kernel_initializer):
            self.initializer = tf.keras.initializers.HeNormal()
        elif hasattr(tf.keras.initializers, kernel_initializer):
            self.initializer = getattr(tf.keras.initializers, kernel_initializer)
        else:
            raise ValueError(f"Could not find initializer <{kernel_initializer}> in tf.keras.initializers...")
        
        self.regularizer = kernel_regularizer
        self.kwargs = kwargs

    def build(self, input_shape):
        """
        Build the weights of the layer.
        """
        Fin = int(input_shape[-1]) #number of batches need to be defined.
        self.kernel = self.add_weight("kernel",
                                      shape=[Fin, self.depth_multiplier, self.K],
                                      trainable=True,
                                      initializer=self.initializer, 
                                      regularizer=self.regularizer,
                                      **self.kwargs)
        if self.use_bias:
            self.bias = self.add_weight("bias", 
                                        shape=[1, 1, Fin*self.depth_multiplier],
                                        trainable=True)            
        
        if self.polyK is not None:
            if tf.keras.backend.floatx() == 'float16':
                self.polyK = tf.cast(self.polyK, tf.float16)
            if tf.keras.backend.floatx() == 'float32':
                self.polyK = tf.cast(self.polyK, tf.float32)
            if tf.keras.backend.floatx() == 'float64':
                self.polyK = tf.cast(self.polyK, tf.float64)
        if self.L is not None:
            if tf.keras.backend.floatx() == 'float16':
                self.L = tf.cast(self.L, tf.float16)
            if tf.keras.backend.floatx() == 'float32':
                self.L = tf.cast(self.L, tf.float32)
            if tf.keras.backend.floatx() == 'float64':
                self.L = tf.cast(self.L, tf.float64)  
            
    def call(self, input_tensor, training=True, *args, **kwargs):
        """
        Calls the layer on a input tensor
        :param input_tensor: input of the layer shape (batch, nodes, channels)
        :param args: further arguments
        :param training: wheter we are training or not
        :param kwargs: further keyword arguments
        :return: the output of the layer
        """

        N, M, Fin = input_tensor.get_shape()
        M, Fin = int(M), int(Fin)
        #polyK shape: K*M x M
        x0 = tf.transpose(input_tensor, perm=[1, 0, 2]) # M x N x Fin
        x0 = tf.reshape(x0, [M, -1]) #M x N*Fin
        if self.polyK:
            x = tf.sparse.sparse_dense_matmul(self.polyK, x0) #K*M x N*Fin

        elif self.L:
            stack = [x0]
            if self.poly_type == 'mono':
                for k in range(1, self.K):
                    x1 = tf.sparse.sparse_dense_matmul(self.L, x0) #P_1(L) = L
                    stack.append(x1)
                    x0 = x1
            else:    
                if self.K > 1:
                    x1 = tf.sparse.sparse_dense_matmul(self.L, x0)
                    stack.append(x1)
                for k in range(2, self.K):
                    if self.poly_type == 'chebyshev':
                        x2 = 2*tf.sparse.sparse_dense_matmul(self.L, x1) - x0
                    if self.poly_type == 'legendre':  
                        x2 = ((2*k-1)/k)*tf.sparse.sparse_dense_matmul(self.L, x1) - ((k-1)/k)*x0
                    stack.append(x2)
                    x0, x1 = x1, x2
            x = tf.stack(stack, axis=0) #K x M x N*Fin
        x = tf.reshape(x, [self.K, M, -1, Fin])  # K x M x N x Fin
        x = tf.transpose(x, [3, 0, 2, 1]) #Fin x K x N x M
        x = tf.reshape(x, [Fin, self.K, -1]) #Fin x K x N*M
        #depthwise kernel shape: Fin x depth_multiplier x K
        x = tf.matmul(self.kernel, x) #Fin x depth_multiplier x N*M
        x = tf.transpose(x, [2, 0, 1]) #N*M x Fin x depth_multiplier
        x = tf.reshape(x, [-1, M, Fin*self.depth_multiplier])

        if self.use_bias:
            x = tf.add(x, self.bias)

        if self.activation is not None:
            x = self.activation(x)
        
        if self.use_bn:
            x = self.bn(x, training=training)

        return x

class GraphSeparableConv(Layer):
    """
    Abstract depthwise separable graph convolutional layer.
    """
    def __init__(self, 
                 K, 
                 Fout, 
                 depth_multiplier=1,
                 L=None,
                 poly_type='chebyshev',
                 polyK=None,
                 activation=None, 
                 use_bias=False, 
                 use_bn=False, 
                 pointwise_initializer=None, 
                 depthwise_initializer=None, 
                 pointwise_regularizer=None, 
                 depthwise_regularizer=None, 
                 **kwargs):
        """
        Initializes the graph convolutional layer, assuming the input has dimension (N, M, Fin)    
        :param K: highest order polynomial to use
        :param polyK: tf.sparse_tensor of shape [K*M, M] where M is the number of nodes. 
                      given the graph laplacian L, polyK = concatenate([P_0(L), P_1(L), P_2(L), ..., P_K(L)]) where
                      each P_k are polynomials (Chebyshev, Legendre, Monomial, etc.) to use to approximate the graph fourier 
                      transform. 
        :param depthwise_multiplier: number of output features per input feature
        :param activation: activation function to apply after the convolution (and batch norm and bias)
        :param use_bias: whether to use bias or not.
        :param use_bn: whether to use batch normalization after the convolution
        :param pointwise_initializer: the initializer to use for the pointwise kernel
        :param depthwise_initializer: the initializer to use for the depthwise kernel
        :param pointwise_regularizer: regularization function to apply on the pointwise kernel weights
        :param depthwise_regularizer: regularization function to apply on the depthwise kernel weights
        :param kwargs: additional kwargs passed to add_weight
        """
        super(GraphSeparableConv, self).__init__(**kwargs)
        self.K = K
        self.Fout = Fout
        self.depth_multiplier = depth_multiplier
        self.L = L
        if self.L:
            if poly_type not in ['chebyshev', 'legendre', 'mono']:
                raise NotImplementedError(f"The requested polynomial type {poly_type} is not supported."
                                      f"Choose from chebyshev, legendre or mono.")
        self.poly_type = poly_type
        self.polyK = polyK
        if self.L is None:
            if self.polyK is None:
                raise NotImplementedError(f"Either the graph laplacian L or polyK is needed."
                                          f"If L is supplied, please choose the polynomial type."
                                          f"poly_type is either 'chebyshev', 'legendre' or 'mono'.")        

        if activation is None or callable(activation):
            self.activation = activation
        elif hasattr(tf.keras.activations, activation):
            self.activation = getattr(tf.keras.activations, activation)
        else:
            raise ValueError(f"Could not find activation <{activation}> in tf.keras.activations...")        

        self.use_bias = use_bias
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = tf.keras.layers.BatchNormalization(axis=-1, 
                                                         momentum=0.9, 
                                                         epsilon=1e-5, 
                                                         center=False, 
                                                         scale=False)        

        if pointwise_initializer is None or callable(pointwise_initializer):
            self.pointwise_initializer = tf.keras.initializers.HeNormal()
        elif hasattr(tf.keras.initializers, pointwise_initializer):
            self.pointwise_initializer = getattr(tf.keras.initializers, pointwise_initializer)
        else:
            raise ValueError(f"Could not find initializer <{pointwise_initializer}> in tf.keras.initializers...")
        
        if depthwise_initializer is None or callable(depthwise_initializer):
            self.depthwise_initializer = tf.keras.initializers.HeNormal()
        elif hasattr(tf.keras.initializers, depthwise_initializer):
            self.depthwise_initializer = getattr(tf.keras.initializers, depthwise_initializer)
        else:
            raise ValueError(f"Could not find initializer <{depthwise_initializer}> in tf.keras.initializers...")

        self.pointwise_regularizer = pointwise_regularizer
        self.depthwise_regularizer = depthwise_regularizer
        self.kwargs = kwargs

    def build(self, input_shape):
        """
        Build the weights of the layer.
        """
        Fin = int(input_shape[-1]) #number of batches need to be defined.
                
        self.pkernel = self.add_weight("pointwise_kernel", 
                                       shape=[self.Fout, Fin*self.depth_multiplier],
                                       trainable=True,
                                       initializer=self.pointwise_initializer, 
                                       regularizer=self.pointwise_regularizer,
                                       **self.kwargs)
        
        self.dkernel = self.add_weight("depthwise_kernel",
                                       shape=[Fin, self.depth_multiplier, self.K],
                                       trainable=True,
                                       initializer=self.depthwise_initializer, 
                                       regularizer=self.depthwise_regularizer,
                                       **self.kwargs)
        
        if self.use_bias:
            self.bias = self.add_weight("bias", 
                                        shape=[1, 1, Fout],
                                        trainable=True)            
        
        if self.polyK is not None:
            if tf.keras.backend.floatx() == 'float16':
                self.polyK = tf.cast(self.polyK, tf.float16)
            if tf.keras.backend.floatx() == 'float32':
                self.polyK = tf.cast(self.polyK, tf.float32)
            if tf.keras.backend.floatx() == 'float64':
                self.polyK = tf.cast(self.polyK, tf.float64)
        if self.L is not None:
            if tf.keras.backend.floatx() == 'float16':
                self.L = tf.cast(self.L, tf.float16)
            if tf.keras.backend.floatx() == 'float32':
                self.L = tf.cast(self.L, tf.float32)
            if tf.keras.backend.floatx() == 'float64':
                self.L = tf.cast(self.L, tf.float64)  
            
    def call(self, input_tensor, training=True, *args, **kwargs):
        """
        Calls the layer on a input tensor
        :param input_tensor: input of the layer shape (batch, nodes, channels)
        :param args: further arguments
        :param training: wheter we are training or not
        :param kwargs: further keyword arguments
        :return: the output of the layer
        """

        N, M, Fin = input_tensor.get_shape()
        M, Fin = int(M), int(Fin)
        #depthwise step
        #polyK shape: K*M x M
        x0 = tf.transpose(input_tensor, perm=[1, 0, 2]) # M x N x Fin
        x0 = tf.reshape(x0, [M, -1]) #M x N*Fin
        if self.polyK:
            x = tf.sparse.sparse_dense_matmul(self.polyK, x) #K*M x N*Fin
        elif self.L:
            stack = [x0]
            if self.poly_type == 'mono':
                for k in range(1, self.K):
                    x1 = tf.sparse.sparse_dense_matmul(self.L, x0) #P_1(L) = L
                    stack.append(x1)
                    x0 = x1
            else:    
                if self.K > 1:
                    x1 = tf.sparse.sparse_dense_matmul(self.L, x0)
                    stack.append(x1)
                for k in range(2, self.K):
                    if self.poly_type == 'chebyshev':
                        x2 = 2*tf.sparse.sparse_dense_matmul(self.L, x1) - x0
                    if self.poly_type == 'legendre':  
                        x2 = ((2*k-1)/k)*tf.sparse.sparse_dense_matmul(self.L, x1) - ((k-1)/k)*x0
                    stack.append(x2)
                    x0, x1 = x1, x2
            x = tf.stack(stack, axis=0) #K x M x N*Fin
       
        x = tf.reshape(x, [self.K, M, -1, Fin])  # K x M x N x Fin
        x = tf.transpose(x, [3, 0, 2, 1]) #Fin x K x N x M
        x = tf.reshape(x, [Fin, self.K, -1]) #Fin x K x N*M
        #depthwise kernel shape: Fin x depth_multiplier x K
        x = tf.matmul(self.dkernel, x) #Fin x depth_multiplier x N*M
        
        #pointwise step
                
        x = tf.reshape(x, [Fin*self.depth_multiplier, -1]) # Fin*depth_multiplier x N*M
        #pointwise_kernel shape: Fout x Fin*depth_multiplier
        x = tf.matmul(self.pkernel, x) # Fout x N*M
        x = tf.transpose(x) #N*M x Fout
        x = tf.reshape(x, [-1,M,self.Fout]) # N x M x Fout

        if self.use_bias:
            x = tf.add(x, self.bias)

        if self.activation is not None:
            x = self.activation(x)
        
        if self.use_bn:
            x = self.bn(x, training=training)

        return x
