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

from .graphconv import *

class GCW():
    """
    Helper class for defining wrapper methods for graph convolution layers
    """
    def __init__(self):
        """
        Wrapper layers and helper functions defined for class inheritance.
        """
#############------------------------------Wrapper Layers------------------------------#############
    def Conv(self, 
             nside, 
             indices, 
             n_neighbors,
             poly_type,
             K, 
             Fout, 
             activation=None, 
             use_bias=False, 
             use_bn=False, 
             kernel_initializer=None, 
             kernel_regularizer=None):
        
        if self.use_polyK == True:
            if (poly_type, nside, n_neighbors, K) not in self.polydict.keys():
                if self.verbose ==True:
                    print('Calculating P(L)')
                self.polydict[(poly_type, nside, n_neighbors, K)] = self._get_poly(K, 
                                                                               nside, 
                                                                               indices, 
                                                                               n_neighbors, 
                                                                               poly_type)
            
            else:
                if self.verbose ==True:
                    print('P(L) found, no need to calculate.')

            return HealpyConv(nside=nside,
                              indices=indices,
                              n_neighbors=n_neighbors,
                              poly_type=poly_type,
                              K=K,
                              Fout=Fout,
                              activation=activation,
                              use_bias=use_bias,
                              use_bn=use_bn,
                              kernel_initializer=kernel_initializer,
                              kernel_regularizer=kernel_regularizer)._get_layer(L=None,
                                                                                polyK=self.polydict[(poly_type, 
                                                                                               nside, 
                                                                                               n_neighbors, 
                                                                                               K)])
        if self.use_polyK == False:
            if (nside, n_neighbors) not in self.Ldict.keys():
                if self.verbose ==True:
                    print('Calculating L')
                self.Ldict[(nside, n_neighbors)] = self._get_L(nside, indices, n_neighbors)
            else:
                if self.verbose ==True:
                    print('L found, no need to calculate.')
                self.Ldict[(nside, n_neighbors)] = self._get_L(nside, indices, n_neighbors)
            
            return HealpyConv(nside=nside,
                              indices=indices,
                              n_neighbors=n_neighbors,
                              poly_type=poly_type,
                              K=K,
                              Fout=Fout,
                              activation=activation,
                              use_bias=use_bias,
                              use_bn=use_bn,
                              kernel_initializer=kernel_initializer,
                              kernel_regularizer=kernel_regularizer)._get_layer(L=self.Ldict[(nside, 
                                                                                              n_neighbors)],
                                                                                polyK = None)
    
    def DepthwiseConv(self, 
                      nside, 
                      indices, 
                      n_neighbors, 
                      poly_type,
                      K,
                      depth_multiplier,
                      activation=None,
                      use_bias=False,
                      use_bn=False,
                      kernel_initializer=None,
                      kernel_regularizer=None):

        if self.use_polyK == True:
            if (poly_type, nside, n_neighbors, K) not in self.polydict.keys():
                if self.verbose ==True:
                    print('Calculating P(L)')
                self.polydict[(poly_type, nside, n_neighbors, K)] = self._get_poly(K, 
                                                                               nside, 
                                                                               indices, 
                                                                               n_neighbors, 
                                                                               poly_type)
            
            else:
                if self.verbose ==True:
                    print('P(L) found, no need to calculate.')        

        
            return HealpyDepthwiseConv(nside=nside,
                                   indices=indices,
                                   n_neighbors=n_neighbors,
                                   poly_type=poly_type,
                                   K=K,
                                   depth_multiplier=depth_multiplier,
                                   activation=activation,
                                   use_bias=use_bias,
                                   use_bn=use_bn,
                                   kernel_initializer=kernel_initializer,
                                   kernel_regularizer=kernel_regularizer)._get_layer(L=None,
                                                                                polyK=self.polydict[(poly_type, 
                                                                                               nside, 
                                                                                               n_neighbors, 
                                                                                               K)])
    
        else:
            if (nside, n_neighbors) not in self.Ldict.keys():
                if self.verbose ==True:
                    print('Calculating L')
                self.Ldict[(nside, n_neighbors)] = self._get_L(nside, indices, n_neighbors)
            else:
                if self.verbose ==True:
                    print('L found, no need to calculate.')  
                self.Ldict[(nside, n_neighbors)] = self._get_L(nside, indices, n_neighbors)
            
            return HealpyDepthwiseConv(nside=nside,
                                   indices=indices,
                                   n_neighbors=n_neighbors,
                                   poly_type=poly_type,
                                   K=K,
                                   depth_multiplier=depth_multiplier,
                                   activation=activation,
                                   use_bias=use_bias,
                                   use_bn=use_bn,
                                   kernel_initializer=kernel_initializer,
                                   kernel_regularizer=kernel_regularizer)._get_layer(L=self.Ldict[(nside, 
                                                                                              n_neighbors)],
                                                                                     polyK = None)
    
    
    
    
    def SeparableConv(self,
                      nside, 
                      indices, 
                      n_neighbors,
                      poly_type,
                      K,
                      Fout,
                      depth_multiplier,
                      activation=None, 
                      use_bias=False, 
                      use_bn=False, 
                      pointwise_initializer=None,
                      depthwise_initializer=None,
                      pointwise_regularizer=None,
                      depthwise_regularizer=None):
        
        if self.use_polyK == True:
            if (poly_type, nside, n_neighbors, K) not in self.polydict.keys():
                if self.verbose ==True:
                    print('Calculating P(L)')
                self.polydict[(poly_type, nside, n_neighbors, K)] = self._get_poly(K, 
                                                                               nside, 
                                                                               indices, 
                                                                               n_neighbors, 
                                                                               poly_type)
            
            else:
                if self.verbose ==True:
                    print('P(L) found, no need to calculate.')      

            return HealpySeparableConv(nside=nside,
                            indices=indices,
                            n_neighbors=n_neighbors,
                            poly_type=poly_type,
                            K=K,
                            Fout=Fout,
                            depth_multiplier=depth_multiplier,
                            activation=activation,
                            use_bias=use_bias,
                            use_bn=use_bn,
                            pointwise_initializer=pointwise_initializer,
                            depthwise_initializer=pointwise_regularizer,
                            pointwise_regularizer=pointwise_regularizer,
                            depthwise_regularizer=depthwise_regularizer)._get_layer(polyK=self.polydict[(
                                                                                                    poly_type, 
                                                                                                    nside, 
                                                                                                    n_neighbors, 
                                                                                                    K)])
        
        else:
            if (nside, n_neighbors) not in self.Ldict.keys():
                self.Ldict[(nside, n_neighbors)] = self._get_L(nside, indices, n_neighbors)
                if self.verbose ==True:
                    print('Calculating L')
                
            else:
                self.Ldict[(nside, n_neighbors)] = self._get_L(nside, indices, n_neighbors)
                if self.verbose ==True:
                    print('L found, no need to calculate.')          
            return HealpySeparableConv(nside=nside,
                                indices=indices,
                                n_neighbors=n_neighbors,
                                poly_type=poly_type,
                                K=K,
                                Fout=Fout,
                                depth_multiplier=depth_multiplier,
                                activation=activation,
                                use_bias=use_bias,
                                use_bn=use_bn,
                                pointwise_initializer=pointwise_initializer,
                                depthwise_initializer=pointwise_regularizer,
                                pointwise_regularizer=pointwise_regularizer,
                                depthwise_regularizer=depthwise_regularizer)._get_layer(L=self.Ldict[(nside, 
                                                                                              n_neighbors)],
                                                                                              polyK = None)           
            
    
###########---------------------------------Helper Functions-------------------------------###########    
    def _get_L(self, nside, indices, n_neighbors):
        sphere = SphereHealpix(subdivisions=nside, indexes=indices, nest=True, 
                               k=n_neighbors, lap_type='normalized')
        L = sphere.L.astype(np.float32)
        L = sparse.csr_matrix(L)
        lmax = 1.02 * eigsh(L, k=1, which='LM', return_eigenvectors=False)[0]
        L = self._rescale_L(L, lmax=lmax, scale=0.75)
        L = self._construct_tf_sparse(L)
        return L
    
    def _rescale_L(self, L, lmax=2, scale=1):
        """Rescale the Laplacian eigenvalues in [-scale,scale]."""
        M, M = L.shape
        I = sparse.identity(M, format='csr', dtype=L.dtype)
        L *= 2 * scale / lmax
        L -= I
        return L

    def _construct_tf_sparse(self, L):
        L_tf = L.tocoo()
        indices_L_tf = np.column_stack((L_tf.row, L_tf.col))
        L_tf = tf.SparseTensor(indices_L_tf, L_tf.data, L_tf.shape)
        L_tf = tf.sparse.reorder(L_tf)
        return L_tf    
    
    
    def _get_poly_dict(self, keydict):
        polydict = {}
        for key in list(keydict.keys()):
            indices_out = self._transformed_indices(nside_in=32,
                                                    nside_out=key[1],
                                                    indices_in=self.indices)
            polydict[key] = self._get_poly(K=key[-1],
                                           nside=key[1],
                                           indices=indices_out,
                                           n_neighbors=key[2],
                                           poly_type=key[0])
        return polydict

    
    def _get_poly(self, K, nside, indices, n_neighbors, poly_type):
        """
        Computes the graph laplacian operator to act on the input maps in tf.sparse format
        args:
        K: order of the polynomial to use
        nside: nside of the input maps
        indices: pixel indices of the input maps
        n_neighbors: number of neighbors to take into account when constructing the grap laplacian L
        poly_type: type of polynomial to use. 'chebyshev' for Chebyshev polynomials, 'legendre' for Legendre polynomials,
                   'mono' for monomials.

        Returns: the graph laplacian operator of order K with the desired polynomial type. in tf.sparse format.
                 will return a K*M x M sparse matrix where K is the order of polynomials and M is the number of pixels
        """
        sphere = SphereHealpix(subdivisions=nside, indexes=indices, nest=True, 
                               k=n_neighbors, lap_type='normalized')
        L = sphere.L.astype(np.float32)
        L = sparse.csr_matrix(L)
        lmax = 1.02 * eigsh(L, k=1, which='LM', return_eigenvectors=False)[0]
        L = self._rescale_L(L, lmax=lmax, scale=0.75)
        L0 = sparse.identity(len(indices), format='csr', dtype=L.dtype)
        L0_tf = self._construct_tf_sparse(L0)
        stack = [L0_tf]
        if poly_type == 'chebyshev':
            if K > 1:
                L1 = L
                L1_tf = self._construct_tf_sparse(L1)
                stack.append(L1_tf)
            for k in range(2, K):
                L2 = 2*L@L1-L0
                L2_tf = self._construct_tf_sparse(L2)
                stack.append(L2_tf)
                L0 = L1 
                L1 = L2
        if poly_type == 'legendre':
            if K > 1:
                L1 = L
                L1_tf = self._construct_tf_sparse(L1)
                stack.append(L1_tf)
            for k in range(2, K):
                L2 = ((2*k-1)/k)*L@L1-((k-1)/k)*L0
                L2_tf = self._construct_tf_sparse(L2)
                stack.append(L2_tf)
                L0 = L1 
                L1 = L2
        if poly_type == 'mono':
            for k in range(1, self.K):
                L1 = L@L0
                L1_tf = self._construct_tf_sparse(L1)
                stack.append(L1_tf)
                L0 = L1
        tf_LK = tf.sparse.concat(0, stack)
        return tf_LK
    
    
    def _get_poly_list(self, K, nside, indices, n_neighbors, poly_type):
        """
        Computes the graph laplacian operator to act on the input maps in tf.sparse format
        args:
        K: order of the polynomial to use
        nside: nside of the input maps
        indices: pixel indices of the input maps
        n_neighbors: number of neighbors to take into account when constructing the grap laplacian L
        poly_type: type of polynomial to use. 'chebyshev' for Chebyshev polynomials, 'legendre' for Legendre polynomials,
                   'mono' for monomials.

        Returns: the graph laplacian operator of order K with the desired polynomial type. in tf.sparse format.
                 will return a K*M x M sparse matrix where K is the order of polynomials and M is the number of pixels
        """
        sphere = SphereHealpix(subdivisions=nside, indexes=indices, nest=True, 
                               k=n_neighbors, lap_type='normalized')
        L = sphere.L.astype(np.float32)
        L = sparse.csr_matrix(L)
        lmax = 1.02 * eigsh(L, k=1, which='LM', return_eigenvectors=False)[0]
        L = self._rescale_L(L, lmax=lmax, scale=0.75)
        L0 = sparse.identity(len(indices), format='csr', dtype=L.dtype)
        L0_tf = self._construct_tf_sparse(L0)
        stack = [L0_tf]
        if poly_type == 'chebyshev':
            if K > 1:
                L1 = L
                L1_tf = self._construct_tf_sparse(L1)
                stack.append(L1_tf)
            for k in range(2, K):
                L2 = 2*L@L1-L0
                L2_tf = self._construct_tf_sparse(L2)
                stack.append(L2_tf)
                L0 = L1 
                L1 = L2
        if poly_type == 'legendre':
            if K > 1:
                L1 = L
                L1_tf = self._construct_tf_sparse(L1)
                stack.append(L1_tf)
            for k in range(2, K):
                L2 = ((2*k-1)/k)*L@L1-((k-1)/k)*L0
                L2_tf = self._construct_tf_sparse(L2)
                stack.append(L2_tf)
                L0 = L1 
                L1 = L2
        if poly_type == 'mono':
            for k in range(1, self.K):
                L1 = L@L0
                L1_tf = self._construct_tf_sparse(L1)
                stack.append(L1_tf)
                L0 = L1
        return stack


    def _transformed_indices(self, nside_in, nside_out, indices_in):
        """
        utility function to get new indices after a change in n_side

        Arguments:
        nside_in: n_side of the input map, integer of form 2**p
        nside_out: n_side of the output map, integer of form 2**p
        indices: set of healpy map indices, if mask is applied, indices should
                 be generated by deepsphere.extend_indices. should be 
                 in nest ordering.

        Returns:
        transformed_indices: the set of indices with n_side out
                             given n_side in and the set of indices.
        """
        mask_in = np.zeros(hp.nside2npix(nside_in))
        mask_in[indices_in] = 1.0
        mask_out = hp.ud_grade(map_in=mask_in, nside_out=nside_out, order_in="NEST", order_out="NEST")
        transformed_indices = np.arange(hp.nside2npix(nside_out))[mask_out > 1e-12]
        return transformed_indices
    
###########-------------------Wrapper layers for the abstract graph layers--------------------------##########
class HealpyConv(): 
    """
    A wrapper layer for the abstract GraphConv layer
    """
    def __init__(self, 
                 nside, 
                 indices, 
                 n_neighbors, 
                 poly_type,
                 K, 
                 Fout, 
                 activation=None, 
                 use_bias=False, 
                 use_bn=False, 
                 kernel_initializer=None, 
                 kernel_regularizer=None, 
                 **kwargs):
        """
        Initializes the graph convolutional layer, assuming the input has dimension (B, M, F)
        :param nside: nside of the input map
        :param indices: indices of the input map
        :param n_neighbors: n_neighbors of the graph laplacian
        :param poly_type: the type of polynomial to use. 'chebyshev', 'legendre' or 'mono'
        :param K: Order of the polynomial to use
        :param Fout: Number of features (channels) of the output, default to number of input channels
        :param activation: the activation function to use after the layer, defaults to linear
        :param use_bias: Use learnable bias weights
        :param use_bn: Apply batch norm before adding the bias
        :param kernel_initializer: optional. initializer to use for weight initialization, 
                                   defaults to 'he_normal'
        :param kerneL_regularizer: regularization function for the weights. defaults to None.
        :param kwargs: additional keyword arguments passed on to add_weight        
        """

        if n_neighbors not in [8, 20, 40, 60]:
            raise NotImplementedError(f"The requested number of neighbors {n_neighbors} is nor supported."
                                      f"Choose either 8, 20, 40 or 60.")
        if poly_type not in ['chebyshev', 'legendre', 'mono']:
            raise NotImplementedError(f"The requested polynomial type {poly_type} is not supported."
                                      f"Choose from chebyshev, legendre or mono.")
        self.nside = nside
        self.indices = indices
        self.n_neighbors = n_neighbors
        self.poly_type = poly_type
        self.K = K
        self.Fout = Fout
        self.activation = activation
        self.use_bias = use_bias
        self.use_bn = use_bn
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kwargs = kwargs
        self.key = self.poly_type, self.nside, self.n_neighbors, self.K #dictionary keys for the P_L dictionary.
        self.nside_out = self.nside
        self.indices_out = self.indices
        
    def _get_layer(self, L=None, polyK=None):
        #get the abstract graphconv layer
        return GraphConv(K=self.K, 
                         Fout=self.Fout, 
                         L=L,
                         poly_type=self.poly_type,
                         polyK = polyK,
                         activation=self.activation, 
                         use_bias=self.use_bias, 
                         use_bn=self.use_bn,
                         kernel_initializer=self.kernel_initializer, 
                         kernel_regularizer=self.kernel_regularizer, 
                         **self.kwargs)
    
    def call(self, inputs):
        actual_layer = self._get_layer(L, polyK)
        return actual_layer(inputs)

class HealpyDepthwiseConv(): 
    """
    A wrapper layer for the abstract GraphDepthwiseConv layer
    """
    def __init__(self, 
                 nside, 
                 indices, 
                 n_neighbors, 
                 poly_type,
                 K, 
                 depth_multiplier=1, 
                 activation=None, 
                 use_bias=False, 
                 use_bn=False, 
                 kernel_initializer=None, 
                 kernel_regularizer=None, 
                 **kwargs):
        """
        Initializes the graph convolutional layer, assuming the input has dimension (B, M, F)
        :param nside: nside of the input map
        :param indices: indices of the input map
        :param n_neighbors: n_neighbors of the graph laplacian
        :param poly_type: the type of polynomial to use. 'chebyshev', 'legendre' or 'mono'
        :param K: Order of the polynomial to use
        :param depth_multiplier: Number of features output per number of input channels
        :param activation: the activation function to use after the layer, defaults to linear
        :param use_bias: Use learnable bias weights
        :param use_bn: Apply batch norm before adding the bias
        :param kernel_initializer: optional. initializer to use for weight initialization, 
                                   defaults to 'he_normal'
        :param kernel_regularizer: regularization function for the weights. defaults to None.
        :param kwargs: additional keyword arguments passed on to add_weight        
        """

        if n_neighbors not in [8, 20, 40, 60]:
            raise NotImplementedError(f"The requested number of neighbors {n_neighbors} is nor supported."
                                      f"Choose either 8, 20, 40 or 60.")
        if poly_type not in ['chebyshev', 'legendre', 'mono']:
            raise NotImplementedError(f"The requested polynomial type {poly_type} is not supported."
                                      f"Choose from chebyshev, legendre or mono.")
        self.nside = nside
        self.indices = indices
        self.n_neighbors = n_neighbors
        self.poly_type = poly_type
        self.K = K
        self.depth_multiplier = depth_multiplier
        self.activation = activation
        self.use_bias = use_bias
        self.use_bn = use_bn
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.kwargs = kwargs
        self.key = self.poly_type, self.nside, self.n_neighbors, self.K #dictionary keys for the P_L dictionary.
        self.nside_out = self.nside
        self.indices_out = self.indices
        
    def _get_layer(self, L=None, polyK=None):
        #get the abstract graphconv layer
        return GraphDepthwiseConv(K=self.K,
                                  depth_multiplier=self.depth_multiplier,
                                  L=L,
                                  poly_type=self.poly_type,
                                  polyK=polyK,
                                  activation=self.activation, 
                                  use_bias=self.use_bias, 
                                  use_bn=self.use_bn,
                                  kernel_initializer=self.kernel_initializer, 
                                  kernel_regularizer=self.kernel_regularizer, 
                                  **self.kwargs)
    
    def call(self, inputs):
        actual_layer = self._get_layer(L, polyK)
        return actual_layer(inputs)
        
class HealpySeparableConv(): 
    """
    A wrapper layer for the abstract GraphSeparableConv layer
    """
    def __init__(self, 
                 nside, 
                 indices, 
                 n_neighbors, 
                 poly_type,
                 K,
                 Fout,
                 depth_multiplier=1, 
                 activation=None, 
                 use_bias=False, 
                 use_bn=False, 
                 pointwise_initializer=None,
                 depthwise_initializer=None,
                 pointwise_regularizer=None,
                 depthwise_regularizer=None,
                 **kwargs):
        """
        Initializes the graph convolutional layer, assuming the input has dimension (B, M, F)
        :param nside: nside of the input map
        :param indices: indices of the input map
        :param n_neighbors: n_neighbors of the graph laplacian
        :param poly_type: the type of polynomial to use. 'chebyshev', 'legendre' or 'mono'
        :param K: Order of the polynomial to use
        :param Fout: number of features of the output
        :param depth_multiplier: Number of features output per number of input channels
        :param activation: the activation function to use after the layer, defaults to linear
        :param use_bias: Use learnable bias weights
        :param use_bn: Apply batch norm before adding the bias
        :param pointwise_initializer: optional. initializer to use for weight initialization for the
                                      pointwise kernel, defaults to 'he_normal'                        
        :param depthwise_initializer: optional. initializer to use for weight initialization for the
                                      depthwise kernel, defaults to 'he_normal'          
        :param pointwise_regularizer: regularization function for the pointwise weights. defaults to None.
        :param depthwise_regularizer: regularization function for the depthwise weights. defaults to None.
        :param kwargs: additional keyword arguments passed on to add_weight        
        """

        if n_neighbors not in [8, 20, 40, 60]:
            raise NotImplementedError(f"The requested number of neighbors {n_neighbors} is nor supported."
                                      f"Choose either 8, 20, 40 or 60.")
        if poly_type not in ['chebyshev', 'legendre', 'mono']:
            raise NotImplementedError(f"The requested polynomial type {poly_type} is not supported."
                                      f"Choose from chebyshev, legendre or mono.")
        self.nside = nside
        self.indices = indices
        self.n_neighbors = n_neighbors
        self.poly_type = poly_type
        self.K = K
        self.Fout = Fout
        self.depth_multiplier = depth_multiplier
        self.activation = activation
        self.use_bias = use_bias
        self.use_bn = use_bn
        self.pointwise_initializer = pointwise_initializer
        self.depthwise_initializer = depthwise_initializer                             
        self.pointwise_regularizer = pointwise_regularizer
        self.depthwise_regularizer = depthwise_regularizer
        self.kwargs = kwargs
        self.key = self.poly_type, self.nside, self.n_neighbors, self.K #dictionary keys for the P_L dictionary.
        self.nside_out = self.nside
        self.indices_out = self.indices
        
    def _get_layer(self, L=None, polyK=None):
        #get the abstract graphconv layer
        return GraphSeparableConv(K=self.K, 
                                  Fout=self.Fout,
                                  depth_multiplier=self.depth_multiplier,
                                  L=L,
                                  poly_type=self.poly_type,
                                  polyK=polyK,
                                  activation=self.activation, 
                                  use_bias=self.use_bias, 
                                  use_bn=self.use_bn,
                                  pointwise_initializer=self.pointwise_initializer, 
                                  depthwise_initializer=self.depthwise_initializer,
                                  pointwise_regularizer=self.pointwise_regularizer,
                                  depthwise_regularizer=self.depthwise_regularizer,
                                  **self.kwargs)
    
    def call(self, inputs):
        actual_layer = self._get_layer(L, polyK)
        return actual_layer(inputs)           