import arviz as az
import GPy
import numpy as np
import pymc3 as pm

from functools  import partial
from GPy.models import GPKroneckerGaussianRegression as KGR
from GPy.models import GPRegression                  as GPR

class SpaceTimeKron(object):
    '''
    Provides a convenient API for fitting a space-time-process Kronecker GP model in PyMC3 and generating
    predictions at new coordinates.
    '''

    def __init__(self, ls_bounds=None, white_noise=True):
        '''
        Arguments
        ---------
        ls_bounds : 2-tuple of sequences
            A pair of lists or other sequence which are each 3 elements long corresponding
            to the lower and upper limits for the GP correlation length scale in space and time. 
            They respectively are assigned to the spatial proximity, temporal periodic, and temporal 
            proximity parts of the GP kernel. For example, ls_bounds=([0.1, 0.1, 0.5], (2, 2, 3))
            restricts the length scale of the spatial proximity kernel to be in the set [0.1, 2] while the temporal
            periodic kernel should be restricted  to [0.1, 2] and finally the temporal proximity kernel is restricted
            to [0.5, 3]. If this argument is set to None, these limits will automatically be determined from the
            training data.         
        white_noise : bool or float
            If True, this specifies that there is observation noise associated with the data and that
            the intensity of this noise is a parameter to be estimated. If False, this quantity is omitted
            from the model (though a superficially small value is still retained for numerical purposes).
            If set to a float, this indicates that the observation noise is known at a fixed value.

        '''
        if ls_bounds:
            self.ls_bounds = ls_bounds
        else:
            self.ls_bounds = None
            
        self.white_noise = white_noise
    
    def fit(self, X, y, ls_kwargs={}, fit_method='mcmc', vi_iter=100_000, mcmc_iter=1000, chains=2, cores=2):
        '''
        Runs Markov chain Monte Carlo to fit the free parameters of the Gaussian process model.
        
        Arguments
        ---------
        X : sequence of Numpy arrays
            Each array in this sequence must have the same length in the first dimension. The first
            array will usually be a Sx2 array of spatial coordinates while the second array will be
            a Tx1 array of temporal coordinates. The third and final coordinate array indicates positions
            in parameter space.            
        y : Numpy array
            (S*T)x1 array of observations at the space-time grid locations.
        ls_kwargs : dict
            Additional keyword arguments used in automatic extraction of length scale bounds. See
            "autocalc_lengthscale_bounds" for more information.
            
        '''
        
        _check_training_data(X,y)
        
        if not self.ls_bounds:
            self.ls_bounds = autocalc_lengthscale_bounds(X, **ls_kwargs)
        
        model, gp = _instantiate_stk_model(X, y, self.ls_bounds,
                                            self.white_noise)
        self.model = model
        self.gp    = gp
        
        with model:
            if fit_method == 'map':
                # Expand dims twice to get extra axes for chain and sample
                point_extra_dim = {k: np.expand_dims(np.expand_dims(v, axis=0),axis=0) for k,v in pm.find_MAP().items()}
                self.trace = az.from_dict(point_extra_dim)
            elif fit_method == 'mcmc':
                self.trace = pm.sample(tune=mcmc_iter, draws=mcmc_iter, return_inferencedata=True, chains=chains, cores=cores)
            elif fit_method == 'vi':
                self.approx = pm.fit(vi_iter)
                self.trace  = self.approx.sample()
                self.trace  = az.from_pymc3(self.trace)
            else:
                raise NotImplementedError(f'Fit method {fit_method} not supported')
            
    def predict(self, X, predict_kwargs={'diag':True, 'pred_noise':True}):
        '''
        Generate GP predictions at new spatiotemporal coordinates.
        
        Arguments
        ---------
        X : sequence of Numpy arrays
            The first array will usually be a Sx2 array of spatial coordinates while the second array will be
            a Tx1 array of temporal coordinates.
        predict_kwargs : dict
            Additional keyword arguments passed to PyMC3 gp object. "diag" controls whether or not
            a full predictive covariance matrix is returned versus a diagonal approxmation,
            and "pred_noise" controls whether additive measurement noise is included.
            
        Returns
        -------
        mu : Numpy array
            Posterior predictive mean
        var : Numpy array
            Posterior predictive variance
        '''
        
        with self.model:
            mu, var = self.gp.predict(X, **predict_kwargs)
            
        return mu, var 

def _instantiate_stk_model(kron_Xs, y, ls_bounds, white_noise, model=None):
    '''
    Instantiates a PyMC3 model object for the space-time-parameter Kronecker Gaussian 
    process. This GP has a 2D spatial 5/2 Matern covariance kernel a 
    5/2 Matern temporal covariance kernel function, and 5/2 Matern parameter space covariance kernel.
    '''
    if not model:
        model = pm.Model()
        
    with model:
        # Scalar average of the spatiotemporal field
        mean      = pm.Normal('mean', sd=10)
        mean_func = pm.gp.mean.Constant(mean)

        cov_fns = []
        
        n_components = len(kron_Xs)

        # Parameters governing the correlation distances and magnitudes
        # for different components of this model
        ls = pm.Uniform('ls',  lower=ls_bounds[0], upper=ls_bounds[1], shape=n_components)
        gp_variances = pm.HalfNormal('gp_variances', sd=1.0, shape=n_components)

        for i, xs in enumerate(kron_Xs):
            n_local_dims = xs.shape[-1]
            cov_fns += [pm.gp.cov.Matern52(n_local_dims, ls=ls[i], active_dims=np.arange(n_local_dims)) * gp_variances[i]]

        # White noise term that can either be fixed to a single
        # value or estimated from the data.
        if isinstance(white_noise, bool):
            if white_noise:
                sigma = pm.HalfNormal('sigma', sigma=1.)
            else:
                sigma = 0.01

        elif isinstance(white_noise, float):
            sigma = white_noise

        # We'll skip the Kronecker representation if there is only
        # a single covariance matrix
        if n_components == 1:
            X  = kron_Xs[0]
            gp = pm.gp.Marginal(mean_func=mean_func, cov_func=cov_fns[0])
            _  = gp.marginal_likelihood('ll', X, y, noise=sigma, is_observed=True)
        else:
            gp = pm.gp.MarginalKron(mean_func=mean_func, 
                                    cov_funcs=cov_fns)
            _ = gp.marginal_likelihood('ll', kron_Xs, y,
                                        sigma=sigma, is_observed=True)
    
    return model, gp

def length_bbox_diagonal(x):
    '''
    Calculates the Euclidean distance between diagonally opposite corners
    of a N-dimensional bounding hypervolume. In 1D case, this returns the length
    of the interval containing all the data. In the 2D case, this yields the length
    of the diagonal of the bounding box containing the coordinates. The main use of 
    this function is to calculate the largest possible relevant length scale for a 
    set of points in space.
    '''
    if x.ndim==1:
        x = x[:, np.newaxis]

    bbox_coords = np.zeros([2,x.shape[1]])
    
    for col in range(x.shape[1]): 
        bbox_coords[:, col] = np.min(x[:,col]), np.max(x[:,col])
        
    squared_distance = np.sum((bbox_coords[1]-bbox_coords[0])**2, axis=0)
    return squared_distance**0.5

def autocalc_lengthscale_bounds(kron_inputs, min_fraction=0.1, max_fraction=0.75):
    '''
    Automatically determine bounds on the lengthscales 
    '''
    bbox_lengths = [length_bbox_diagonal(x) for x in kron_inputs]
    
    lower_bounds = [min_fraction*x for x in bbox_lengths]
    upper_bounds = [max_fraction*x for x in bbox_lengths]
    
    return lower_bounds, upper_bounds 

def _check_training_data(X,y):
        
    if any([np.any(np.isnan(ary)) for ary in X]) or np.any(np.isnan(y)):
        raise ValueError('NaNs are not allowed in either the spatiotemporal coordinates nor the observation values.')

class SpaceTimeKronGPy(object):
    '''
    Class for instantiating and fitting a Kronecker Matern Gaussian process in GPy.
    '''
    def __init__(self, n_spatial_dims=2):
        self.spatial_kernel  = GPy.kern.Matern52(input_dim=n_spatial_dims) + GPy.kern.White(n_spatial_dims)
        self.temporal_kernel = GPy.kern.Matern52(input_dim=1)

    def fit(self, XY, T, vals, opt_kwargs={'max_f_eval':10000}):
        self.gp_model = KGR(XY, T, vals, self.spatial_kernel, self.temporal_kernel)
        self.gp_model.optimize(**opt_kwargs)

    def predict(self, XY, T):
        return self.gp_model.predict(XY, T)
        
class IndependentGR(object):
    def __init__(self, XY, T, vals, spatial_kernel_func):
        self.XY = XY
        self.T = T
        self.vals = vals
        self.gp_models = [GPR(XY, vals[:, [t]], spatial_kernel_func()) for t in np.arange(len(self.T))]
        
    def optimize(self, opt_kwargs={}):
        [m.optimize(**opt_kwargs) for m in self.gp_models]
    
    def predict(self, XY, T):
        out = [m.predict(XY) for m in self.gp_models]
        mean, var = np.concatenate([x[0] for x in out], axis=1), np.concatenate([x[1] for x in out], axis=1)
        return mean, var 
    
def make_kernel(n_spatial_dims):
    return GPy.kern.Matern52(input_dim=n_spatial_dims) + GPy.kern.White(n_spatial_dims)
    
class SpaceTimeIndependentGPy(object):
    '''
    Class for instantiating and fitting a Kronecker Matern Gaussian process in GPy.
    '''
    def __init__(self, n_spatial_dims=2, latent_noise=True):
        self.spatial_kernel_func  = partial(make_kernel, n_spatial_dims=n_spatial_dims)

            
    def fit(self, XY, T, vals, opt_kwargs=None):
        self.gp_model = IndependentGR(XY, T, vals, self.spatial_kernel_func)
        self.gp_model.optimize()
        self.gp_models = self.gp_model.gp_models

    def predict(self, XY, T):
        return self.gp_model.predict(XY, T)