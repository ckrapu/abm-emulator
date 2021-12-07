import fire
import json
import logging
import numpy as np

from emulators import SpaceTimeKron


'''
Hack to try to keep progressbar output from disappearing before
it hits the log files.
See https://github.com/fastai/fastprogress/issues/62.
'''
from fastprogress import fastprogress
fastprogress.printing = lambda: True

logging.basicConfig(level=logging.CRITICAL,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

'''
Functions used to preprocess response variables prior to modeling.
'''
transform_fn_mapping = {
        'plus1log' : lambda x: np.log(x + 1),
        'none'     : lambda x: x      
    }

transform_fn_mapping_inv = {
        'plus1log' : lambda x: np.exp(x) - 1,
        'none'     : lambda x: x      
    }

VALID_MODEL_TYPES = ['space', 'time', 'process']


def dump_default(obj):
    '''
    Helps json.dumps figure out what to do with
    a Numpy array which is not natively serializable.
    '''
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError('Not serializable')

def average_point(trace):
    '''
    Compute posterior mean dict from Arviz InferenceData object.
    '''
    return {k: trace.posterior[k].mean(axis=(0,1)).to_numpy() for k in trace.posterior.keys()}

def at_least_2d(array):
    '''
    Forces a Numpy array to have at least two dimensions.
    '''
    if array.ndim == 1:
        return array[:, None]
    else:
        return array
    
def constrain_unit_cube(array, reduce_axis=0):
    '''
    Subtract minimum and divide by maximum
    to ensure returned values are constrained to the 
    unit cube.
    '''
    offset = array.min(axis=reduce_axis)
    unit_array = array - offset
    scale = unit_array.max(axis=reduce_axis)
    return unit_array / scale, offset, scale

def fit_model(input_filepath, output_filepath, model_type, fit_method, 
              response_transform='none', split_char='+', vi_iter=100_000, mcmc_iter=500,
              return_trace=False):
    '''
    Utility for determining type of surrogate model to construct, preprocessing input/response data, 
    and running a parameter estimation algorithm.
    '''
    
    transform_fn = transform_fn_mapping[response_transform]

    with open(input_filepath, 'r') as src:
        data = json.load(src)

    for k in data.keys():
        data[k] = np.asarray(data[k])

    coord_keys = model_type.split(split_char)

    for k in coord_keys:
        if k not in VALID_MODEL_TYPES:
            raise ValueError(f'Model type {k} not recognized.')

    # The PyMC3 convention is to require coordinate arrays to 
    # be 2D, even if the domain is actually 1D.
    kron_Xs_raw = [at_least_2d(data[k]) for k in coord_keys]

    input_scales  = []
    input_offsets = []

    # Shift and rescale all inputs to unit cube
    kron_Xs = []
    for arr in kron_Xs_raw:
        unit_array, offset, scale = constrain_unit_cube(arr)
        input_scales  += [scale]
        input_offsets += [offset]
        kron_Xs += [unit_array]

    # Select only training input points in process variable
    # space
    kron_Xs[-1] = kron_Xs[-1][data['train_indices']]

    # A power transform or similar preprocessing
    # technique may be applied at this stage before
    # scaling and subtracting an offset.
    response_array = transform_fn(data[f'{model_type}_response'])
    response_axes  = tuple(np.arange(response_array.ndim))
    response_array, offset, scale = constrain_unit_cube(response_array, 
                                                        reduce_axis=response_axes)
    response_array = response_array[...,data['train_indices']]
    y = response_array.flatten()

    stpk = SpaceTimeKron()
    logging.critical(f'Starting parameter estimation with {fit_method.upper()} for {len(y)} observations.')

    for i, Xs in enumerate(kron_Xs):
        logging.critical(f'Input matrix {i} has dimension {Xs.shape}')

    stpk.fit(kron_Xs, y, fit_method=fit_method, vi_iter=vi_iter, mcmc_iter=mcmc_iter)    
    
    # Merge dictionaries and cast to list type so that target
    # arrays are serializable and human-readable
    trace = stpk.trace.to_dict()

    # We'll add these extra variables into the trace
    # so that we can generate predictions at new sites
    # using only the trace data structure.s
    extra_bookkeeping = {
        'input_scales'    : input_scales,
        'input_offsets'   : input_offsets,
        'response_offset' : offset,
        'response_scale'  : scale,
        'transform_fn'    : response_transform
    }

    for k, v in extra_bookkeeping.items():
        if k in trace.keys():
            raise KeyError 
        trace[k] = v
    
    with open(output_filepath, 'w') as outfile:
        json.dump(trace, outfile, default=dump_default)

    logging.critical(f'Outputs saved to {output_filepath}.')

    if return_trace:
        return trace

if __name__ == '__main__':
    fire.Fire()
