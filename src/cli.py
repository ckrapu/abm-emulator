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

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)): 
            return None

        return json.JSONEncoder.default(self, obj)


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

    unit_array  = array - offset
    scale       = unit_array.max(axis=reduce_axis)
    unit_array  = unit_array.astype(float) / scale.astype(float)

    return unit_array, offset, scale

def prep_data(input_filepath, model_type, response_transform='none', split_char='+', train_only=True):
    transform_fn = transform_fn_mapping[response_transform]

    with open(input_filepath, 'r') as src:
        data = json.load(src)

    for k in data.keys():
        data[k] = np.asarray(data[k])

    train_indices, test_indices = data['train_indices'], data['test_indices']

    coord_keys = model_type.split(split_char)

    for k in coord_keys:
        if k not in VALID_MODEL_TYPES:
            raise ValueError(f'Model type {k} not recognized. Remember to demarcate tokens with {split_char}')

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

    # A power transform or similar preprocessing
    # technique may be applied at this stage before
    # scaling and subtracting an offset.
    response_array = transform_fn(data[f'{model_type}_response'])
    response_axes  = tuple(np.arange(response_array.ndim))
    response_array, offset, scale = constrain_unit_cube(response_array, 
                                                        reduce_axis=response_axes)

    logging.critical(f'Response variable with shape {response_array.shape} standardized with mapping {response_transform}, scale={scale} and offset={offset}')     
    
    # This if clause is for compatibility with earlier versions of the
    # scripts.
    if train_only:
        # Select only training input points in process variable space
        kron_Xs[-1] = kron_Xs[-1][train_indices]                                
        response_array = response_array[..., train_indices]
        y = response_array.flatten()
        outputs = kron_Xs, y, offset, scale, input_scales, input_offsets
    else:
        y = response_array.flatten()
        outputs = kron_Xs, y, offset, scale, input_scales, input_offsets, train_indices, test_indices

    
    return outputs


def fit_model(input_filepath, output_filepath, model_type, fit_method, 
              response_transform='none', split_char='+', vi_iter=100_000, mcmc_iter=1000,
              return_trace=False, target_accept=0.80, init='jitter+adapt_diag',white_noise=True,
              return_model=False, independent=False):
    '''
    Utility for determining type of surrogate model to construct, preprocessing input/response data, 
    and running a parameter estimation algorithm.
    '''

    inputs =  prep_data(input_filepath, model_type, response_transform=response_transform, split_char=split_char)
    kron_Xs, y, offset, scale, input_scales, input_offsets = inputs
    
    stpk = SpaceTimeKron(white_noise=white_noise)
    logging.critical(f'Starting parameter estimation with {fit_method.upper()} for {len(y)} observations.')

    for i, Xs in enumerate(kron_Xs):
        logging.critical(f'Input matrix {i} has dimension {Xs.shape}')
    
    model = stpk.fit(kron_Xs, y, fit_method=fit_method, vi_iter=vi_iter, mcmc_iter=mcmc_iter, init=init, target_accept=target_accept, return_model=return_model)    
    if return_model:
        return kron_Xs, y, stpk, model

    # Merge dictionaries and cast to list type so that target
    # arrays are serializable and human-readable
    trace = stpk.trace.to_dict()

    # We'll add these extra variables into the trace
    # so that we can generate predictions at new sites
    # using only the trace data structure.s
    extra_bookkeeping = {
        'input_scales'    : [list(x) for x in input_scales],
        'input_offsets'   : [list(x) for x in input_offsets],
        'response_offset' : offset,
        'response_scale'  : scale,
        'transform_fn'    : response_transform,
        'input_file'      : input_filepath
    }

    for k, v in extra_bookkeeping.items():
        if k in trace.keys():
            raise KeyError 
        trace['posterior'][k] = v

    if return_trace:
        return trace

    else:          
        print(trace)
        with open(output_filepath, 'w') as outfile:
            json.dump(trace, outfile, cls=NumpyEncoder)

        logging.critical(f'Outputs saved to {output_filepath}.')

    

if __name__ == '__main__':
    fire.Fire()
