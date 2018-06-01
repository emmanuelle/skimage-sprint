import numpy as np
import inspect
from skimage import exposure, feature, filters, measure, morphology, \
                    restoration, segmentation, transform, util, data, color
from timeit import default_timer
import pandas as pd
import pickle


def only_one_nondefault(args):
    """
    Returns True if the function has only one non-keyword parameter,
    False otherwise.
    """
    defaults = 0 if args.defaults is None else len(args.defaults)
    if len(args.args) >= 1 and (len(args.args) - defaults <= 1):
        return True
    else:
        return False


def run_benchmark(im, function_list, depth_list):
    times = {}
    times_dask = {}
    l = im.shape[0] // 2
    for function, depth in zip(function_list, depth_list):
        start = default_timer()
        _ = function[1](im)
        output_dtype = _.dtype
        end = default_timer()
        times[function[0]] = (end - start)
        start = default_timer()
        _ = util.apply_parallel(function[1], im, depth=depth, 
                                        dtype=output_dtype)
        end = default_timer()
        times_dask[function[0]] = (end - start)
    return times, times_dask

l1 = 4000

func_list, depth_list = pickle.load(open('chunk_func.pkl', 'rb'))

im_uint8 = data.binary_blobs(length=l1, volume_fraction=0.3).astype(np.uint8)
im_float = im_uint8.astype(np.float)

skip_functions = []
times, times_dask = run_benchmark(im_uint8, func_list, depth_list)
function_names = sorted(times, key=times.get)
sorted_times = sorted(times.values())

for func_name, t in zip(function_names, sorted_times):
    print(func_name, t, times_dask[func_name])

all_times = {}

for key in times.keys():
    all_times[key] = (times[key], times_dask[key], times[key]/times_dask[key])


df = pd.DataFrame.from_dict(all_times, orient='index', 
			    columns=['execution time', 'apply_parallel time', 'ratio'])
df = df.sort_values(by='execution time')

df.to_csv('times_dask.csv')

