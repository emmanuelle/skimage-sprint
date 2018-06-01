import numpy as np
import inspect
from skimage import exposure, feature, filters, measure, morphology, \
                    restoration, segmentation, transform, util, data, color
from timeit import default_timer
import pandas as pd
import pickle
from chunk_joblib import apply_parallel_joblib


def run_benchmark(im, function_list, depth_list, skip_functions=[]):
    times = {}
    times_joblib = {}
    l = im.shape[0] // 4
    for function, depth in zip(function_list, depth_list):
        if function[0] in skip_functions:
            continue
        start = default_timer()
        _ = function[1](im)
        output_dtype = _.dtype
        end = default_timer()
        times[function[0]] = (end - start)
        start = default_timer()
        _ = apply_parallel_joblib(function[1], im, chunk=l, overlap=2)
        end = default_timer()
        times_joblib[function[0]] = (end - start)
    return times, times_joblib

l1 = 4000

func_list, depth_list = pickle.load(open('chunk_func.pkl', 'rb'))

im_uint8 = data.binary_blobs(length=l1, volume_fraction=0.3).astype(np.uint8)
im_float = im_uint8.astype(np.float)



skip_functions = ['median', 'convex_hull_object', 'denoise_nl_means']
times, times_joblib = run_benchmark(im_uint8, func_list, depth_list, skip_functions=skip_functions)
function_names = sorted(times, key=times.get)
sorted_times = sorted(times.values())

for func_name, t in zip(function_names, sorted_times):
    print(func_name, t, times_joblib[func_name])

all_times = {}

for key in times.keys():
    all_times[key] = (times[key], times_joblib[key], times[key]/times_joblib[key])


df = pd.DataFrame.from_dict(all_times, orient='index', 
			    columns=['execution time', 'apply_parallel time', 'ratio'])
df = df.sort_values(by='execution time')

df.to_csv('times_joblib_thread.csv')

