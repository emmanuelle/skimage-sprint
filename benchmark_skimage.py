import numpy as np
import inspect
from skimage import exposure, feature, filters, measure, morphology, \
                    restoration, segmentation, transform, util, data, color
from timeit import default_timer
import pandas as pd

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


def run_benchmark(im, module_list=[color, exposure, feature, filters, measure,
                                   morphology, restoration, segmentation,
                                   transform, util], skip_functions=[]):
    times = {}

    functions = []
    for submodule in [exposure, feature, filters, measure, morphology,
                                restoration, segmentation, transform, util]:
        functions += inspect.getmembers(submodule, inspect.isfunction)


    for function in functions:
        args = inspect.getargspec(function[1])
        only_one_argument = only_one_nondefault(args)
        if function[0] in skip_functions:
            continue
        if only_one_argument:
            try:
                print(function[0])
                start = default_timer()
                function[1](im)
                end = default_timer()
                times[function[0]] = (end - start)
            except TypeError:
                print('wrong type ', function[0])
            except:
                print('error ', function[0])
    return times

l1 = 1000
l2 = 4000

im_uint8 = data.binary_blobs(length=l1, volume_fraction=0.3).astype(np.uint8)
im_float = im_uint8.astype(np.float)

skip_functions = ['hough_ellipse']
times = run_benchmark(im_uint8, skip_functions=skip_functions)
function_names = sorted(times, key=times.get)
sorted_times = sorted(times.values())

for func_name, t in zip(function_names, sorted_times):
    print(func_name, t)

skip_functions += function_names[-4:]
bigim_uint8 = data.binary_blobs(length=l2,
                    volume_fraction=0.3).astype(np.uint8)
times_long = run_benchmark(bigim_uint8, skip_functions=skip_functions)
function_names = sorted(times_long, key=times.get)
sorted_times = sorted(times_long.values())

for func_name, t in zip(function_names, sorted_times):
    print(func_name, t)


all_times = {}

for key in times.keys():
    if key in times_long.keys():
    	all_times[key] = (times[key], times_long[key])
    else:
    	all_times[key] = (times[key], np.nan)


df = pd.DataFrame.from_dict(all_times, orient='index')
df = df.sort_values(by=['0'])

df.to_csv('times.csv')
