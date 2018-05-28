import numpy as np
import inspect
from skimage import exposure, feature, filters, measure, morphology, \
                    restoration, segmentation, transform, util, data, color
from line_profiler import LineProfiler

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


def run_profile(im, filename='profile.txt',
                    module_list=[color, exposure, feature, filters,
                                 measure, morphology, restoration, segmentation,
                                   transform, util], skip_functions=[]):
    lp = LineProfiler()

    functions = []
    for submodule in [exposure, feature, filters, measure, morphology,
                                restoration, segmentation, transform, util]:
        functions += inspect.getmembers(submodule, inspect.isfunction)

    with open(filename, 'a') as f:
        for function in functions:
            args = inspect.getargspec(function[1])
            only_one_argument = only_one_nondefault(args)
            if function[0] in skip_functions:
                continue
            if only_one_argument:
                try:
                    print(function[0])
                    lp_wrapper = lp(function[1])
                    res = lp_wrapper(im)
                    lp.print_stats(stream=f)
                except TypeError:
                    print('wrong type ', function[0])
                except:
                    print('error ', function[0])
    f.close()

l1 = 1000

im_uint8 = data.binary_blobs(length=l1, volume_fraction=0.3).astype(np.uint8)
im_float = im_uint8.astype(np.float)

skip_functions = ['hough_ellipse']
times = run_profile(im_uint8, skip_functions=skip_functions)





