import numpy as np
import inspect
from skimage import exposure, feature, filters, measure, morphology, \
                    restoration, segmentation, transform, util, data, color
from timeit import default_timer
import pandas as pd
import cv2
import time

skimagefunctions = []
skimagefuncdict={}
for submodule in [exposure, feature, filters, measure, morphology,
                            restoration, segmentation, transform, util]:
    skimagefunctions += inspect.getmembers(submodule, inspect.isfunction)
for x, y in skimagefunctions:
    skimagefuncdict[x]=y

def run_benchmark(img):
    print('image dimensions are {} by {}'.format(np.shape(img)[0], np.shape(img)[1]))
    results=[]
    for function in (benchmark_adjust_gamma, benchmark_adjust_log, \
                        benchmark_CLAHE, benchmark_equalize_hist, \
                        benchmark_CLAHE, benchmark_corner_harris):
        skimagetime, cv2time = function(img)
        results += [[function.__code__.co_name[10:], skimagetime, cv2time]]
        print('\tfor function {}'.format(function.__code__.co_name[10:]))
        print('\t\tskimage takes {} sec to run'.format(skimagetime))
        print('\t\tcv2 takes {} sec ot run'.format(cv2time))
        print()

    return [np.shape(img), results]

def benchmark_adjust_gamma(img):
    gamma = 2

    def cv2_adjust_gamma(image, gamma=1.0):

       invGamma = 1.0 / gamma
       table = np.array([((i / 255.0) ** invGamma) * 255
          for i in np.arange(0, 256)]).astype("uint8")

       return cv2.LUT(image, table)


    start=default_timer()
    skimg1=skimagefuncdict['adjust_gamma'](img, gamma)
    stop=default_timer()
    skimagetime = (stop - start)

    start = default_timer()
    cv2img1=cv2_adjust_gamma(img, gamma)
    stop = default_timer()
    cv2time = (stop - start)

    return (skimagetime, cv2time)

def benchmark_adjust_log(img):
    gain = 1

    def cv2_adjust_log(image, gain=1.0):

       table = (255*gain*np.log2([1 + (i / 255.0)
          for i in np.arange(0, 256)])).astype("uint8")

       return cv2.LUT(image, table)


    start=default_timer()
    skimg1= skimagefuncdict['adjust_log'](img, gain)
    stop=default_timer()
    skimagetime = (stop - start)

    start = default_timer()
    cv2img1= cv2_adjust_log(img, gain)
    stop = default_timer()
    cv2time = (stop - start)

    return (skimagetime, cv2time)

def benchmark_CLAHE(img):

    start=default_timer()
    skimg1= (255*skimagefuncdict['equalize_adapthist'](img)).astype("uint8")
    stop=default_timer()
    skimagetime = (stop - start)

    start = default_timer()
    clahe = cv2.createCLAHE(clipLimit=26, tileGridSize=(8,8))
    cv2img1 = clahe.apply(img)
    stop = default_timer()
    cv2time = (stop - start)

    return (skimagetime, cv2time)

def benchmark_equalize_hist(img):

    start=default_timer()
    skimg1= (255*skimagefuncdict['equalize_hist'](img)).astype("uint8")
    stop=default_timer()
    skimagetime = (stop - start)

    start = default_timer()
    cv2img1 = cv2.equalizeHist(img)
    stop = default_timer()
    cv2time = (stop - start)

    return (skimagetime, cv2time)

def benchmark_canny(img):

    start=default_timer()
    skimg1 = skimagefuncdict['canny'](img, low_threshold=0.1, \
                                        high_threshold=0.2)
    stop=default_timer()
    skimagetime = (stop - start)

    start = default_timer()
    cv2img1 = cv2.Canny(img, 25, 51)
    stop = default_timer()
    cv2time = (stop - start)

    return (skimagetime, cv2time)

def benchmark_corner_harris(img):

    start=default_timer()
    skimg1 = skimagefuncdict['corner_harris'](img)
    stop=default_timer()
    skimagetime = (stop - start)

    start = default_timer()
    cv2img1 = cv2.cornerHarris(img, blockSize=15, ksize=3, k=0.05)
    stop = default_timer()
    cv2time = (stop - start)

    return (skimagetime, cv2time)

im_uint8_1 = data.camera()
run_benchmark(im_uint8_1)

im_uint8_2 = np.vstack([np.hstack([data.camera(), data.camera()]), \
                np.hstack([data.camera(), data.camera()])])
run_benchmark(im_uint8_2)

im_uint8_3 = np.vstack([np.hstack([im_uint8_2, im_uint8_2]), \
                np.hstack([im_uint8_2, im_uint8_2])])
run_benchmark(im_uint8_3)
