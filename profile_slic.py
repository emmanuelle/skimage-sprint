import numpy as np
from skimage.segmentation import slic
from time import time

im = np.zeros((400, 400, 100))
im[::2] = 1
times = []
for n_jobs in range(1, 5):
    t1 = time()
    res = slic(im, enforce_connectivity=False, n_jobs=n_jobs)
    t2 = time()
    print(t2 - t1)
    times.append(t2 - t1)


