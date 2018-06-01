import numpy as np
from sklearn.externals.joblib import Parallel, delayed
from multiprocessing import cpu_count




def apply_parallel_joblib(func, data, *args, chunk=None, overlap=10,
					n_jobs=None, **kwargs):
    """
    Apply a function in parallel to overlapping chunks of an array

    Parameters
    ----------

    func : function
        name of function. Its first argument needs to be ``data``

    data : ndarray
        data to be chunked

    chunk : int
        chunk size (default value 100)

    overlap : int
        size of overlap between consecutive chunks

    n_jobs : int
        number of jobs to be used by joblib for parallel processing

    *args, **kwargs : other arguments to be passed to func

    Examples
    --------
    >>> from skimage import data, filters
    >>> coins = data.coins()
    >>> res = apply_parallel_joblib(filters.gaussian, coins, 2)
    """
    if chunk is None:
        l = len(data)
        try:
            ncpu = cpu_count()
        except NotImplementedError:
            ncpu = 4
        chunk = l // ncpu

    if n_jobs is None:
        n_jobs = -1
    sh0 = data.shape[0]
    nb_chunks = sh0 // chunk
    end_chunk = sh0 % chunk
    arg_list = [data[max(0, i*chunk - overlap):
                     min((i+1)*chunk + overlap, sh0)]
                            for i in range(0, nb_chunks)]
    if end_chunk > 0:
        arg_list.append(data[-end_chunk - overlap:])
    res_list = Parallel(n_jobs=n_jobs, backend="threading")(delayed(func)
                        (sub_im, *args, **kwargs) for sub_im in arg_list)
    output_dtype = res_list[0].dtype
    out_data = np.empty(data.shape, dtype=output_dtype)
    for i in range(1, nb_chunks):
        out_data[i*chunk:(i+1)*chunk] = res_list[i][overlap:overlap+chunk]
    out_data[:chunk] = res_list[0][:-overlap]
    if end_chunk>0:
        out_data[-end_chunk:] = res_list[-1][overlap:]
    return out_data



