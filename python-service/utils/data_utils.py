import numpy as np

def round_array_to_nearest(array, base):
    array = [round(float(t), base) for t in array]
    return array

def replace_edge_zeros(array):
    arr = np.array(array)
    if np.all(arr == 0):
        return arr
    nonzero = np.nonzero(arr)[0]
    first_nonzero = nonzero[0]
    last_nonzero = nonzero[-1]
    # Replace leading zeros
    arr[:first_nonzero] = arr[first_nonzero]
    # Replace trailing zeros
    arr[last_nonzero+1:] = arr[last_nonzero]
    return arr