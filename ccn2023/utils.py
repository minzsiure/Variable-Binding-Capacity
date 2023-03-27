import numpy as np

def hebbian_update(W, out_vec, in_vec, beta):
    for i in np.where(out_vec!=0.)[0]:
        for j in np.where(in_vec!=0.)[0]:
            W[i,j] *= 1. + beta

def capk(activations, k):
    output = np.zeros_like(activations)
    if len(activations.shape) == 1:
        idx = np.argsort(activations)[-k:]
        output[idx] = 1.
    else:
        idx = np.argsort(activations, axis=0)[-k:, :]
        np.put_along_axis(output, idx, 1., axis=0)
    return output

def hamming(vec1, vec2):
    return np.sum(np.absolute(vec1-vec2))

