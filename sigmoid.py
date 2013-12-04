import numpy as np

def sigmoid(z):
    zz  = np.mat(z)
    size_z = np.shape(z)
    g = np.zeros(size_z)
    dim_i = size_z[0]
    dim_j = size_z[1]

    for i in range(0,dim_i):
        for j in range(0,dim_j):
            g[i,j]= 1.0 / (1.0 + np.exp(-z[i,j]))
    return float(g)
