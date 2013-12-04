import numpy as np
from sigmoid import *

def costFunction(theta, X, y):
    m = len(y)
    grad = np.zeros(np.shape(theta))
    J = 0
    thetat = theta.transpose()
    for i in range(0,m):
        J = J + 1.0/ m * ( -y[i] * np.log(sigmoid(thetat*X[i, :].transpose())) - (1 - y[i]) * np.log(1-sigmoid(thetat*X[i, :].transpose())))

    for j in range(0, len(theta)):
        for i in range(0, m):
            grad[j] = grad[j] + 1.0 / m * (sigmoid(thetat*X[i,:].transpose())-y[i])* X[i,j]

    return [J, grad]

def grad(theta, X, y):
    m = len(y)
    grad = np.zeros(np.shape(theta))
    thetat = theta.transpose()
    for j in range(0, len(theta)):
        for i in range(0, m):
            grad[j] = grad[j] + 1.0 / m * (sigmoid(thetat*X[i,:].transpose())-y[i])* X[i,j]

    return grad

def cost(theta, X, y):

    m = len(y)
    grad = np.zeros(np.shape(theta))
    J = 0.0
    thetat = theta.transpose()
    for i in range(0,m):
        if y[i]==1:
            J = J + 1.0/ m * ( -y[i] * np.log(sigmoid(thetat*X[i, :].transpose())))
        else:
            J = J + 1.0/ m * (-(1 - y[i]) * np.log(1-sigmoid(thetat*X[i, :].transpose())))
        #J = J + 1.0/ m * ( -y[i] * np.log(sigmoid(thetat*X[i, :].transpose())) - (1 - y[i]) * np.log(1-sigmoid(thetat*X[i, :].transpose())))
    print J
    return J

