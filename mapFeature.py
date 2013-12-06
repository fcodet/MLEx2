import numpy as np
def mapFeature(X1,X2):
	degree = 6
	m = len(X1)
	X = np.ones((m, 1))
	for i in range(1,degree+1):
		for j in range(0,i+1):
			temp = np.zeros((m,1))
			for k in range(0,m):
				#temp[k] = np.exp((i-j)*np.log(X1[k]))*np.exp((j)*np.log(X2[k]))
				temp[k] = X1[k]**(i-j) * X2[k]**j
			X = np.concatenate((X,temp), 1)
	return X

