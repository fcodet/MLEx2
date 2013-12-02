import numpy as np

def onevector(n): #returns a column vector of n 1's
	onematrix = np.ones((n,1))
	return onematrix[:, 0]

def zerovector(n): #returns a column vector of n 0's
	zeromatrix = np.zeros((n,1))
	return zeromatrix[:, 0]

