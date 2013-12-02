from FileOperations import *
from PlottingFunctions import *
from AlgebraFunctions import *
import numpy as np

print('loading data...')
data = loadcsv('ex2data1.txt')
print('data loaded.')
data = np.mat(data)
vX = np.concatenate((data[:, 0],data[:, 1]), 1)
vy = data[:, 2]
m = len(vy)
X = np.concatenate((np.ones((m, 1)), vX), 1)
#X = np.mat(np.array([np.ones((m , 1)), np.mat(vX )] ))
y = np.mat(vy)

list_abs = data[:,0]
list_ord = data[:,1]
label = data[:,2]

x0_abs = []
x0_ord = []
x1_abs = []
x1_ord = []
for i in range(0,len(label)):
	if label[i]==1:
		x1_abs.append(list_abs[i])
		x1_ord.append(list_ord[i])

	else:
		x0_abs.append(list_abs[i])
		x0_ord.append(list_ord[i])
sets = [[x0_abs,x0_ord,'o'],[x1_abs,x1_ord,'+']]
MultiScatter(sets)




