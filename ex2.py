from FileOperations import *
from PlottingFunctions import *
from AlgebraFunctions import *
from sigmoid import *
from costFunction import *
import numpy as np
from scipy import optimize

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

zz = np.mat([[1,2,3],[4,5,6]])
zz  = np.mat(1)
zz = np.mat([-723.2])
gg = sigmoid(zz)
print gg

[m, n] = np.shape(X)
initial_theta  = np.zeros((n, 1))

result_jg = costFunction(initial_theta, X ,y)
print result_jg

#xx = map(lambda t: 0.206+t/25.0,range(-25,25, 1))
#yy = map(lambda t: 0.201+t/25.0,range(-25,25, 1))

#zz = [0.0]*len(xx)
#for i in range(0,len(xx)):
#Plot(xx,zz)

result_opt = optimize.fmin_ncg((lambda theta: cost(theta,X,y)) , initial_theta,(lambda theta: grad(theta,X,y)), maxiter = 400)
print result_opt

print 'final cost'
print cost(np.mat(result_opt).transpose(),X,y)
thetaf = result_opt
x1 = map(lambda t: t+0.0,range(0,100, 1))
x2 = map(lambda t: -1.0 /thetaf[2] * (thetaf[0]+thetaf[1]*t),x1)
Plot(x1,x2)
#sets.append([x1,x2,'0'])
sets.append([np.mat(x1),np.mat(x2),'0'])
MultiScatter(sets)