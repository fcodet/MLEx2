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
gg = sigmoid(zz)
print gg

[m, n] = np.shape(X)
initial_theta  = np.zeros((n, 1))

result_jg = costFunction(initial_theta, X ,y)
print result_jg

xx = range(-10,10, 1)
yy = range(-10,10, 1)
zzz = cost(np.mat([xx,yy]),X,y)
fig = plt.figure()
ax = fig.gca(projection='3d')
xxx, yyy = np.meshgrid(xx, yy)
ax.plot_surface(xxx,yyy,zzz,rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
plt.show()

result_opt = optimize.fmin_ncg((lambda theta: cost(theta,X,y)) , initial_theta,(lambda theta: grad(theta,X,y)), maxiter = 400)
