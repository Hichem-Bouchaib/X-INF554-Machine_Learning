import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
np.set_printoptions(precision=3)

# Load the data

data = np.loadtxt('data/data_train.csv', delimiter=',')

# Prepare the data

X = data[:,0:-1]
y = data[:,-1]
# 

# Inspect the data

plt.figure()
plt.hist(X[:,1], 10)
plt.savefig("fig/hist1.pdf")

# TODO 

plt.figure()
plt.plot(X[:,1],X[:,2], 'o')
plt.xlabel('$x_2$')
plt.ylabel('$x_3$')
plt.savefig("fig/data.pdf")

# TODO 

# Standardization

# TODO 

# 

# Feature creation

def phi(X, degree):
    N,D = X.shape
    for d in range(2,degree+1):
        X = np.column_stack([X,X[:,0:D]**d])
    X = np.column_stack([np.ones(len(X)), X])
    return X

# Polynomial degree
degree = 2

Z = phi(X,degree)

# Estimating the coefficients
# TODO 

# Evaluation 
# TODO 
