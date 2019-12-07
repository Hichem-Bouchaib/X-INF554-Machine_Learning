import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
np.set_printoptions(precision=3)

# Load the data

data = np.loadtxt('data/data_train.csv', delimiter=',')

# Prepare the data

X = data[:,0:-1]
y = data[:,-1]
# <!--
y[y >= 8] = 0       # <-- Task 7 data cleaning / removing outlier
# -->

# Inspect the data

plt.figure()
plt.hist(X[:,1], 10)
plt.savefig("fig/hist1.pdf")

# TODO <!--
plt.figure()
plt.hist(X[:,2], 10)
plt.savefig("fig/hist2.pdf")
# -->

plt.figure()
plt.plot(X[:,1],X[:,2], 'o')
plt.xlabel('$x_2$')
plt.ylabel('$x_3$')
plt.savefig("fig/data.pdf")

# TODO <!--
plt.figure()
plt.plot(X[:,0],y, 'o')
plt.xlabel('$x_1$')
plt.ylabel('$y$')
plt.savefig("fig/data_y.pdf")
plt.show()
# -->

# Standardization

# TODO <!--
m = np.mean(X,axis=0)
s = np.std(X,axis=0)
X = (X - m) / s
# -->

plt.figure()
plt.plot(X[:,1],X[:,2], 'o')
plt.xlabel('$x_2$')
plt.ylabel('$x_3$')
# plt.savefig("fig/data.pdf")
plt.show()
# <!-- PCA (Bonus Task #2. N.B. Need to do same projection on test data.)
from sklearn.decomposition import PCA

def pca_com(X, num_components):
    ''' Using principal components of PCA '''
    Z = PCA(n_components=2).fit_transform(X)
    return Z 
# -->

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
# TODO <!--
w = inv(Z.T @ Z) @ Z.T @ y
print("w = ", w)
# -->

# Evaluation 
# TODO <!--

def MSE(yt,yp):
    return np.mean((yt-yp)**2)

yp = Z @ w

data = np.loadtxt('data/data_test.csv', delimiter=',')

# Preparation and preprocessing 

X_test = data[:,0:-1]
X_test = (X_test - m) / s
Z_test = phi(X_test,degree)
y_test = data[:,-1]

# Prediction

y_pred = Z_test @ w

# Score

print("MSE on test data  ", MSE(y_test,y_pred))
print("MSE baseline      ", MSE(y_test,np.ones(len(y_test))*np.mean(y)))

# <TASK 8: You will need to make changes from '# Feature creation'
#          To get the exact results, you will need to leafe the anomally in

# -->
