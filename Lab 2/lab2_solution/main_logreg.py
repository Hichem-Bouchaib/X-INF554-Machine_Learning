import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    # Sigmoid fn TODO <! -- 
    g = 1./(1.+np.exp(-z))
    g = np.clip(g,0.001,0.999)
    return g
    # -->
             
def cost(w, X, y): 
    # Computes the cost using w as the parameters for logistic regression. 
    E = 0
    # TODO <!-- 
    N = X.shape[0] # number of examples
    for i in range(N):
        E = E + (-y[i] * np.log(sigmoid(np.dot(X[i,:],w))) - (1 - y[i]) * np.log(1 - sigmoid(np.dot(X[i,:],w))))
    E /= N
    # -->
    return E[0]
    
def compute_grad(w, X, y):
    # Computes the gradient of the cost with respect to the parameters.
    
    dE = np.zeros_like(w) # initialize gradient
    # TODO <!-- 
    dE = X.T @ (sigmoid(X @ w) - y)
    # -->
    return dE

def predict(w, X):
    # Predict whether each label is 0 or 1 using learned logistic regression parameters w. The threshold is set at 0.5

    N = X.shape[0] # number of examples
    yp = np.zeros(N) # predicted classes of examples
    
    # TODO <! --
    
    for i in range(N):
        yp[i] = (sigmoid(np.dot(X[i],w)) > 0.5) * 1
            
    # -->
    return yp



#======================================================================
# Load the dataset
data = np.loadtxt('./data/data.csv', delimiter=',')
 
#Add intercept term to 
data_1 = np.ones((data.shape[0], 4))
data_1[:, 1:] = data

# Standardize the data N.B. This line was missing, but it's quite important:
# (It will still work without standardization, but may behave eratically)
data_1[:,1:3] = (data_1[:,1:3] - np.mean(data_1[:,1:3],axis=0)) / np.std(data_1[:,1:3],axis=0)

n_test = 20
X = data_1[:-n_test, 0:3]
y = data_1[:-n_test, -1]
X_test = data_1[-n_test:, 0:3]
y_test = data_1[-n_test:, -1]


# Plot data 
pos = np.where(y == 1) # instances of class 1
neg = np.where(y == 0) # instances of class 0
plt.scatter(X[pos, 1], X[pos, 2], marker='o', c='b')
plt.scatter(X[neg, 1], X[neg, 2], marker='x', c='r')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(['Admitted', 'Not Admitted'])
plt.show()


N = X.shape[0]

# Initialize fitting parameters 
w = np.random.randn(3,1) * 0.05

# Gradient Descent
# TODO <!-- 
alpha = 0.005
n_iterations = 1000
log = []
log_test = []
for i in range(n_iterations): 
    w = w - compute_grad(w,X,y.reshape(-1,1)) * alpha
    # SGD
    #n = np.random.choice(N)
    #w = w - compute_grad(w,X[n].reshape(1,-1),y[n].reshape(1,-1)) * alpha
    if i % 10 == 0:
        c = cost(w,X,y)
        log.append(c)
        c_test = cost(w,X_test,y_test)
        log_test.append(c_test)
        print("Cost: %4.3f" % c)
# --> 

# Plot the decision boundary
plt.figure()
plot_x = np.array([min(X[:, 1]), max(X[:, 2])])
plot_y = (- 1.0 / w[2,0]) * (w[1,0] * plot_x + w[0,0])
plt.plot(plot_x, plot_y)
plt.scatter(X[pos, 1], X[pos, 2], marker='o', c='b')
plt.scatter(X[neg, 1], X[neg, 2], marker='x', c='r')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(['Decision Boundary', 'Admitted', 'Not Admitted'])
plt.show()

# Compute accuracy on the training set
accuracy = np.mean(predict(w, X) - y)
print("\nAccuracy: %4.3f" % accuracy)

# TODO Plot error curves <!-- 
plt.figure()
plt.plot(range(0,n_iterations,10),log,label="train")
plt.plot(range(0,n_iterations,10),log_test,label="test")
plt.legend()
plt.show()
#-->
