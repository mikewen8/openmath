import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('C:\Users\M Wen\Desktop\Python pipeline\data\Cookingdata.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[0:1000].T
X_dev = data_dev[1:n]
Y_dev = data_dev[0]

data_train = data[1000:m]
Y_train = data_train[0]
X_train = data_train[1:n]

###parameters
def init_params():
    W1 = np.random.rand(10, 10)
    b1 = np.random.rand(10,9)
    W2 = np.random.rand(10,9)
    b2 = np.random.rand(10,1)
    return W1,b1,W2,b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    return np.exp(Z)/ np.sum(np.exp(Z))

def forward_prop(W1,b1,W2,b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(X) + b2
    A2 = softmax(A1)
    
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max)+1)
    one_hot_Y[np.arange(Y.size,Y)]
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
    
def deriv_ReLU(Z):
    return Z>0
    
def back_prop(Z1, A1, A2, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1/m*dZ2.dot(A1.T)
    db2 = 1/m*np.sum(dZ2,2)
    dZ1 = W2.T.dot(dZ2)*deriv_ReLU(Z1)
    dW1 = 1/m*dZ2.dot(X.T)
    db1 = 1/m*np.sum(dZ1,2)
    return dW1, db1, dW2, db2

def update_params(W1,b1,W2,b2,dW1,db1,dW2,db2, alpha):
    W1 = W1-alpha*dW1
    b1 = b1-alpha*db1
    W2 = W2-alpha*dW2
    b2 = b2-alpha*db2
    return W1, b1, W2, b2

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print (predictions, Y)
    return  np.sum(predictions == Y)/Y.size

def gradient_descent(X, Y, iterations, alpha):
    W1,b1,W2,b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1,b1,W2,b2,X)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 ==0:
            print("Iterations:", i)
            print("Accuracy", get_accuracy(get_predictions(A2),Y))
        return W1,b1,W2,b2
    


