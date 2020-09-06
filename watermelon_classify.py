# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 08:53:28 2020

@author: 未海
"""
#import the lib
import numpy as np
from sklearn.model_selection import train_test_split
#imoort the data set 
data=np.array([[0.697, 0.460, 1],
               [0.774, 0.376, 1],
               [0.634, 0.264, 1],
               [0.608, 0.318, 1],
               [0.556, 0.215, 1],
               [0.403, 0.237, 1],
               [0.481, 0.149, 1],
               [0.437, 0.211, 1],
               [0.666, 0.091, 0],
               [0.243, 0.267, 0],
               [0.245, 0.057, 0],
               [0.343, 0.099, 0],
               [0.639, 0.161, 0],
               [0.657, 0.198, 0],
               [0.360, 0.370, 0],
               [0.593, 0.042, 0],
               [0.719, 0.103, 0]])
x=data[:,0:2]
y=data[:,2]
#split the set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)
def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s
#calculate the gradient of w and b
def sunshi(w,b,x,y):
    y_hat=sigmoid(np.dot(w.T,x)+b)
    m=x.shape[1]
    cost=-np.sum(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))/m
    dw=np.dot(x,(y_hat-y).T)/m
    db=np.sum(y_hat-y)/m
    grads={"dw":dw,
           "db":db}
    return grads,cost
#initialize the vector w and b
def initialize(dim):
    w=np.zeros((dim,1))
    b=0
    return w,b
#optimizes w and b by running a gradient descent algorithm
def optimize(x,y,times,learning_rate):
    costs=[]
    w,b=initialize(x.shape[0])
    for i in range(times):
        grads,cost=sunshi(w,b,x,y)
        dw=grads["dw"]
        db=grads["db"]
        w=w-learning_rate*dw
        b=b-learning_rate*db
        if i%100==0:
            costs.append(cost)
            print("Cost after iteration %i:%f"%(i,cost))
    parmas={"w":w,
            "b":b}
    grads={"dw":dw,
           "db":db}
    return parmas,grads,costs
#predict the label of the data using the logistic regression
def predict(x,w,b):
    m=x.shape[1]
    y_prediction=np.zeros((1,m))
    w=w.reshape(x.shape[0],1)
    A=sigmoid(np.dot(w.T,x)+b)
    for i in range (A.shape[1]):
        if A[0,i]>=0.5:
            y_prediction[0,i]=1
        else:
            y_prediction[0,i]=0
    return y_prediction
def model(x_train,x_test,y_train,y_test,times,learning_rate):
    parmas,grads,costs=optimize(x_train,y_train,times,learning_rate)
    w=parmas["w"]
    b=parmas["b"]
    y_train_hat=predict(x_train,w,b)
    y_test_hat=predict(x_test,w,b)
    print("The accuracy of the train data: {}%".format(100 - np.mean(np.abs(y_train-y_train_hat))*100))
    print("The accuracy of the test data: {}%".format(100 - np.mean(np.abs(y_test-y_test_hat))*100))
    dic={"w":w,
         "b":b,
         "learning_rate":learning_rate,
         "times":times,
         "y_train_hat":y_train_hat,
         "y_test_hat":y_test_hat,
         "costs":costs}
    return dic
x_train=x_train.T
x_test=x_test.T
y_train=y_train.T
y_test=y_test.T
d=model(x_train, x_test, y_train, y_test, times=2000, learning_rate=0.5)



    
    
    
    
    
    
    
       
            
    
    
    
    
