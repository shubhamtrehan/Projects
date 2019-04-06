# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 20:41:32 2019

@author: ShubhamTREHAN
"""

import numpy as np
from numpy.linalg import inv
from numpy import linalg as la
import matplotlib.pyplot as plt
#np.random.seed(45)
lamb = [0.049, 0.135, 0.367, 1, 2.718, 7.389]
log_lamb = np.matrix(np.log(lamb)).T
            # Initialising a column of zeros
variance = []
bias_2 = []
X = np.random.uniform(0,1,(25,1))
X = np.matrix(np.linspace(X.min(),X.max(),25,endpoint = True)).T
tar = np.matrix(np.sin(np.pi*2*X))
mew=[0.15,0.25,0.5,0.75,0.9]
E_rms_test = []

for z in range(0,6):
    Y_train_mat = np.zeros((25,1))
    total = np.zeros((25,1)) 
    E_tot = 0
    for iter in range(0,100):
        X_train = np.random.uniform(0,1,(25,1))
        X_train = np.linspace(X_train.min(),X_train.max(),25,endpoint = True)    
        X_train = np.matrix(X_train).T
        e = np.random.normal(0, 0.3,(25,1))
        
        t_train = np.sin(X_train*np.pi*2.) + e 
        t_train = np.matrix(t_train)
        
        #test
        X_test = np.random.uniform(0,1,(1000,1))
        X_test = np.linspace(X_test.min(),X_test.max(),1000,endpoint = True)    
        X_test = np.matrix(X_test).T
        e = np.random.normal(0, 0.3,(1000,1))
        
        t_test = np.sin(X_test*np.pi*2.) + e         
        
        
        phi = np.ones((25,1))
        s = 0.1
        for i in range(0,5):
             diff = X_train - mew[i]*np.ones(X_train.shape)
             gau_col = np.exp(-np.square(diff)/(2*np.square(s)))
             phi = np.concatenate((phi, gau_col),axis = 1)
        
        k = phi.T@phi
        W = inv(k + lamb[z]*np.matrix(np.identity(6)))*phi.T*t_train
        Y = W.T@phi.T
        Y = Y.T
        Y_train_mat = np.concatenate((Y_train_mat, Y),axis = 1)
        total = total + Y
        phi_test = np.ones((1000,1))
        s = 0.1
        for i in range(0,5):
             diff = X_test - mew[i]*np.ones(X_test.shape)
             gau_col = np.exp(-np.square(diff)/(2*np.square(s)))
             phi_test = np.concatenate((phi_test, gau_col),axis = 1)
        
        
        Y_test = W.T@phi_test.T
        Y_test = Y_test.T
        col_diff = Y_test - t_test
        J_test = np.square(la.norm(col_diff))
        E = np.sqrt(J_test/1000)
        E_tot = E_tot + E
    
        
   
        plt.plot(X, tar,color='blue')
        plt.plot(X, Y,color='red',alpha=0.1,lw=2)
        plt.xlabel('X')
        plt.ylabel('target / predicted')
    plt.show()
    
    aver_vector = total/100
    bias = np.sum(np.square(aver_vector - tar),axis=0) / 25
    
    
    cols = np.split(Y_train_mat,101,axis=1) 
    col_sum = np.zeros((25,1))    #Initialising a coulms of zeros
    for col in range(1,101):
        col_diff = cols[col] - aver_vector
        col_diff_sq = np.square(col_diff)
        col_sum = col_sum + col_diff_sq
    
    col_sum = col_sum/100    
    var = np.array(np.sum(col_sum, axis=0) / 25)
    
    E_aver = E_tot/100
    E_rms_test.append(E_aver)
    
    variance.append(var)
    bias_2.append(bias)
for  num in range(6):
    bias_2[num] = bias_2[num].tolist()
    variance[num] = variance[num].tolist()
    
bias_2 = [i[0] for i in bias_2]
bias_2 = [i[0] for i in bias_2]
variance = [i[0] for i in variance]
variance = [i[0] for i in variance]

r=[]
for i in range(6):
    r.append(bias_2[i] + variance[i])
plt.figure(7)
plt.plot(log_lamb, variance,color='red')
plt.plot(log_lamb, bias_2,color='blue')
plt.plot(log_lamb, r,color='magenta')
plt.plot(log_lamb,  E_rms_test ,color='black')
plt.xlabel("ln"+u"\u03BB")
plt.legend(["variance","bias_2","bias_2+variance","test_error"])
plt.show()

     