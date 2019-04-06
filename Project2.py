#import random
import numpy as np
from numpy import linalg as la

np.random.seed(45)
# Training DATA
# When samples equal to 10
e = np.random.normal(0, 0.3,(1,10))
x_train = np.random.uniform(0,1,10)
#e = np.random.normal(0, 0.3,(1,10))
y_train = np.sin(x_train * np.pi*2.) + e

#Testing Data
e = np.random.normal(0, 0.3,(1,100))
x_test = np.random.uniform(0,1,100)
#e = np.random.normal(0, 0.3,(1,100))
y_test = np.sin(x_test * np.pi*2.) + e


x_test = np.matrix(x_test).T
x_train = np.matrix(x_train).T
y_train = np.matrix(y_train).T
y_test = np.matrix(y_test).T
N = x_train.shape[0]

ones_train = np.ones((x_train.shape[0], 1))
phi = np.c_[ones_train,np.power(x_train,1),np.power(x_train,2),np.power(x_train,3),np.power(x_train,4),np.power(x_train,5),np.power(x_train,6),np.power(x_train,7),np.power(x_train,8),np.power(x_train,9)]
E_rms_train = []  
weight = []  

# Calculating Training Weights   
for i in range(10):
    k = phi[:,0:i+1].T@phi[:,0:i+1]
    COND = np.linalg.cond(k)
    #inv = k.I
    inv = np.linalg.pinv(k)
    r = inv*phi[:,0:i+1].T
    w = r*y_train
    w = w.T
    w = np.matrix(w)
    weight.append(w)
    train_y =  w* phi[:,0:i+1].T
    #L2 norm
    diff = y_train - train_y.T
    NORM = np.square(la.norm(diff))
    #Erms
    E_train = np.sqrt(NORM/N)
    E_rms_train.append(E_train)

print(E_train)   
 
M = x_test.shape[0]
ones_test = np.ones((x_test.shape[0], 1))
phi = np.c_[ones_test,np.power(x_test,1),np.power(x_test,2),np.power(x_test,3),np.power(x_test,4),np.power(x_test,5),np.power(x_test,6),np.power(x_test,7),np.power(x_test,8),np.power(x_test,9)]
E_rms_test = []       
for i in range(10):
    test_y =  weight[i]* phi[:,0:i+1].T
    #L2 norm
    diff = y_test - test_y.T
    J_test = np.square(la.norm(diff))
    #Erms
    E_test = np.sqrt(J_test/M)
    E_rms_test.append(E_test)

print(E_test)      
M =[] 
for i in range(10):
    M.append(i)
import matplotlib.pyplot as plt
#plt.scatter(M, E_train)
plt.plot(M, E_rms_train, '-ok',color='blue')
plt.plot(M, E_rms_test, '-ok',color='red')
plt.xlabel('Model Complexity')
plt.ylabel('Error')
plt.legend(["Train error","Test Error"])
plt.title('Number of samples equal to 10')
plt.xlim([0,10])
plt.yticks(np.arange(min(E_rms_test), max(E_rms_test)+1, 1.0))
#plt.ylim([0,4])
plt.show()

#### When samples equal to 100

e = np.random.normal(0, 0.3,(1,100))
x_train = np.random.uniform(0,1,100)
y_train = np.sin(x_train * np.pi*2.) + e

#Testing Data
e = np.random.normal(0, 0.3,(1,100))
x_test = np.random.uniform(0,1,100)
y_test = np.sin(x_test * np.pi*2.) + e


x_test = np.matrix(x_test).T
x_train = np.matrix(x_train).T
y_train = np.matrix(y_train).T
y_test = np.matrix(y_test).T
N = x_train.shape[0]

ones_train = np.ones((x_train.shape[0], 1))
phi = np.c_[ones_train,np.power(x_train,1),np.power(x_train,2),np.power(x_train,3),np.power(x_train,4),np.power(x_train,5),np.power(x_train,6),np.power(x_train,7),np.power(x_train,8),np.power(x_train,9)]
E_rms_train = []  
weight = []  

# Calculating Training Weights   
for i in range(10):
    k = phi[:,0:i+1].T@phi[:,0:i+1]
    COND = np.linalg.cond(k)
    #inv = k.I
    inv = np.linalg.pinv(k)
    r = inv*phi[:,0:i+1].T
    w = r*y_train
    w = w.T
    w = np.matrix(w)
    weight.append(w)
    train_y =  w* phi[:,0:i+1].T
    #L2 norm
    diff = y_train - train_y.T
    NORM = np.square(la.norm(diff))
    #Erms
    E_train = np.sqrt(NORM/N)
    E_rms_train.append(E_train)

print(E_train)   
 
M = x_test.shape[0]
ones_test = np.ones((x_test.shape[0], 1))
phi = np.c_[ones_test,np.power(x_test,1),np.power(x_test,2),np.power(x_test,3),np.power(x_test,4),np.power(x_test,5),np.power(x_test,6),np.power(x_test,7),np.power(x_test,8),np.power(x_test,9)]
E_rms_test = []       
for i in range(10):
    test_y =  weight[i]* phi[:,0:i+1].T
    #L2 norm
    diff = y_test - test_y.T
    J_test = np.square(la.norm(diff))
    #Erms
    E_test = np.sqrt(J_test/M)
    E_rms_test.append(E_test)

print(E_test)      
M =[] 
for i in range(10):
    M.append(i)
import matplotlib.pyplot as plt
#plt.scatter(M, E_train)
plt.plot(M, E_rms_train, '-ok',color='blue')
plt.plot(M, E_rms_test, '-ok',color='red')
plt.xlabel('Model Complexity')
plt.ylabel('Error')
plt.legend(["Train error","Test Error"])
plt.title('Number of samples equal to 100')
plt.show()



