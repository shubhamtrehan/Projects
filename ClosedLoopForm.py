import numpy as np
import pandas 
import matplotlib.pyplot as plt

data = pandas.read_csv('carbig.csv')

data = data.replace(np.nan, 0)
A = np.array(data)
A.shape

x = A[:,0:1]
#normalise
h= np.max(x[:,0])
x = x/h
one_col = np.ones((len(x), 1), dtype=float)
X = np.hstack([one_col,x])
t = A[:,1:2]
gram_mat = np.mat(X.T)*np.mat(X)
inv = gram_mat.I
gram_mat.shape

W = inv*(X.T)*t
W.shape

Y = np.matmul((W.T),(X.T))
#Denormalise
x = x*h
plt.scatter(x, t,label= "stars", color= "green",  
            marker= "*", s=30)
Y = Y.T
plt.plot(x, Y,linewidth=2.0)
plt.title('Closed Form')
plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.legend(["ClosedForm"])
plt.show()


