import numpy as np
import pandas 
import matplotlib.pyplot as plt

data = pandas.read_csv('carbig.csv')
data = data.replace(np.nan, 0)
A = np.array(data)
A.shape


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

iters = 0                                #iteration counter
max_iters = 5000
lr1 = 0.0001                          # Learning Rate
gram_mat = np.mat(X.T)*np.mat(X)
np.random.seed(600)
curr_w = np.random.randn(1,2)
gram_mat.shape


old_w = np.matrix([0,0])
while iters < max_iters and not np.allclose(curr_w, old_w) :
    grad_J = (2*curr_w*gram_mat) - (2*np.mat(t.T)*np.mat(X))
    #print(grad_J)
    old_w = curr_w
    curr_w = curr_w -  lr1*(grad_J) #Grad descent
    print(curr_w)
    # diff = abs(last_w - new_w) #Change in x
    iters = iters+1 #iteration count


Y_n = np.matmul((X),(curr_w.T))
print(Y_n.shape)

#denormalise
x = x*h
plt.scatter(x, t, label= "stars", color= "green",  
            marker= "*", s=30)
plt.plot(x, Y_n,linewidth=2.0)
plt.title('Gradient Descent')
plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.legend(["Gradient Descent"])
plt.show()
#print(y_n.shape)