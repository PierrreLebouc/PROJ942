import numpy as np
np.set_printoptions(precision=3)

# Load data
X = np.loadtxt('data.txt')
print(X)
# dimensions
n,m = X.shape
print(n,m)
# centering and reduction
mean = np.mean(X, axis = 0)
print("mean : ",mean)
std = np.std(X, axis = 0,ddof=1)
print("standard deviation : ",std)

print("\nACP can be performed with or without reduction")
print("with reduction = normaliztion by standard deviation")
print("without reduction = no normalization")
red = int(input("Reduction or not (1/0) ? "))
if red == 1 :
    Xc = (X - mean)/std
else :
    Xc = X - mean

# Covariance matrix on transpose of X
C = np.cov(Xc.T)
print(C)
# eigen values, eigen vectors
L,V = np.linalg.eig(C)
# eigen values and eigen vectors are ordered according to eigen values decreasing values
idx = L.argsort()[::-1]  # ordered eigen values index
L = L[idx] # ordered eigen values
V = V[:,idx] # corresponding ordered eigen vectors

print("\neigen values : ")
print(L)
print("eigen vectors : ")
print(V)


# Chosing the number of relevant eigen values
print("\nK is the number of principal components ( 1 <= K <= ",m,")")
K = int(input("K = ")) 
V_used = np.zeros((m,K)) # corresponding used eigen vectors

for k in range(0,K,1):
    V_used[:,k] = V[:,k]
#new components
Y = np.dot(X,V_used)
# Quality measure
qual = 0
for k in range(K):
    qual = qual + L[k]
qual = qual / np.sum(L)
print("\nquality indicator = ",qual)

#
#######################
# Application


# x0 = a new data1
x0 = np.zeros(m)

# TEST 1
######## 
x0[0] = 33; x0[1] = 6; x0[2] = 29; x0[3] = 199; x0[4] = 90
# OK for K>=2 , index = 5 [30.68, 6.80, 40.55, 208.76, 89.48] (whatever the normalization)
# KO for K=1, index = 4 with normalization, index = 2 without normalization

# first step : we look the nearest vector of x0 among all X vectors 
# 'nearest' : from the euclidean distance point ov view
tab_dist_x0 = sum( ((X - x0.T)**2).T )
ind_min_x = np.argmin(tab_dist_x0)
print("\nin the initial space, the nearest vector from x0 is at index ",ind_min_x)

# # second step : we look the nearest vector of x0 among all Y vectors 
# 'nearest' : from the euclidean distance point ov view

# TO DO


# TEST 2
######## 
#x0[0] = 37; x0[1] = 7; x0[2] = 42; x0[3] = 186; x0[4] = 94
