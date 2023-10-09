import numpy as np

x = np.array([1.0, 2.0, 3.0])
print(x)

y = np.array([2.0, 4.0, 6.0])
print (x + y)

A = np.array([[1,2], [3,4]])
print(A.shape)
print(A.dtype)

B = np.array([10, 20])
# broadcast
print(A * B)

X = np.array([[51, 55], [14, 19], [0,4]])
print(X)
X = X.flatten()
print(X)
print(X[np.array([0, 2, 4])])
print(X > 15)
print(X[X>15])