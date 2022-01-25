import numpy as np

a = np.array([1,2,4,4,5])
print(type(a))
print(a.shape)

b = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

c = np.array([[1,2,3,6],[4,5,6,4], [7,8,9,1], [10,11,12,15],[1,3,3333,22]])
# print(c.shape)
d = np.array([0,2,3,1,3])
# print(c[np.arange(5),d])

print(np.sum(b, axis = 1))
