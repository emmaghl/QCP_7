import numpy as np
import cmath
n = 4

j = 2 ** n
# returns j coefficients which normalise to 1
coeffs = (0+0*1j)*np.zeros(j)
for i in range(j):
    coeffs[i] = (np.random.random() + np.random.random()*1j)

print(coeffs)

norm = 0
for i in range(j):
    norm += np.absolute(coeffs[i]) ** 2
norm = norm**0.5

coeffs2 = (0+0*1j)*np.zeros(j)
for i in range(j):
    theta = np.random.random()*np.pi*2
    newcoeff = np.absolute(coeffs[i])*(np.cos(theta)+np.sin(theta)*1j)/(norm**2)
    coeffs2[i] = newcoeff

print(coeffs2)

sum = 0
for i in range(j):
    sum += np.absolute(coeffs2[i])

print(sum)

#idea is to get a random angle and set the absolute value to 1/2**n. (would reduce comp. time)
