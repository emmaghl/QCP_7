import numpy as np
n = 4

j = 2 ** n
# returns j coefficients which normalise to 1
coeffs = []
for i in range(j):
    coeffs.append(np.random.random() + np.random.random()*1j)


norm = 0
for i in range(j):
    norm += np.absolute(coeffs[i]) ** 2
print(norm)
norm = norm**0.5
print(norm)

coeffs2 = []
for i in range(j):
    theta = np.random.random()*np.pi*2
    newcoeff = np.absolute(coeffs[i])/norm
    coeffs2.append(newcoeff)


sum = 0
for i in range(j):
    sum += np.absolute(coeffs2[i])

print(sum)

