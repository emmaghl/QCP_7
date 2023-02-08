import numpy as np
import cmath

'''
a graveyard of trials
'''

# def Qubit(self):
#     self.a = np.random.random()+np.random.random()*1j #generates a random complex number to be assigned as coefficient to the |0> vector state
#     modb = np.sqrt(1 - (np.absolute(self.a))**2) #produces norm of a second complex number as coefficient to |1> vector state to ensure normalisation
#     randtheta = np.random.random() * 2 * np.pi #produces random angle in [0, 2π), which combined with the norm above will produce a second random complex number
#     self.b = modb*np.cos(randtheta)+modb*np.sin(randtheta)*1j #generates a complex number to be assigned as coefficient to the |1> vector state
#     # self.a = (0+1j)/2**0.5 #the norm of a**2 plus norm of b**2 should = 1, this is the porbability of finding qbit in either state,
#     # self.b = (0+1j)/2**0.5 # the norm of b**2 by contrast is prob of finding qbit in state b. If measured, it will be in either a or b.
#     return self.a*self.Zero + self.b*self.One


# n = 4
#
# j = 2 ** n
# coeffs = (0+0*1j)*np.zeros(j)
# for i in range(j):
#     theta = np.random.random()*np.pi*2
#     coeffs[i] =(np.cos(theta)+np.sin(theta)*1j)/j
#
# sum = 0
# for i in range(j):
#     sum += np.absolute(coeffs[i])
#
# print(sum)
#
#
#
#
#
# #
# #
# #
# # # returns j coefficients which normalise to 1
# # coeffs = (0+0*1j)*np.zeros(j)
# # for i in range(j):
# #     coeffs[i] = (np.random.random() + np.random.random()*1j)
# #
# # print(coeffs)
# #
# # norm = 0
# # for i in range(j):
# #     norm += np.absolute(coeffs[i]) ** 2
# # norm = norm**0.5
# #
# # coeffs2 = (0+0*1j)*np.zeros(j)
# # for i in range(j):
# #     theta = np.random.random()*np.pi*2
# #     newcoeff = np.absolute(coeffs[i])*(np.cos(theta)+np.sin(theta)*1j)/(norm**2)
# #     coeffs2[i] = newcoeff
# #
# # print(coeffs2)
# #
# # sum = 0
# # for i in range(j):
# #     sum += np.absolute(coeffs2[i])
# #
# # print(sum)
# #
# # #idea is to get a random angle and set the absolute value to 1/2**n. (would reduce comp. time)
