import numpy as np

class Quantum_Computer:

    def __init__(self, Qubits):
        self.Register_Size = Qubits
        self.X = Gates.X()
        self.Y = Gates.Y()
        self.Z = Gates.Z()
        self.H = Gates.Hadamard()
        self.CNot = Gates.C_not()
        self.swap = Gates.Swap()

    def Register(self):
        pass

    def Tensor_Prod(self):
        pass

    def Qubit(self):
        self.Zero = np.array([[1, 0],[0, 0]]) #This is |0> vector state
        self.One = np.array([[0,0], [0,1]]) #This is |1> vector state
        #Look into normalize function se we can change the values for a and b
        self.a = (0+1j)/2**0.5 #the norm of a**2 plus norm of b**2 should = 1, this is the porbability of finding qbit in either state,
        self.b = (0+1j)/2**0.5 # the norm of b**2 by contrast is prob of finding qbit in state b. If measured, it will be in either a or b.
        return a*Zero + b*One


class Gates:
    def __init__(self, input):
        self.inpt = input

    def X(self):
        return np.matmul(np.array([[0, 1], [1, 0]]), self.inpt)

    def Y(self):
        return np.matmul(np.array([[0, 0 + 1j], [0 - 1j, 0]], dtype=complex), self.inpt)

    def Z(self):
        return np.matmul(np.array([[1, 0], [0, -1]]), self.inpt)

    def Root_Not(self):
        return np.matmul(1 / np.sqrt(2) * np.array([[1, -1], [1, 1]]), self.inpt)

    def Hadamard(self):
        return np.matmul(1 / np.sqrt(2) * np.array([[1, 1], [1, -1]]), self.inpt)

    def Phase(self):
        return np.matmul(np.array([[1, 0], [1, 0 + 1j]], dtype=complex), self.inpt)

    def T(self):
        return np.matmul(np.array([[1,0],[0,1 / np.sqrt(2) * (1+1j)]], dtype=complex), self.inpt)

    def C_not(self):
        return np.matmul(np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]), self.inpt)

    def Swap(self):
        return np.matmul(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]), self.inpt)


#
#
#     def Tensor_Prod(self, a, b):
#         # a and b matrices as numpy array of arrays
#         len
#         for element in a:
#             counter = 0
#             check = 0
#             while check < len(b)
#
#         return self.c
#
# # a = np.arange(60.).reshape(3,4,5)
# # b = np.arange(24.).reshape(4,3,2)
#
# a =  np.array([[1,-1],[3,2]])
#
# print(len(a))



