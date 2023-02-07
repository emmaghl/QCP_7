import numpy as np

class Quantum_Computer:

    def __init__(self, Qubits):
        self.Register_Size = Qubits

    def Register(self):
        pass
    def Tensor_Prod(self, Q1, Q2):
        #IMPORTANT: Tensor product multiples the values of Q1 with the matrix Q2
        tensorprod = []
        for x in np.nditer(Q1): #iterate x over Q1
            tensorprod = np.append(tensorprod, x * Q2) #appends tensorprod with x'th value of Q1 * (matrix) Q2
        return tensorprod #ouput is linear tensor product (NOTE: matrix form infromation lost)

    def Qubit(self):
        self.Zero = np.array([[1, 0],[0, 0]]) #This is |0> vector state
        self.One = np.array([[0,0], [0,1]]) #This is |1> vector state
        self.a = (0+1j)/2**0.5 #the norm of a**2 plus norm of b**2 should = 1, this is the porbability of finding qbit in either state, 
        self.b = (0+1j)/2**0.5 # the norm of b**2 by contrast is prob of finding qbit in state b. If measured, it will be in either a or b. 
        return a*Zero + b*One

class Gates:
    def __init__(self, expression):
        self.input = expression

    def X(self):
        return np.matmul(np.array([[0, 1], [1, 0]]), self.input)

    def Y(self):
        return np.matmul(np.array([[0, 0 + 1j], [0 - 1j, 0]], dtype=complex), self.input)

    def Z(self):
        return np.matmul(np.array([[1, 0], [0, -1]]), self.input)

    def Root_Not(self):
        return np.matmul(1 / np.sqrt(2) * np.array([[1, -1], [1, 1]]), self.input)

    def Hadamard(self):
        return np.matmul(1 / np.sqrt(2) * np.array([[1, 1], [1, -1]]), self.input)

    def Phase(self):
        return np.matmul(np.array([[1, 0], [1, 0 + 1j]], dtype=complex), self.input)

    def T(self):
        return np.matmul(np.array([[1,0],[0,1 / np.sqrt(2) * (1+1j)]], dtype=complex), self.input)

    def CNot(self):
        return np.matmul(np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]), self.input)

    def Swap(self):
        return np.matmul(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]), self.input)


