import numpy as np

class Quantum_Computer:

    def __init__(self, Qubits):
        self.Register_Size = Qubits
        self.Zero = np.array([1, 0])  # This is |0> vector state
        self.One = np.array([0, 1])  # This is |1> vector state
        self.input = 0

    def Register(self):
        pass

    def Tensor_Prod(self, Q1, Q2):
        #IMPORTANT: Tensor product multiples the values of Q1 with the matrix Q2
        self.tensorprod = []
        for x in np.nditer(Q1): #iterate x over Q1
            self.tensorprod = np.append(self.tensorprod, x * Q2) #appends tensorprod with x'th value of Q1 * (matrix) Q2
        return self.tensorprod #ouput is linear tensor product (NOTE: matrix form infromation lost)


    def Normalising(self):
        # returns an array of 2**n complex coefficients and ensures normalisation.
        j = 2**self.Register_Size
        self.coeffs = (0 + 0 * 1j) * np.zeros(j) #create an arbitrary numpy array of complex coefficients
        for i in range(j): #compute random complex numbers in polar form
            theta = np.random.random() * np.pi * 2 #generate random angles ranging [0, 2π)
            self.coeffs[i] = (np.cos(theta) + np.sin(theta) * 1j) / j # form complex numbers and set modulus to be 1/j for each so that j coefficients normalise to 1.
        return self.coeffs

    def Basis(self):
        # returns a basis for the tensor product space given by the product of single qubit states
        n = self.Register_Size
        self.basis = np.zeros(2**n)

        return self.basis

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


