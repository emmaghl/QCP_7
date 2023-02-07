import numpy as np

class Quantum_Computer:

    def __init__(self, Qubits):
        self.Register_Size = Qubits

    def ComputationalBasis(self):
        self.Zero = np.array([1, 0])  # This is |0> vector state
        self.One = np.array([0, 1])  # This is |1> vector state

    def Register(self):
        pass

    def Tensor_Prod(self, Q1, Q2):
        #IMPORTANT: Tensor product multiples the values of Q1 with the matrix Q2
        self.tensorprod = []
        for x in np.nditer(Q1): #iterate x over Q1
            self.tensorprod = np.append(self.tensorprod, x * Q2) #appends tensorprod with x'th value of Q1 * (matrix) Q2
        return self.tensorprod #ouput is linear tensor product (NOTE: matrix form infromation lost)

    def Qubit(self):
        self.a = np.random.random()+np.random.random()*1j #generates a random complex number to be assigned as coefficient to the |0> vector state
        modb = np.sqrt(1 - (np.absolute(self.a))**2) #produces norm of a second complex number as coefficient to |1> vector state to ensure normalisation
        randtheta = np.random.random() * 2 * np.pi #produces random angle in [0, 2π), which combined with the norm above will produce a second random complex number
        self.b = modb*np.cos(randtheta)+modb*np.sin(randtheta)*1j #generates a complex number to be assigned as coefficient to the |1> vector state
        # self.a = (0+1j)/2**0.5 #the norm of a**2 plus norm of b**2 should = 1, this is the porbability of finding qbit in either state,
        # self.b = (0+1j)/2**0.5 # the norm of b**2 by contrast is prob of finding qbit in state b. If measured, it will be in either a or b.
        return self.a*self.Zero + self.b*self.One

    def Basis(self):
        # returns a basis for the tensor product space given by the product of single qubit states
        n = self.Register_Size
        basis = np.zeros(2**n)
        return basis

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


