import numpy as np

class Quantum_Computer:

    def __init__(self, Qubits):
        self.Register_Size = Qubits
        self.Zero = np.array([1, 0])  # This is |0> vector state
        self.One = np.array([0, 1])  # This is |1> vector state
        self.I = np.array([[1,0],[0,1]]) #Identity gate
        self.X = np.array([[0, 1], [1, 0]]) #Flips the |0> to |1> and vice versa
        self.Y = np.array([[0, 0 + 1j], [0 - 1j, 0]], dtype=complex) #converts |0> to i|1> and |1> to -i|0>
        self.Z = np.array([[1, 0], [0, -1]]) #sends |1> to -|1> and |0> to |0>
        self.RNot = 1 / np.sqrt(2) * np.array([[1, -1], [1, 1]]) #sends |0> to 0.5^(-0.5)(|0>+|1>) and |1> to 0.5^(-0.5)(|1>-|0>)
        self.Hadamard = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]]) #sends |0> to |+> and |1> to |->
        self.Phase = np.array([[1, 0], [1, 0 + 1j]], dtype=complex) #sends |0>+|1> to |0>+i|1>
        self.T = np.array([[1,0],[0,1 / np.sqrt(2) * (1+1j)]], dtype=complex) #square root of phase (rotates by pi/8)
        self.CNot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]) #reversable xor: |00> -> |00>, |01> -> |11>
        self.Swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) #¯\_(ツ)_/¯

    def Tensor_Prod(self, Q1, Q2):
        #IMPORTANT: Tensor product multiples the values of Q1 with the matrix Q2
        self.tensorprod = []
        for x in np.nditer(Q1): #iterate x over Q1
            self.tensorprod = np.append(self.tensorprod, x * Q2) #appends tensorprod with x'th value of Q1 * (matrix) Q2
        self.tensorprod = np.asmatrix(self.tensorprod)
        #ouput is linear tensor product (NOTE: matrix form infromation lost)

    def Coefficients(self):
        # returns an array of 2**n complex coefficients and ensures normalisation.
        j = 2**self.Register_Size
        self.coeffs = (0 + 0 * 1j) * np.zeros(j) #create an arbitrary numpy array of complex coefficients
        for i in range(j): #compute random complex numbers in polar form
            theta = np.random.random() * np.pi * 2 #generate random angles ranging [0, 2π)
            self.coeffs[i] = (np.cos(theta) + np.sin(theta) * 1j) / j # form complex numbers and set modulus to be 1/j for each so that j coefficients normalise to 1.

    def Basis(self):
        # returns a basis for the tensor product space given by the product of single qubit states
        N = self.Register_Size
        Q = []
        for i in range(0, 2 ** N):
            digit = []
            if i < (2 ** N) / 2:
                base = self.Zero
                digit.append(0)
            else:
                base = self.One
                digit.append(1)
            for j in range(1, N):
                x = i
                for k in range(0, len(digit)):
                    x -= digit[k] * (2 ** N) / (2 ** (k + 1))
                if x < (2 ** N) / (2 ** (j + 1)):
                    base = self.Tensor_Prod(base, self.Zero)
                    digit.append(0)
                else:
                    base = self.Tensor_Prod(base, self.One)
                    digit.append(1)
            Q.append(base)
        # to look up how numpy stores information and if it's more efficient to return the transposed basis or to transpose it each time on use
        self.Q = np.asmatrix(Q)
        for i in range(len(Q)):
            self.Q.append(np.transpose(Q[i])) #transposes all the incoming basis states

    def Psi(self):  #Our register doesn't need to call the basis states (yet), all we need is a column with n entries all equal to 1 (the sum of all the basis states), our normalised coefficients
        column_ones = np.transpose(np.ones(self.Register_Size))
        return np.matmul(self.coeffs, column_ones)

    def Hadamard_Logic(self, k):
        # k is the kth state on which the hadamard is acting, and k in (0, n-1) where n is the number of quibits
        if k == 0:
            H = np.asmatrix(self.tensorprod(self.Hadamard, self.I))
        elif k == 1:
            H = np.asmatrix(self.tensorprod(self.I, self.Hadamard))
        else:
            H = np.asmatrix(self.tensorprod(self.I, self.I))

        for i in range(2, len(self.Register_Size)):
            if i == k:
                H = np.asmatrix(self.tensorprod(H, self.Hadamard))
            else:
                H = np.asmatrix(self.tensorprod(H, self.I))



#function logicgate(1, 3) =  tensor.product (tensor.product(hadarmard, identity ) , identity) 
 
#function logicgate(2, 3) =  tensor.product (tensor.product(identity, hadarmard ) , identity)    

#function logicgate(3, 3) =  tensor.product (tensor.product(identity, identity ) , hadamrd)

#function logicgate(3, 4) =  tensor product(tensor.product (tensor.product(identity, identity ) , hadamard) , identity)

#function logicgate (k, n) =  tensor.product (tensor.product(hadarmard, identity ) , identity) 
 
   # n= qubits k = 0, ... (n - 1) 
