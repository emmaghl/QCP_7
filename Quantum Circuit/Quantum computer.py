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
        tensorprod = []
        for x in np.nditer(Q1): #iterate x over Q1
            tensorprod = np.append(tensorprod, x * Q2) #appends tensorprod with x'th value of Q1 * (matrix) Q2
        tensorprod = np.asmatrix(tensorprod)
        #ouput is linear tensor product (NOTE: matrix form infromation lost)
        return tensorprod

    def Sparse(self, Matrix): #defines a sparse matrix of the form row i column j has value {}
        rows = np.shape(Matrix)[0]
        cols = np.shape(Matrix)[1]
        SMatrix = [] #output matrix
        for i in range(rows):
            for j in range(cols):
                if Matrix[i,j] != 0: #if the value of the matrix element i,j is not 0 then store the value and the location
                    SMatrix.append([i,j,Matrix[i,j]]) #Output array: (row, column, value)
        return SMatrix #return output

    def Coefficients(self):
        # returns an array of 2**n complex coefficients and ensures normalisation.
        j = 2**self.Register_Size
        self.coeffs = (0 + 0 * 1j) * np.zeros(j) #create an arbitrary numpy array of complex coefficients
        for i in range(j): #compute random complex numbers in polar form
            theta = np.random.random() * np.pi * 2 #generate random angles ranging [0, 2π)
            self.coeffs[i] = (np.cos(theta) + np.sin(theta) * 1j) / j # form complex numbers and set modulus to be 1/j for each so that j coefficients normalise to 1.
        self.coeffs.shape = (j, 1)

    def Basis(self):
        N = self.Register_Size
        self.Q = []

        for i in range(0, 2 ** N):
            self.Q.append(np.zeros(2 ** N))
            self.Q[i][i] = 1
            self.Q[i].shape = (2 ** N, 1)

    def Psi(self):  #Our register doesn't need to call the basis states (yet), all we need is a column with n entries all equal to 1 (the sum of all the basis states), our normalised coefficients
        return np.matmul(self.Q, np.transpose(self.coeffs))

    def Single_Logic(self, gate, positions):
        '''
        - param gate: list of gate names to be applied
        - param positions: list of lists. each entry corresponds to the respective gate
        and contains a list of qubit position(s) on which to apply that gate
        '''

        assert len(gate) == len(positions), "unequal list lenghts" #the number of gates should match the position lists

        # this is only one step. so only one logic gate can be applied to a single qubit. so must return an error
        # if any value within the position sub-lists is repeated
        list_check = [] #create a unique list with the each of the arguments of the sublists of positions
        for k in range(len(positions)):
            for i in range(len(positions[k])):
                list_check.append((positions[k])[i])

        assert len(list(list_check)) == len(set(list_check)), "repeated position value"  #ensure lenght of list is equal to lenght of unique values in list i.e. avoid repetiotion

        gate_inputs = ["H", "RNot", "Phase", "X", "Y", "Z", "T"] #maps the string input to the relevant matrix and creates an array
        matrices = [self.Hadamard, self.RNot, self.Phase, self.X, self.Y, self.Z, self.T]
        M = []
        for j in range(len(gate)):
            for i in range(len(gate_inputs)):
                if str(gate[j]) == str(gate_inputs[i]):
                    M.append(np.asmatrix(matrices[i]))
        assert len(M) > 0, ("Please enter one of the following gates and ensure correct spelling: H, RNot, Phase, X, Y, Z, T")

        L = self.I
        for i in range(len(positions)):
            for j in range(len(positions[i])):
                if j == 0:
                    L = M[i]
                else:
                    L = L

        for l in range(1, self.Register_Size):
            for i in range(len(positions)):
                for j in range(len(positions[i])):
                    if l == j:
                        L = (self.Tensor_Prod(L, M[i]))
                        print(L)
                    else:
                        L = (self.Tensor_Prod(L, self.I))

        return L

'''
#tesing Single_Logic
Q = Quantum_Computer(4)

gate = ["H", "RNot"]
positions = [[1,3], [4]]
print(Q.Single_Logic(gate, positions))
'''