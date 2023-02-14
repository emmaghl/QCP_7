import numpy as np

class Quantum_Computer:

    def __init__(self, Qubits):
        self.Register_Size = Qubits

        self.Zero = np.array([1, 0])  # This is |0> vector state
        self.One = np.array([0, 1])  # This is |1> vector state

        #gates
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

        # produce binary digits for 2 input gates
        self.binary = self.produce_digits()

    # def Tensor_Prod(self, Q1, Q2):
    #     print(f"h{len(Q1)}")
    #     #IMPORTANT: Tensor product multiples the values of Q1 with the matrix Q2
    #     tensorprod = []
    #     for x in np.nditer(Q1): #iterate x over Q1
    #         tensorprod = np.append(tensorprod, x * Q2) #appends tensorprod with x'th value of Q1 * (matrix) Q2
    #     tensorprod = np.asmatrix(tensorprod)
    #     #ouput is linear tensor product (NOTE: matrix form infromation lost)
    #
    #     return tensorprod
    #

    def Tensor_Prod(self, Q1, Q2): #rewritten by Emma
        R = []
        for i in range(len(Q1)):
            R.append(Q1[i][0]*Q2)
            for j in range(1, len(Q1[i])):
                R[i] = np.concatenate((R[i], (Q1[i][j]*Q2)), axis=1)


        C = R[0]
        for i in range(1, len(Q1)):
            C = np.concatenate((C, R[i]), axis = 0)

        return C

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

        # this is only one step. so only one logic gate can be applied to a single Qubit. so must return an error
        # if any value within the position sub-lists is repeated
        list_check = [] #create a unique list with the each of the arguments of the sublists of positions
        for k in range(len(positions)):
            for i in range(len(positions[k])):
                list_check.append((positions[k])[i])

        assert len(list(list_check)) == len(set(list_check)), "repeated position value"  #ensure lenght of list is equal to lenght of unique values in list i.e. avoid repetiotion

        single_gate_inputs = ["H", "RNot", "Phase", "X", "Y", "Z", "T"] #maps the string input to the relevant matrix and creates an array
        matrices = [self.Hadamard, self.RNot, self.Phase, self.X, self.Y, self.Z, self.T]
        M = []
        for j in range(len(gate)):
            for i in range(len(single_gate_inputs)):
                if str(gate[j]) == str(single_gate_inputs[i]):
                    M.append(matrices[i])
        assert len(M) > 0, ("Please enter one of the following gates and ensure correct spelling: H, RNot, Phase, X, Y, Z, T")

        L = self.I
        for i in range(len(positions)):
            for j in range(len(positions[i])):
                if positions[i][j] == 0:
                    L = M[i] #and else L = L

        for l in range(1, self.Register_Size):
            size1 = 1
            for dim in np.shape(L): size1 *= dim

            for i in range(len(positions)):
                for j in range(len(positions[i])):
                    if l == positions[i][j]:
                        L = self.Tensor_Prod(L, M[i])

                size2 = 1
                for dim in np.shape(L): size2 *= dim

                if size1 == size2:
                    L = self.Tensor_Prod(L, self.I)

        L.shape = (2**self.Register_Size, 2**self.Register_Size)

        return np.asmatrix(L)

    #normalisation function (if needed?)
    def Norm(self, array):
        total = 0
        for i in range(0, len(array)):
            total += np.sqrt(np.real(array[i]) ** 2 + np.imag(array[i]) ** 2)

        array /= total

        return array

    def recog_digits(self, digits):
        N = self.Register_Size
        numbers = []

        for i in range(0, 2 ** N):
            num = 0
            for j in range(0, N):
                num += 2 ** (N - j - 1) * digits[i][j]
            numbers.append(num)

        return numbers

    def produce_digits(self):
        N = self.Register_Size
        digits = []
        for i in range(0, 2 ** N):
            digit = []
            if i < (2 ** N) / 2:
                digit.append(0)
            else:
                digit.append(1)
            for j in range(1, N):
                x = i
                for k in range(0, len(digit)):
                    x -= digit[k] * (2 ** N) / (2 ** (k + 1))
                if x < (2 ** N) / (2 ** (j + 1)):
                    digit.append(0)
                else:
                    digit.append(1)
            digits.append(digit)
        return digits


    def CNot(self, c, t):  # c is the position of the control qubit, t is the position of the target qubit
        N = self.Register_Size

        cn = []
        digits = self.binary

        for i in range(0, 2 ** N):
            if digits[i][c] == 1:
                digits[i][t] = 1 - digits[i][t] % 2

        index = self.recog_digits(digits)

        for i in range(0, 2 ** N):
            new_row = self.Q[index[i]]
            new_row.shape = (1, 2 ** N)
            cn.append(new_row)

        cn = np.asmatrix(np.asarray(cn))

        return cn

    def CV(self, c, t):  # c is the position of the control qubit, t is the position of the target qubit
        N = self.Register_Size

        cv = []
        digits = self.binary

        for i in range(0, 2 ** N):
            if digits[i][c] == 1 and digits[i][t] == 1:
                new_row = 1j * self.Q[i]
            else:
                new_row = self.Q[i]
            new_row.shape = (1, 2 ** N)
            cv.append(new_row)

        cv = np.asmatrix(np.asarray(cv))

        return cv


    def gate_logic(self, timesteps, gates, positions):
        '''
        - param gate: list of gate names to be applied
        - param positions: list of lists. each entry corresponds to the respective gate
        and contains a list of qubit position(s) on which to apply that gate
        '''
        assert len(gates) == timesteps and len(positions) == timesteps, "error"

#tesing Single_Logic
Q = Quantum_Computer(2)

gate = ["H"]
positions = [[1]]
# print(Q.Single_Logic(gate, positions))


I = np.array([[1,0],[0,1]]) #Identity gate
# M = np.asmatrix((Q.Tensor_Prod(I, I)))
# M.shape = (4,4)
#
# M2 = np.asmatrix((Q.Tensor_Prod(M, 2*I)))
# M2.shape = (8,8)
#
# print(M2)

M3 = Q.Tensor_Prod_Emma(I, I)
M4 = Q.Tensor_Prod_Emma(M3, I)

H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
M5 = Q.Tensor_Prod_Emma(H, H)
print(M5)





