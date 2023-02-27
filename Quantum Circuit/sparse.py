from PrintingCircuit import PrintingCircuit
import numpy as np
import time

class Sparse_Quantum_Computer:

    def __init__(self, Qubits):
        """! The Quantum_Computer class initializer.
            @param Qubits: the number of Qubits in the quantum register
            @return  An instance of the Quantum_Computer class initialized with the specified number of qubits
        """

        self.Register_Size = Qubits

        # computational basis
        self.Zero = np.array([1, 0])  # This is |0> vector state
        self.One = np.array([0, 1])  # This is |1> vector state

        # # produce basis and quantum register
        # self.Basis()
        self.Sparse_Basis()
        # self.Q_Register()
        #
        # gates
        self.I = np.array([[1, 0], [0, 1]])  # Identity gate
        self.Hadamard = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])  # sends |0> to |+> and |1> to |->
        self.Phase = np.array([[1, 0], [1, 0 + 1j]], dtype=complex)  # sends |0>+|1> to |0>+i|1>

        # more gates (unused as of 16/02)
        self.X = np.array([[0, 1], [1, 0]])  # Flips the |0> to |1> and vice versa
        self.Y = np.array([[0, 0 + 1j], [0 - 1j, 0]], dtype=complex)  # converts |0> to i|1> and |1> to -i|0>
        self.Z = np.array([[1, 0], [0, -1]])  # sends |1> to -|1> and |0> to |0>
        self.RNot = 1 / np.sqrt(2) * np.array(
            [[1, -1], [1, 1]])  # sends |0> to 0.5^(-0.5)(|0>+|1>) and |1> to 0.5^(-0.5)(|1>-|0>)
        self.T = np.array([[1, 0], [0, 1 / np.sqrt(2) * (1 + 1j)]],
                          dtype=complex)  # square root of phase (rotates by pi/8)
        self.CNot = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])  # reversable xor: |00> -> |00>, |01> -> |11>
        self.Swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])  # ¯\_(ツ)_/¯
        #
        self.single_inputs = ["H", "RNot", "P", "X", "Y", "Z",
                                   "T"]  # maps the string input to the relevant matrix and creates an array
        self.matrices = [self.Hadamard, self.RNot, self.Phase, self.X, self.Y, self.Z, self.T]
        #
        # self.double_inputs = ["CV", "CNOT"]
        #
        # produce binary digits for 2 input gates
        self.binary = self.produce_digits()
        #
        # # Keeps track of the gates that we've added via Gate Logic
        # self.gate_histroy = []


    def Dense_to_Sparse(self, Matrix):  # defines a sparse matrix of the form row i column j has value {}
        """! What the class/method does
            @param list the parameters and what they do
            @return  what the function returns
        """
        rows = np.shape(Matrix)[0]
        cols = np.shape(Matrix)[1]
        SMatrix = []  # output matrix
        for i in range(rows):
            for j in range(cols):
                if Matrix[i][j] != 0:  # if the value of the matrix element i,j is not 0 then store the value and the location
                    SMatrix.append([i, j, Matrix[i][j]])  # Output array: (row, column, value)
        return SMatrix  # return output
        #comment how this is okay because last and first row always have a non zero element - why do i say this tho?

    def Size_Sparse(self, SMatrix):
        ncol = SMatrix[-1][0]+1 #number of columns is the coloumn value of the last entry in the sparse matrix

        nr = 0 #number of rows is the maximum row value across the array (+1 because of Python indexing)
        for j in range(len(SMatrix)):
            if SMatrix[j][1] > nr:
                nr = SMatrix[j][1]
        nrow = nr+1

        #following takes longer to compute but doesn't require sparse matrix to be in set order:
        '''
        nc = 0 #number of rows is the maximum row value across the array (+1 because of Python indexing)
        for j in range(len(SMatrix)):
            if SMatrix[j][0] > nc:
                nc = SMatrix[j][0]
        ncol = nc+1
        '''

        return(ncol, nrow)

    def Sparse_to_Dense(self, SMatrix):
        """! Takes in a sparse matrix and returns the corresponding dense matrix.
            Note: suppose you're converting a dense matrix to sparse and back to dense,
            if the last row(s) and/or coloumn(s) of the original dense matrix are all zero entries,
            these will be lost in the sparse conversion.
             @param Matrix: a sparse matrix: an array of triples [a,b,c] where a is the row, b is the colomn and c is the non-zero value
             @return  DMatrix: the converted dense matrix (in array form)
         """
        count = 0
        for row in SMatrix:
            if type(row[2]) == "complex": #check correct synatx!!
                count += 1
        if count == 0:
            typex = "int"
        if count == 1:
            typex = "complex"


        DMatrix = np.zeros((self.Size_Sparse(SMatrix)[0])*self.Size_Sparse(SMatrix)[1], dtype=typex) #create an array of zeros of the right size
        DMatrix.shape = self.Size_Sparse(SMatrix)
        for j in range(len(SMatrix)): #iterate over each row of the sparse matrix
            DMatrix[SMatrix[j][0]][SMatrix[j][1]] = (SMatrix[j][2]) #change the non zero entries of the dense matrix
        return DMatrix

    def Sparse_Tensor(self, SM1, SM2):
        # STcol = self.Size_Sparse(SM1)[0] * self.Size_Sparse(SM2)[0]
        # STrow = self.Size_Sparse(SM1)[1] * self.Size_Sparse(SM2)[1]
        #
        # SM1col = self.Size_Sparse(SM1)[0]
        # SM21row = self.Size_Sparse(SM1)[1]

        SM2col = self.Size_Sparse(SM2)[0] #STcol/SM1col = SM2col etc.
        SM2row = self.Size_Sparse(SM2)[1]

        STensor = []
        for j in range(len(SM1)):
            for i in range(len(SM2)):
                column = SM2col * SM1[j][0] + SM2[i][0]
                row = SM2row * SM1[j][1] + SM2[i][1]
                value = SM1[j][2] * SM2[i][2]
                STensor.append([column, row, value])

        return STensor

    def Sparse_MatMul(self, SM1, SM2):
        # Convert SM1 and SM2 to a dictionaries with (row, col) keys and values for matrix manipulation when adding terms for matrix multiplication
        dict1 = {(row, col): val for row, col, val in SM1}
        dict2 = {(row, col): val for row, c, v in SM2}

        Sdict = {}
        for (r1, c1), v1 in dict1.items(): #iterate over SM1
            for (r2, c2), v2 in dict2.items(): #and SM2
                if c1 == r2: #when the coloumn entry of SM1 and row entry of SM2 match, this is included in the non-zero terms for the matmul matrix
                    Sdict[(r1, c2)] = Sdict.get((r1, c2), 0) + v1 * v2 #there may be more non-zero adding terms for each item in the matmul so the dictionary takes care of that

        SMatMul = [[r, c, v] for (r, c), v in Sdict.items()] #return in sparse matric form
        return SMatMul

    def Q_Register(self):
        """! What the class/method does
            @param list the parameters and what they do
            @return  what the function returns
        """
        # returns an array of 2**n complex coefficients and ensures normalisation.
        j = 2 ** self.Register_Size
        coeffs = (0 + 0 * 1j) * np.zeros(j)
        for i in range(j):
            theta = np.random.random() * np.pi * 2
            coeffs[i] = (np.cos(theta) + np.sin(theta) * 1j) / j

        self.psi = coeffs
        self.psi.shape = (j, 1)


    def Sparse_Basis(self):
        N = self.Register_Size
        self.Qs = []

        for i in range(0, 2 ** N):
            self.Qs.append([i,0,1])


    def Single_Gates(self, gate, qnum):
        """! What the class/method does
            @param list the parameters and what they do
            @return  what the function returns
        """
        M = [0] * self.Register_Size

        for i in range(0, len(self.single_inputs)):
            for j in range(0, len(gate)):
                if self.single_inputs[i] == gate[j]:
                    for k in range(0, len(qnum[j])):
                        M[qnum[j][k]] = self.matrices[i]

        for i in range(0, len(M)):
            if type(M[i]) != np.ndarray:
                M[i] = self.I

        m = self.Dense_to_Sparse(M[0])
        for i in range(1, len(M)):
            m = self.Sparse_Tensor(m, self.Dense_to_Sparse(M[i]))
        return m

    def __recog_digits(self, digits):
        """! What the class/method does
            @param list the parameters and what they do
            @return  what the function returns
        """
        N = self.Register_Size
        numbers = []

        for i in range(0, 2 ** N):
            num = 0
            for j in range(0, N):
                num += 2 ** (N - j - 1) * digits[i][j]
            numbers.append(num)

        return numbers

    def produce_digits(self):
        """! What the class/method does
            @param list the parameters and what they do
            @return  what the function returns
        """
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


    def Sparse_CNOT(self, c, t):  # c is the position of the control qubit, t is the position of the target qubit
        """! What the class/method does
            @param list the parameters and what they do
            @return  what the function returns
        """
        N = self.Register_Size

        cn = []
        digits = self.binary

        for i in range(0, 2 ** N):
            if digits[i][c] == 1:
                digits[i][t] = 1 - digits[i][t] % 2

        index = self.__recog_digits(digits)

        for i in range(0, 2 ** N):
            cn.append([i, self.Qs[index[i]][0], 1])
            # new_row = [self.Qs[index[i]][1], self.Qs[index[i]][0], self.Qs[index[i]][2]] #change to row vector
            # cn.append(new_row)

        return cn

    def Sparse_CV(self, c, t):  # c is the position of the control qubit, t is the position of the target qubit
        """! What the class/method does
            @param list the parameters and what they do
            @return  what the function returns
        """
        N = self.Register_Size

        cv = []
        digits = self.binary

        for i in range(0, 2 ** N):
            if digits[i][c] == 1 and digits[i][t] == 1:
                new_row = [self.Qs[i][0], self.Qs[i][0], self.Qs[i][2]*1j]
            else:
                new_row = [self.Qs[i][0], self.Qs[i][0], self.Qs[i][2]]
            cv.append(new_row)

        return cv

    def Sparse_CZ(self, c, t):
        N = self.Register_Size

        cv = []
        digits = self.binary

        for i in range(0, 2 ** N):
            if digits[i][c] == 1 and digits[i][t] == 1:
                new_row = [self.Qs[i][0], self.Qs[i][0], self.Qs[i][2]*-1]
            else:
                new_row = [self.Qs[i][0], self.Qs[i][0], self.Qs[i][2]]
            cv.append(new_row)

        return cv

    def __Double_Gates(self, gate, qnum):
        """! What the class/method does
            @param list the parameters and what they do
            @return  what the function returns
        """

        if gate[0] == "CV":
            return self.Sparse_CV(qnum[0][0], qnum[0][1])
        if gate[0] == "CNOT":
            return self.Sparse_CNOT(qnum[0][0], qnum[0][1])
        if gate[0] == "CZ":
            return self.Sparse_CZ(qnum[0][0], qnum[0][1])

    def Sparse_Gate_Logic(self, inputs): #still to be tested
        """! What the class/method does
            @param list the parameters and what they do
            @return  what the function returns
        """
        N = self.Register_Size
        step_n = len(inputs)

        # Add the gates to the gate history for printing later.
        [self.gate_histroy.append(i) for i in inputs]

        M = []

        for i in range(0, step_n):
            for j in range(0, len(self.single_gate_inputs)):
                if self.single_gate_inputs[j] in inputs[i][0]:
                    M.append(self.__Single_Gates(inputs[i][0], inputs[i][1]))
            for j in range(0, len(self.double_inputs)):
                if self.double_inputs[j] in inputs[i][0]:
                    M.append(self.__Double_Gates(inputs[i][0], inputs[i][1]))

        m = M[0]
        for i in range(1, len(M)):
            m = self.Sparse_MatMul(m, M[i])

        return m

    def Check_Same(self, Q1, Q2):
        assert Q1.shape == Q2.shape, "different shapes"

        check = 0
        for i in range(len(Q1)):
            for j in range(len(Q1[i])):
                if Q1[i][j] != Q2[i][j]:
                    check += 1

        if check == 0:
            print("matrices match")
        else:
            print("matrices DO NOT match")


comp = Sparse_Quantum_Computer(3)
A = comp.Sparse_CZ(0,2)
B = comp.Sparse_to_Dense(A)

print(B)


# # comp.Check_Same(comp.Basis, comp.Sparse_to_Dense(comp.Sparse_Basis()))
# t1 = time.time()
# A = comp.Sparse_CNOT(1,2)
# t2 = time.time()
# B = comp.Sparse_to_Dense(A)
# print(t2-t1)
#
from QuantumComputer import Quantum_Computer

compdense = Quantum_Computer(3)
C = compdense.CZ(0,2)
print(C)

#
# comp.Check_Same(B, C)


# t3 = time.time()
# C = np.asarray(compdense.CNOT(1,2))
# t4 = time.time()
# print(t4-t3)
# # print(C[0][1])
# comp.Check_Same(B, C)

# print(comp.produce_digits())
# print(comp.CNOT(1,2))

# print(comp.Dense_to_Sparse(comp.CNOT(1,2)))
# print(comp.Dense_to_Sparse(comp.CNOT(1,2)))

# print(comp.Sparse_CNOT(1,2))
# print(comp.Sparse_to_Dense(comp.Sparse_CNOT(1,2)))

# t1 = time.time()
# # print(comp.Dense_to_Sparse(comp.Q))
# single = comp.Single_Gates(["H"], [[1, 3]])
#
# Q1 = comp.Sparse_to_Dense(single)
# t2 = time.time()
# print(t2-t1)
# from QuantumComputer import Quantum_Computer
#
#
# compdense = Quantum_Computer(12)
# t3 = time.time()
# Q2 = compdense.Single_Gates(["H"], [[1, 3]])
# t4 = time.time()
# print(t4-t3)
# comp.Check_Same(Q1, Q2)
#





# SM1 = [[1,1,3], [2,1,1]]
# SM2 = [[0,0,2], [1,1,1]]
#
# M1 = comp.Sparse_to_Dense(SM1)
# M2 = comp.Sparse_to_Dense(SM2)
#
# MM = comp.Mat_Mul(M1, M2)
# print(comp.Dense_to_Sparse(MM))
#
# SM = comp.Sparse_MatMul(SM1, SM2)
# print(SM)


# print(comp.Sparse_to_Dense(SM1))
# print(comp.Sparse_to_Dense(SM2))
#
# t = comp.Sparse_Tensor(SM1, SM2)
# print(t)
# print(comp.Sparse_to_Dense(t))
#
#
#
# # mat = [[0,0,0], [0,0,0], [0,0,1]]
# # smat = comp.Dense_to_Sparse(mat)
# # print(mat)
# # print(smat)
# # print(f'len{len(smat)}')
# # dmat = comp.Sparse_to_Dense(smat)
# # print(dmat)