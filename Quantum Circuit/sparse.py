from PrintingCircuit import PrintingCircuit
import numpy as np


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
        # self.Q_Register()
        #
        # # gates
        # self.I = np.array([[1, 0], [0, 1]])  # Identity gate
        # self.Hadamard = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])  # sends |0> to |+> and |1> to |->
        # self.Phase = np.array([[1, 0], [1, 0 + 1j]], dtype=complex)  # sends |0>+|1> to |0>+i|1>
        #
        # # more gates (unused as of 16/02)
        # self.X = np.array([[0, 1], [1, 0]])  # Flips the |0> to |1> and vice versa
        # self.Y = np.array([[0, 0 + 1j], [0 - 1j, 0]], dtype=complex)  # converts |0> to i|1> and |1> to -i|0>
        # self.Z = np.array([[1, 0], [0, -1]])  # sends |1> to -|1> and |0> to |0>
        # self.RNot = 1 / np.sqrt(2) * np.array(
        #     [[1, -1], [1, 1]])  # sends |0> to 0.5^(-0.5)(|0>+|1>) and |1> to 0.5^(-0.5)(|1>-|0>)
        # self.T = np.array([[1, 0], [0, 1 / np.sqrt(2) * (1 + 1j)]],
        #                   dtype=complex)  # square root of phase (rotates by pi/8)
        # self.CNot = np.array(
        #     [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])  # reversable xor: |00> -> |00>, |01> -> |11>
        # self.Swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])  # ¯\_(ツ)_/¯
        #
        # self.single_gate_inputs = ["H", "RNot", "P", "X", "Y", "Z",
        #                            "T"]  # maps the string input to the relevant matrix and creates an array
        # self.matrices = [self.Hadamard, self.RNot, self.Phase, self.X, self.Y, self.Z, self.T]
        #
        # self.double_inputs = ["CV", "CNOT"]
        #
        # # produce binary digits for 2 input gates
        # self.binary = self.produce_digits()
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
        DMatrix = np.zeros(self.Size_Sparse(SMatrix)[0]*self.Size_Sparse(SMatrix)[1]) #create an array of zeros of the right size
        DMatrix.shape = self.Size_Sparse(SMatrix)
        for j in range(len(SMatrix)): #iterate over each row of the sparse matrix
            DMatrix[SMatrix[j][0]][SMatrix[j][1]] = SMatrix[j][2] #change the non zero entries of the dense matrix
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

    def Mat_Mul(self, Q1, Q2):
        """! What the class/method does
            @param list the parameters and what they do
            @return  what the function returns
        """
        assert np.shape(Q1)[1] == np.shape(Q2)[0], "can't perform matrix multiplication"
        M = np.zeros(len(Q1) * len(Q2[0]))
        M.shape = (len(Q1), len(Q2[0]))

        for i in range(len(Q1)):  # rows of Q1
            for j in range(len(Q2[0])):  # columns of Q2
                for k in range(len(Q2)):  # rows of Q2
                    M[i][j] += Q1[i][k] * Q2[k][j]

        return M


# comp = Sparse_Quantum_Computer(2)
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