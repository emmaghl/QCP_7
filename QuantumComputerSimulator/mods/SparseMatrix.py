from QuantumComputerSimulator.mods.MatrixFrame import MatrixFrame

import math
import numpy as np

class SparseMatrix(MatrixFrame):

    def __init__(self, Type: str, *args):
        '''
        Sets up gates initally in the dense method then condenses into sparse matrices.
        :param Type: Take in the gate to be built
        :param args:
        '''
        if Type == 'H':  # hadamard gate
            self.matrix = 1 / math.sqrt(2) * np.array([0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, -1])
        if Type == 'I':  # identity gate
            self.matrix = np.array([0, 0, 1], [1, 1, 1])
        if Type == 'P':
            self.matrix = np.array([0, 0, 1], [1, 0, 1], [1,1,np.exp(1j * args[0])])
        if Type == 'X':
            self.matrix = np.array([0, 1, 1], [1, 0, 1])
        if Type == 'Y':
            self.matrix = np.array([[0,1, 0 - 1j], [1,0,0 + 1j]], dtype=complex)
        if Type == 'Z':
            self.matrix = np.array([0,0,1], [1,1,-1])
        if Type == 'TP' or Type == 'MM': #check that the matrix in args[0] is sparse
            self.matrix = args[0]

        if Type == 'CNOT': #check that the matrix in args[0] is sparse
            self.matrix = self.cnot(args[0], args[1], args[2])
        if Type == 'CV':  #check that the matrix in args[0] is sparse
            self.matrix = self.cv(args[0], args[1], args[2])
        if Type == 'CZ': #check that the matrix in args[0] is sparse
            self.matrix = self.cz(args[0], args[1], args[2])

        if Type == 'M0':
            self.matrix = np.array([0,0,1], [1,1,0])
        if Type == 'M1':
            self.matrix = np.array([1,1,1])

        self.dim = self.size_matrix()[0]

    def size_matrix(self):
        '''
        Gives the dimensions of the matrix. Used to preserve matrix structure information.
        :return:
        '''
        ncol = self.matrix[-1][0] + 1  # number of columns is the column value of the last entry in the sparse matrix
        nr = 0  # number of rows is the maximum row value across the array (+1 because of Python indexing)
        for j in range(len(self.matrix)):
            if self.matrix[j][1] > nr:
                nr = self.matrix[j][1]
        nrow = nr + 1
        return (ncol, nrow)


    @classmethod
    def tensor_prod(self, m1, m2):
        '''
        Preform a tensor product between two matrices
        :param m1: Matrix 1
        :param m2: Matrix 2
        :return: Tensor product of Matrix 1 with Matrix 2
        '''
        m2_col = self.size_matrix(m2)[0]  # STcol/SM1col = SM2col etc.
        m2_row = self.size_matrix(m2)[1]

        tensorprod = []
        for j in range(len(m1)):
            for i in range(len(m1)):
                column = m2_col * m1[j][0] + m2[i][0]
                row = m2_row * m1[j][1] + m2[i][1]
                value = m1[j][2] * m2[i][2]
                tensorprod.append([column, row, value])

        return tensorprod

    @classmethod
    def matrix_multiply(cls, m1, m2):
        '''
        Multiply two matrices
        :param m1: Matrix 1
        :param m2: Matrix 2
        :return: Matrix 1 multiplied by Matrix 2
        '''
        # Convert SM1 and SM2 to a dictionaries with (row, col) keys and values for matrix manipulation when adding terms for matrix multiplication
        dict1 = {(row, col): val for row, col, val in m1}
        dict2 = {(row, c): v for row, c, v in m2}

        dictm = {}
        for (r1, c1), v1 in dict1.items():  # iterate over SM1
            for (r2, c2), v2 in dict2.items():  # and SM2
                if c1 == r2:  # when the coloumn entry of SM1 and row entry of SM2 match, this is included in the non-zero terms for the matmul matrix
                    dictm[(r1, c2)] = dictm.get((r1, c2),
                                              0) + v1 * v2  # there may be more non-zero adding terms for each item in the matmul so the dictionary takes care of that

        matmul = [[r, c, v] for (r, c), v in dictm.items()]  # return in sparse matric form
        return matmul

    def transpose(M):
        '''
        Method to transpose a sparse matrix
        :return: Matrix transposed
        '''
        M_transpose = M.copy()
        print(M_transpose)
        for i in range(len(M)):
            row, column, entry = M[i]
            M_transpose[i] = [column, row, entry]
        return M_transpose

    @classmethod
    def inner_prod(cls, M):
        '''
        Inner product of matrix M
        :param M: input matrix
        :return: Transpose
        '''
        return cls.matrix_multiply(M.matrix, cls.transpose(np.conj(M.matrix)))

    @classmethod
    def trace(cls, M):
        '''
        Trace of a sparse matrix.
        :param M: Input Matrix
        :return: Transpose of matrix
        '''
        trace = 0
        for i in range(len(M)):
            for j in range(cls.size_matrix()[1]): #number of columns
                if M[i][0] == j and M[i][1] == j:
                    trace += M[i][2]
        return trace

    def Basis(self, N): # need to check it's doing what i want it to
        '''
        Builds the sparse basis.
        :param N:
        :return: Returns the basis in the form of a list
        '''
        Q = []
        for i in range(0, 2 ** N):
            Q.append([i,0,1])
            if i != 2**N - 1:
                Q.append([2**N - 1,0,0])
        return Q

    def cnot(self, d, c, t):
        '''
        Inherits from MatrixFrame to produce a CNOT gate.
        :param d:
        :param c:
        :param t:
        :return:
        '''
        digits = copy.deepcopy(d)
        cn = []

        index = super().CNOT_logic(digits, c, t)
        N = int(np.log(len(index)) / np.log(2))
        basis = self.Basis(N)

        for i in range(0, 2 ** N):
            new_row_ascolumn = basis[index[i]]
            new_row = self.transpose(new_row_ascolumn)
            cn.append(new_row)

        return cn

    def cv(self, d, c, t):
        pass

    def cz(self, d, c, t):
        pass

    def output(self, inputs):
        '''
        Output of sparse matrix class.
        :param inputs:
        :return:
        '''
        return self.matrix_multiply(self.matrix, inputs)