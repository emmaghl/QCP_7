from QuantumComputerSimulator.mods.MatrixFrame import MatrixFrame

import math
import numpy as np
import copy

class SparseMatrix(MatrixFrame):

    def __init__(self, Type: str, *args):
        if Type == 'H':  # hadamard gate
            self.matrix = 1 / math.sqrt(2) * np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, -1]])
        if Type == 'I':  # identity gate
            self.matrix = np.array([[0, 0, 1], [1, 1, 1]])
        if Type == 'P': # phase shift gatel, where args[0] is the angle
            self.matrix = np.array([[0, 0, 1], [1, 0, 1], [1,1,np.exp(1j * args[0])]])
        if Type == 'X': # X pauli gate
            self.matrix = np.array([[0, 1, 1], [1, 0, 1]])
        if Type == 'Y': # Y pauli gate
            self.matrix = np.array([[0,1, 0 - 1j], [1,0,0 + 1j]], dtype=complex)
        if Type == 'Z': # Z pauli gate
            self.matrix = np.array([[0,0,1], [1,1,-1]])
        if Type == 'TP' or Type == 'MM':
            self.matrix = args[0] #check that the matrix in args[0] is sparse
        if Type == 'CNOT':
            self.matrix = self.cnot(args[0], args[1], args[2]) #check that the matrix in args[0] is sparse
        if Type == 'CV':
            self.matrix = self.cv(args[0], args[1], args[2])
        if Type == 'CZ':
            self.matrix = self.cz(args[0], args[1], args[2])
        if Type == 'M0':
            self.matrix = np.array([[0,0,1], [1,1,0]])
        if Type == 'M1':
            self.matrix = np.array([[1,1,1]])

        self.dim = self.size_matrix(self.matrix)[0]

    @classmethod
    def size_matrix(cls,M):
        ncol = M[-1][0] + 1  # number of columns is the column value of the last entry in the sparse matrix
        nr = 0  # number of rows is the maximum row value across the array (+1 because of Python indexing)
        for j in range(len(M)):
            if M[j][1] > nr:
                nr = M[j][1]
        nrow = nr + 1
        return (ncol, nrow)


    @classmethod
    def tensor_prod(cls, M1, M2):
        if type(M1) == SparseMatrix:
            m1 = M1.matrix
        else:
            m1 = M1
        if type(M2) == SparseMatrix:
            m2 = M2.matrix
        else:
            m2 = M2

        m2_col = cls.size_matrix(m2)[0]  # STcol/SM1col = SM2col etc.
        m2_row = cls.size_matrix(m2)[1]

        tensorprod = []
        for j in range(len(m1)):
            for i in range(len(m1)):
                column = m2_col * m1[j][0] + m2[i][0]
                row = m2_row * m1[j][1] + m2[i][1]
                value = m1[j][2] * m2[i][2]
                tensorprod.append([column, row, value])

        return tensorprod

    @classmethod
    def matrix_multiply(cls, M1, M2):
        if type(M1) == DenseMatrix:
            m1 = M1.matrix
        else:
            m1 = M1
        if type(M2) == DenseMatrix:
            m2 = M2.matrix
        else:
            m2 = M2
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

    def sparse_multiply(self, num: float, mat):
        '''multiplies a scalar by a sparse matrix'''
        # multiply matrix entry by a scalar and keep row and column information unchanged
        mul = mat
        for i in range(len(mat)):
            mul[i][2] = num*mat[i][2]
        return mul

    def transpose(self, M):
        M_transpose = M.copy()
        for i in range(len(M)):
            row, column, entry = M[i]
            M_transpose[i] = [column, row, entry]
        return M_transpose

    @classmethod
    def inner_prod(cls, M):
        return cls.matrix_multiply(M.matrix, cls.transpose(np.conj(M.matrix)))

    @classmethod
    def trace(cls, M):
        trace = 0
        for i in range(len(M)):
            for j in range(cls.size_matrix()[1]): #number of columns
                if M[i][0] == j and M[i][1] == j:
                    trace += M[i][2]
        return trace

    def Basis(self, N:float): # need to check it's doing what i want it to
        Q = []
        for i in range(0, 2 ** N):
            Q.append([i,0,1])
            if i != 2**N - 1:
                Q.append([2**N - 1,0,0])
        return Q

    def cnot(self, d:list, c:float, t:float):
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

    def cv(self, d:list, c:float, t:float):
        digits = copy.deepcopy(d)
        cv = []

        index = super().CV_logic(digits, c, t)
        N = int(np.log(len(index)) / np.log(2))
        basis = self.Basis(N)

        for i in range(0, 2 ** N):
            if index[i] == 1:
                new_row_ascolumn = self.sparse_multiply(1j, basis[i])
            else:
                new_row_ascolumn = basis[i]
            new_row = self.transpose(new_row_ascolumn)
            cv.append(new_row)

        return cv

    def cz(self, d:list, c:float, t:float):
        digits = copy.deepcopy(d)
        cz = []

        index = super().CZ_logic(digits, c, t)
        N = int(np.log(len(index)) / np.log(2))
        basis = self.Basis(N)

        for i in range(0, 2 ** N):
            if index[i] == 1:
                new_row_ascolumn = self.sparse_multiply(-1, basis[i])
            else:
                new_row_ascolumn = basis[i]
            new_row = self.transpose(new_row_ascolumn)
            cz.append(new_row)

        return cz

    def output(self, inputs:np.array) -> np.array:
        return self.matrix_multiply(self.matrix, inputs)