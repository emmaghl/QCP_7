import numpy as np
import math

#defining dense and sparse matrices with the same philosopy used by logan in lazy

class SparseMatrix(object):
    def __init__(self, Type: str, *args):
        if Type == 'I': #identity gate
            self.matrix = np.array([0,0,1], [1,1,1])
        if Type == 'H': #hadamard gate
            self.matrix = 1 / math.sqrt(2) * np.array([0,0,1], [0,1,1], [1,0,1],[1,1,-1])
        if Type == 'TP' or Type == 'MM': #tensor product or matrix multiplication
            self.matrix = args[0] #'matrix' to be first argument fed into the operation
        self.dim = self.size_matrix()[0]

    def size_matrix(self):
        ncol = self.matrix[-1][0]+1 #number of columns is the coloumn value of the last entry in the sparse matrix
        nr = 0 #number of rows is the maximum row value across the array (+1 because of Python indexing)
        for j in range(len(self.matrix)):
            if self.matrix[j][1] > nr:
                nr = self.matrix[j][1]
        nrow = nr+1
        return (ncol, nrow)

    @classmethod
    def tensor_prod(cls, m1, m2):
        m2_col = self.size_matrix(m2)[0] #STcol/SM1col = SM2col etc.
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
        # Convert SM1 and SM2 to a dictionaries with (row, col) keys and values for matrix manipulation when adding terms for matrix multiplication
        dict1 = {(row, col): val for row, col, val in m1}
        dict2 = {(row, col): val for row, c, v in m2}

        dict = {}
        for (r1, c1), v1 in dict1.items(): #iterate over SM1
            for (r2, c2), v2 in dict2.items(): #and SM2
                if c1 == r2: #when the coloumn entry of SM1 and row entry of SM2 match, this is included in the non-zero terms for the matmul matrix
                    dict[(r1, c2)] = dict.get((r1, c2), 0) + v1 * v2 #there may be more non-zero adding terms for each item in the matmul so the dictionary takes care of that

        matmul = [[r, c, v] for (r, c), v in dict.items()] #return in sparse matric form
        return matmul


    def output(self, inputs):
        return matrix_multiply(self.matrix, inputs)


class LazyMatrix(object):

    def __init__(self, Type: str, *args):
        if Type == 'I':
            self.matrix = [lambda x: x[0], lambda x: x[1]]
        if Type == 'H':
            self.matrix = [lambda x: (x[0] + x[1]) / np.sqrt(2), lambda x: (x[0] - x[1]) / np.sqrt(2)]
        if Type == 'TP' or Type == 'MM':
            self.matrix = args[0]
        self.dim = len(self.matrix)

    @classmethod
    def tensor_prod(cls, m1, m2):
        tp = []
        for i in range(0, m1.dim):
            for j in range(0, m2.dim):
                tp.append(lambda x, y=i, z=j: m1.matrix[y](
                    [m2.matrix[z]([x[m2.dim * k + l] for l in range(0, m2.dim)]) for k in range(0, m1.dim)]))

        new_matrix = LazyMatrix('TP', tp)

        return new_matrix

    @classmethod
    def matrix_multiply(cls, m1, m2):
        mm = []
        for i in range(0, m1.dim):
            mm.append(
                lambda x, y=i: m1.matrix[y]([m2.matrix[k]([x[l] for l in range(0, m2.dim)]) for k in range(0, m1.dim)]))

        new_matrix = LazyMatrix('MM', mm)

        return new_matrix

    def output(self, inputs):
        out = []
        for i in range(0, self.dim):
            out.append(self.matrix[i](inputs))

        return out