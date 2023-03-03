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
        pass


    @classmethod
    def matrix_multiply(cls, m1, m2):
        pass


    def output(self, inputs):
        pass

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