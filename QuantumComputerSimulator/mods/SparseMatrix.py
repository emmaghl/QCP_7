from QuantumComputerSimulator.mods.MatrixFrame import MatrixFrame

import math
import numpy as np
import copy

class SparseMatrix(MatrixFrame):

    def __init__(self, Type: str, *args):
        '''
        Implements the Sparse method for quantum computing simulation.

        <b>Type</b> Gate to be built. <br>
        <b>*args</b> Position of control and target qubits.
        '''
        if Type == 'H':  # hadamard gate
            self.matrix = np.array([[0, 0, 1 / math.sqrt(2)], [0, 1, 1 / math.sqrt(2)], [1, 0, 1 / math.sqrt(2)], [1, 1, -1 / math.sqrt(2)]])
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
        #
        # if Type == 'TP' or Type == 'MM':
        #     self.matrix = args[0] #check that the matrix in args[0] is sparse
        if Type == "general":
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

        if Type == 'zerocol':
            self.matrix = np.array([[0,0,1], [1,0,0]])
        if Type == 'onecol':
            self.matrix = np.array([[1, 0, 1]])

        # print(f"before strip{self.matrix}")
        self.matrix = self.strip_matrix(self.matrix)
        # print(f"after strip{self.matrix}")

        self.size = SparseMatrix.size_matrix(self.matrix)

    def strip_matrix(self, mat): #future take care of real and imag separately to only strip half
        '''Controls what is meant by non-zero.'''
        del_entries = []
        for i in range(len(mat)):
            if np.abs(np.real(mat[i][2])) >= 10 ** (-10) or np.abs(np.imag(mat[i][2])) >= 10 ** (-10) or i == len(mat)-1:
                del_entries.append(i)
        # print(del_entries)

        mat_strip = []
        for i in range(len(mat)):
            for j in del_entries:
                if i == j:
                    mat_strip.append(mat[i])

        return mat_strip

    @classmethod
    def size_matrix(cls, M):
        '''
        Gives the dimensions of the matrix. Used to preserve matrix structure information.
        '''
        if type(M) == SparseMatrix:
            m = M.matrix
        else:
            m = M

        # ncol = M[-1][0] + 1  # number of columns is the column value of the last entry in the sparse matrix
        nc = 0  # number of rows is the maximum row value across the array (+1 because of Python indexing)
        for j in range(len(m)):
            if m[j][0] > nc:
                nc = m[j][0]
        ncol = nc + 1

        nr = 0  # number of rows is the maximum row value across the array (+1 because of Python indexing)
        for j in range(len(m)):
            if m[j][1] > nr:
                nr = m[j][1]
        nrow = nr + 1
        # print(int(ncol), int(nrow))
        return (int(np.real(ncol)), int(np.real(nrow)))

    @classmethod
    def quantum_register(cls, qnum):
        '''Initialises quantum register.'''
        register = np.array([[0, 0, 1], [0, 1, 0]])
        w = 2 ** (qnum) - 2
        for i in range(w):
            register[-1] = [0, i + 2, 0]

        register = SparseMatrix("general", register)
        register.matrix = SparseMatrix.transpose(register.matrix)
        return register

    @classmethod
    def tensor_prod(cls, M2, M1):
        if type(M1) == SparseMatrix:
            m1 = M1.matrix
        else:
            m1 = M1

        if type(M2) == SparseMatrix:
            m2 = M2.matrix
            m2_col = M2.size[0]
            m2_row = M2.size[1]
        else:
            m2 = M2
            m2_col = SparseMatrix.size_matrix(m2)[0]  # STcol/SM1col = SM2col etc.
            m2_row = SparseMatrix.size_matrix(m2)[1]  # STcol/SM1col = SM2col etc.

        
        tensorprod = []

        for j in range(len(m1)):
            for i in range(len(m2)):
                if 1>0:
                    column = m2_col * m1[j][0] + m2[i][0]
                    row = m2_row * m1[j][1] + m2[i][1]
                    value = m1[j][2] * m2[i][2]
                    tensorprod.append([column, row, value])

        column = m2_col * m1[-1][0] + m2[-1][0] #or use size etc..
        row = m2_row * m1[-1][1] + m2[-1][1]
        value = round(m1[-1][2] * m2[-1][2], 10)
        tensorprod.append([column, row, value])
        return SparseMatrix("general", tensorprod)

    @classmethod
    def matrix_multiply(cls, M1, M2):
        if type(M1) == SparseMatrix:
            m1 = M1.matrix
        else:
            m1 = M1
        if type(M2) == SparseMatrix:
            m2 = M2.matrix
        else:
            m2 = M2

        # Convert SM1 and SM2 to a dictionaries with (row, col) keys and values for matrix manipulation when adding terms for matrix multiplication
        dict1 = {(row, col): val for [row, col, val] in m1}
        dict2 = {(row, col): val for [row, col, val] in m2}

        dictm = {}

        for (r1, c1), v1 in dict1.items():  # iterate over SM1
            for (r2, c2), v2 in dict2.items():  # and SM2
                if c1 == r2:  # when the coloumn entry of SM1 and row entry of SM2 match, this is included in the non-zero terms for the matmul matrix
                    dictm[(r1, c2)] = dictm.get((r1, c2),0) + v1 * v2  # there may be more non-zero adding terms for each item in the matmul so the dictionary takes care of that
        matmul = [[r, c, v] for (r, c), v in dictm.items()]  # return in sparse matric form
        return SparseMatrix("general", matmul)

    def sparse_multiply(self, num: float, mat):
        # multiply matrix entry by a scalar and keep row and column information unchanged
        mul = []
        for i in range(len(mat)):
            mul.append([mat[i][0], mat[i][1], num * mat[i][2]])
        return mul

    @classmethod
    def transpose(cls, M):
        if type(M) == SparseMatrix:
            m = M.matrix
        else:
            m = M

        m_transpose = m.copy()
        for i in range(len(m)):
            row, column, entry = m[i]
            m_transpose[i] = [column, row, entry]
        return m_transpose

    @classmethod
    def inner_product(cls, M):
        if type(M) == SparseMatrix:
            m = M.matrix
        else:
            m = M

        return SparseMatrix.matrix_multiply(m, SparseMatrix.transpose(np.conj(m)))

    @classmethod
    def trace(cls, M):

        trace = 0

        if type(M) == SparseMatrix:
            m = M.matrix
            m_col = M.size[0]
        else:
            m = M
            m_col = SparseMatrix.size_matrix(m)[0]  # STcol/SM1col = SM2col etc.

        for i in range(len(m)):
            for j in range(m_col):  # number of columns
                if m[i][0] == j and m[i][1] == j:
                    trace += m[i][2]
        return trace

    @classmethod
    def conjugate(cls, M):
        M_conj = M.copy()
        for i in range(len(M)):
            if type(M[i][2]) == "complex":
                M_conj[i][2] = - M[i][2]
        return M_conj

    def cnot(self, d: list, c: float, t: float):
        '''
        CNot gate.
        '''
        digits = copy.deepcopy(d)
        cn = []

        index = super().CNOT_logic(digits, c, t)
        N = int(np.log(len(index)) / np.log(2))

        for i in range(0, 2 ** N):
            new_entry = [i, index[i], 1]
            cn.append(new_entry)

        return cn

    def cv(self, d:list, c:float, t:float):
        '''CV gate'''
        digits = copy.deepcopy(d)
        cv = []

        index = super().CV_logic(digits, c, t)
        N = int(np.log(len(index)) / np.log(2))

        for i in range(0, 2 ** N):
            if index[i] == 1:
                new_entry = [i, i, 1j]
            else:
                new_entry = [i, i, 1]
            cv.append(new_entry)

        return cv

    def cz(self, d:list, c:float, t:float):
        '''CZ gate'''
        digits = copy.deepcopy(d)
        cz = []

        index = super().CZ_logic(digits, c, t)
        N = int(np.log(len(index)) / np.log(2))

        for i in range(0, 2 ** N):
            if index[i] == 1:
                new_entry = [i, i, -1]
            else:
                new_entry = [i, i, 1]
            cz.append(new_entry)

        return cz

    def Sparse_to_Dense(self, SMatrix):
        '''
        Takes in a sparse matrix and returns the corresponding dense matrix.
            Note: suppose you're converting a dense matrix to sparse and back to dense,
            if the last row(s) and/or coloumn(s) of the original dense matrix are all zero entries,
            these will be lost in the sparse conversion.

        <b>Matrix</b> a sparse matrix: an array of triples [a,b,c] where a is the row, b is the colomn and c is the non-zero value <br>
        <b>return</b>  DMatrix: the converted dense matrix (in array form)
         '''
        count = 0
        for row in SMatrix:
            if type(row[2]) == "complex":  # check correct synatx!!
                count += 1
        if count == 0:
            typex = "int"
        if count == 1:
            typex = "complex"

        DMatrix = np.zeros((self.size_matrix(SMatrix)[0]) * self.size_matrix(SMatrix)[1],
                           dtype=np.complex)  # create an array of zeros of the right size
        DMatrix.shape = self.size_matrix(SMatrix)
        for j in range(len(SMatrix)):  # iterate over each row of the sparse matrix
            DMatrix[int(SMatrix[j][0])][int(SMatrix[j][1])] = (SMatrix[j][2])  # change the non zero entries of the dense matrix
        return DMatrix

    def Dense_to_Sparse(self, Matrix):  # defines a sparse matrix of the form row i column j has value {}
        rows = np.shape(Matrix)[0]
        cols = np.shape(Matrix)[1]
        SMatrix = []  # output matrix
        for i in range(rows):
            for j in range(cols):
                if np.abs(Matrix[i][
                              j]) > 0.01:  # if the value of the matrix element i,j is not 0 then store the value and the location
                    SMatrix.append([i, j, Matrix[i][j]])  # Output array: (row, column, value)
        if self.size_matrix(SMatrix)[0] < cols:
            SMatrix.append([0, cols, 0])

        if self.size_matrix(SMatrix)[1] < rows:
            SMatrix.append([rows, 0, 0])

        return SMatrix  # return output

    def output(self, inputs:np.array) -> np.array:
        '''
        Gives the output state once the register, given by `inputs`, is applied.
        '''
        # inputs = self.Dense_to_Sparse(inputs)
        inputs_sparse = []
        for i in range(len(inputs)):
            if inputs[i] != 0:
                inputs_sparse.append([i, 0, inputs[i]])
        sparse_outputs = SparseMatrix.matrix_multiply(self.matrix, inputs_sparse).matrix

        outputs_dense = np.zeros(len(inputs), dtype = np.complex)
        for j in range(len(inputs)):
            for i in range(len(sparse_outputs)):
                if sparse_outputs[i][0] == j:
                    outputs_dense[j] = sparse_outputs[i][2]

        #to vector form
        outputs_dense = np.array(outputs_dense)
        outputs_dense.shape = (len(outputs_dense), 1)


        return outputs_dense