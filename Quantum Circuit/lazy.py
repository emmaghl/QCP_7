import numpy as np

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








# example of how to use:
I = LazyMatrix('I')
H = LazyMatrix('H')

IH = LazyMatrix.tensor_prod(I, H)

IIH = LazyMatrix.tensor_prod(I, IH)

HII = LazyMatrix.tensor_prod(H, LazyMatrix.tensor_prod(I, I))

mat = LazyMatrix.matrix_multiply(HII, IIH)

# tests
inputs1 = [1, 2, 3, 5]
inputs2 = [1, 2, 3, 5, 7, 11, 13, 17]

print(IH.output(inputs1))
print(mat.output(inputs2))