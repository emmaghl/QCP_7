import numpy as np


class LazyMatrix(object):

    def __init__(self, Type, *args):
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