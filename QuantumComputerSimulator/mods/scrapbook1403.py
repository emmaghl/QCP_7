def Basis(N: float):  # need to check it's doing what i want it to
    Q = []
    for i in range(0, 2 ** N):
        Q.append([i, 0, 1])
        if i != 2 ** N - 1:
            Q.append([2 ** N - 1, 0, 0])
    return Q

def cnot():
    # digits = copy.deepcopy(d)
    # cn = []
    #
    # index = super().CNOT_logic(digits, c, t)
    # N = int(np.log(len(index)) / np.log(2))
    #
    N = 3
    basis = Basis(N)
    index = [0,1,2,3,4,5,6,7]

    for i in range(0, 2 ** N):
        new_row_ascolumn = [basis[index[i]]]
        print(new_row_ascolumn)
        # new_row = self.transpose(new_row_ascolumn)
        # cn.append(new_row)

cnot()