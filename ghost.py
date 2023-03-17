def Q_Register(self):
    '''
    Q_Register() build's the quantum register for the quantum computer for a given number of qubits.
    :return:
    '''
    coeffs = []

    for i in range(0, self.N):
        alpha = np.random.random() + np.random.random() * 1j
        beta = np.random.random() + np.random.random() * 1j
        normF = np.sqrt(alpha * np.conj(alpha) + beta * np.conj(beta))

        alpha /= normF
        beta /= normF

        coeffs.append(np.array([[alpha], [beta]]))

    self.psi = coeffs[0]
    for i in range(1, self.N):
        self.psi = DenseMatrix.tensor_prod(self.psi, coeffs[i])