from PrintingCircuit import PrintingCircuit
import numpy as np
from abc import ABC, abstractmethod

class Quantum_Computer:

    def __init__(self, Qubits):
        """! The Quantum_Computer class initializer.
            @param Qubits: the number of Qubits in the quantum register
            @return  An instance of the Quantum_Computer class initialized with the specified number of qubits
        """

        self.Register_Size = Qubits

        #computational basis
        self.Zero = np.array([1, 0])  # This is |0> vector state
        self.One = np.array([0, 1])  # This is |1> vector state

        # produce basis and quantum register
        self.Basis()
        self.Q_Register()

        #gates
        self.I = np.array([[1,0],[0,1]]) #Identity gate
        self.Hadamard = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])  # sends |0> to |+> and |1> to |->
        self.Phase = np.array([[1, 0], [1, 0 + 1j]], dtype=complex)  # sends |0>+|1> to |0>+i|1>

        #more gates (unused as of 16/02)
        self.X = np.array([[0, 1], [1, 0]]) #Flips the |0> to |1> and vice versa
        self.Y = np.array([[0, 0 + 1j], [0 - 1j, 0]], dtype=complex) #converts |0> to i|1> and |1> to -i|0>
        self.Z = np.array([[1, 0], [0, -1]]) #sends |1> to -|1> and |0> to |0>
        self.RNot = 1 / np.sqrt(2) * np.array([[1, -1], [1, 1]]) #sends |0> to 0.5^(-0.5)(|0>+|1>) and |1> to 0.5^(-0.5)(|1>-|0>)
        self.T = np.array([[1,0],[0,1 / np.sqrt(2) * (1+1j)]], dtype=complex) #square root of phase (rotates by pi/8)
        self.CNot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]) #reversable xor: |00> -> |00>, |01> -> |11>
        self.Swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]) #¯\_(ツ)_/¯

        self.single_inputs = ["H", "RNot", "P", "X", "Y", "Z", "T"] #maps the string input to the relevant matrix and creates an array
        self.matrices = [self.Hadamard, self.RNot, self.Phase, self.X, self.Y, self.Z, self.T]

        self.double_inputs = ["CV", "CNOT", "CZ"]

        # produce binary digits for 2 input gates
        self.binary = self.produce_digits()

        #Keeps track of the gates that we've added via Gate Logic
        self.__gate_histroy = []

    #following is linear tensor product which looses matrix info
    '''    def Tensor_Prod(self, Q1, Q2):
        print(f"h{len(Q1)}")
        #IMPORTANT: Tensor product multiples the values of Q1 with the matrix Q2
        tensorprod = []
        for x in np.nditer(Q1): #iterate x over Q1
            tensorprod = np.append(tensorprod, x * Q2) #appends tensorprod with x'th value of Q1 * (matrix) Q2
        tensorprod = np.asmatrix(tensorprod)
        #ouput is linear tensor product (NOTE: matrix form infromation lost)

        return tensorprod '''

    def print_circuit(self):
        '''
        WARNING: Need to call `print_circuit_ascii` from terminal/cmd and will clear the terminal screen.
            Prints the quantum circuit in an ascii format on the terminal.
        '''
        pc = PrintingCircuit(self.__gate_histroy, self.Register_Size)
        pc.print_circuit_ascii()

    def Tensor_Prod(self, Q1, Q2):
        """! What the class/method does
            @param list the parameters and what they do
            @return  what the function returns
        """
        R = []
        if len(Q1.shape) > 1:
            for i in range(len(Q1)):
                R.append(Q1[i][0] * Q2)
                for j in range(1, len(Q1[i])):
                    R[i] = np.concatenate((R[i], (Q1[i][j] * Q2)), axis=1)

            C = R[0]
            for i in range(1, len(Q1)):
                C = np.concatenate((C, R[i]), axis=0)

        else:
            for i in range(len(Q1)):
                R.append(Q1[i] * Q2)
            C = R[0]
            if Q1.shape[0] > 0:
                ax = 0
            else:
                ax = 1
            for i in range(1, len(Q1)):
                C = np.concatenate((C, R[i]), axis=ax)

        return C
    # needs to be more general? rather than case by case

    def Mat_Mul(self, Q1, Q2):
        """! What the class/method does
            @param list the parameters and what they do
            @return  what the function returns
        """
        assert np.shape(Q1)[1] == np.shape(Q2)[0], "can't perform matrix multiplication"
        M = np.zeros(len(Q1)*len(Q2[0]))
        M.shape = (len(Q1), len(Q2[0]))

        for i in range(len(Q1)): #rows of Q1
            for j in range(len(Q2[0])): #columns of Q2
                for k in range(len(Q2)): #rows of Q2
                    M[i][j] += Q1[i][k] * Q2[k][j]

        return M

    def Q_Register(self):
        """! What the class/method does
            @param list the parameters and what they do
            @return  what the function returns
        """
        # returns an array of 2**n complex coefficients and ensures normalisation.
        j = 2 ** self.Register_Size
        coeffs = (0 + 0 * 1j) * np.zeros(j)
        for i in range(j):
            theta = np.random.random() * np.pi * 2
            coeffs[i] = (np.cos(theta) + np.sin(theta) * 1j) / j

        self.psi = coeffs
        self.psi.shape = (j, 1)


    def Basis(self):
        """! What the class/method does
            @param list the parameters and what they do
            @return  what the function returns
        """
        N = self.Register_Size
        self.Q = []

        for i in range(0, 2 ** N):
            self.Q.append(np.zeros(2 ** N))
            self.Q[i][i] = 1
            self.Q[i].shape = (2 ** N, 1)

    # def Psi(self):  #Our register doesn't need to call the basis states (yet), all we need is a column with n entries all equal to 1 (the sum of all the basis states), our normalised coefficients
    #     return np.matmul(self.Q, np.transpose(self.coeffs))

    def __Norm(self, array): #normalisation function (if needed?)
        """! What the class/method does
            @param list the parameters and what they do
            @return  what the function returns
        """
        total = 0
        for i in range(0, len(array)):
            total += np.sqrt(np.real(array[i]) ** 2 + np.imag(array[i]) ** 2)

        array /= total

        return array

    def __Single_Logic(self, gate, positions):
        """! What the class/method does
            @param list the parameters and what they do
            @return  what the function returns
        """
        '''
        this is another version of Single_Gates which includes assert statements and checks
        intend to save all those in a separate Input_Check method and ensure it's compatible witj Sinle_Gates below
        '''
        pass
        assert len(gate) == len(positions), "unequal list lenghts" #the number of gates should match the position lists

        # this is only one step. so only one logic gate can be applied to a single Qubit. so must return an error
        # if any value within the position sub-lists is repeated
        list_check = [] #create a unique list with the each of the arguments of the sublists of positions
        for k in range(len(positions)):
            for i in range(len(positions[k])):
                list_check.append((positions[k])[i])

        assert len(list(list_check)) == len(set(list_check)), "repeated position value"  #ensure lenght of list is equal to lenght of unique values in list i.e. avoid repetiotion

        single_gate_inputs = ["H", "RNot", "Phase", "X", "Y", "Z", "T"] #maps the string input to the relevant matrix and creates an array
        matrices = [self.Hadamard, self.RNot, self.Phase, self.X, self.Y, self.Z, self.T]
        M = []
        for j in range(len(gate)):
            for i in range(len(single_gate_inputs)):
                if str(gate[j]) == str(single_gate_inputs[i]):
                    M.append(matrices[i])
        assert len(M) > 0, ("Please enter one of the following gates and ensure correct spelling: H, RNot, Phase, X, Y, Z, T")

        L = self.I
        for i in range(len(positions)):
            for j in range(len(positions[i])):
                if positions[i][j] == 0:
                    L = M[i] #and else L = L

        for l in range(1, self.Register_Size):
            size1 = 1
            for dim in np.shape(L): size1 *= dim

            for i in range(len(positions)):
                for j in range(len(positions[i])):
                    if l == positions[i][j]:
                        L = self.Tensor_Prod(L, M[i])

                size2 = 1
                for dim in np.shape(L): size2 *= dim

                if size1 == size2:
                    L = self.Tensor_Prod(L, self.I)

        # L.shape = (2**self.Register_Size, 2**self.Register_Size)

        return L

    def __Single_Gates(self, gate, qnum):
        """! What the class/method does
            @param list the parameters and what they do
            @return  what the function returns
        """
        M = [0] * self.Register_Size

        for i in range(0, len(self.single_inputs)):
            for j in range(0, len(gate)):
                if self.single_inputs[i] == gate[j]:
                    for k in range(0, len(qnum[j])):
                        M[qnum[j][k]] = self.matrices[i]

        for i in range(0, len(M)):
            if type(M[i]) != np.ndarray:
                M[i] = self.I

        m = M[0]
        for i in range(1, len(M)):
            m = self.Tensor_Prod(m, M[i])

        return m

    def __recog_digits(self, digits):
        """! What the class/method does
            @param list the parameters and what they do
            @return  what the function returns
        """
        N = self.Register_Size
        numbers = []

        for i in range(0, 2 ** N):
            num = 0
            for j in range(0, N):
                num += 2 ** (N - j - 1) * digits[i][j]
            numbers.append(num)

        return numbers

    def produce_digits(self):
        """! What the class/method does
            @param list the parameters and what they do
            @return  what the function returns
        """
        N = self.Register_Size
        digits = []
        for i in range(0, 2 ** N):
            digit = []
            if i < (2 ** N) / 2:
                digit.append(0)
            else:
                digit.append(1)
            for j in range(1, N):
                x = i
                for k in range(0, len(digit)):
                    x -= digit[k] * (2 ** N) / (2 ** (k + 1))
                if x < (2 ** N) / (2 ** (j + 1)):
                    digit.append(0)
                else:
                    digit.append(1)
            digits.append(digit)
        return digits

    def __CNOT(self, c, t):  # c is the position of the control qubit, t is the position of the target qubit
        """! What the class/method does
            @param list the parameters and what they do
            @return  what the function returns
        """
        N = self.Register_Size

        cn = []
        digits = self.binary

        for i in range(0, 2 ** N):
            if digits[i][c] == 1:
                digits[i][t] = 1 - digits[i][t] % 2

        index = self.__recog_digits(digits)

        for i in range(0, 2 ** N):
            new_row = self.Q[index[i]]
            new_row.shape = (1, 2 ** N)
            cn.append(new_row)

        cn = np.asmatrix(np.asarray(cn))

        return cn

    def __CV(self, c, t):  # c is the position of the control qubit, t is the position of the target qubit
        """! What the class/method does
            @param list the parameters and what they do
            @return  what the function returns
        """
        N = self.Register_Size

        cv = []
        digits = self.binary

        for i in range(0, 2 ** N):
            if digits[i][c] == 1 and digits[i][t] == 1:
                new_row = 1j * self.Q[i]
            else:
                new_row = self.Q[i]
            new_row.shape = (1, 2 ** N)
            cv.append(new_row)

        cv = np.asmatrix(np.asarray(cv))

        return cv

    def __CZ(self, c, t):
        N = self.Register_Size

        cz = []

        for i in range(0, 2 ** N):
            if self.binary[i][c] == 1 and self.binary[i][t] == 1:
                new_row = -1 * self.Q[i]
            else:
                new_row = self.Q[i]
            new_row.shape = (1, 2 ** N)
            cz.append(new_row)

        cz = np.asarray(np.matrix(np.asarray(cz)))

        return cz

    def __Double_Gates(self, gate, qnum):
        """! What the class/method does
            @param list the parameters and what they do
            @return  what the function returns
        """

        if gate[0] == "CV":
            return self.__CV(qnum[0][0], qnum[0][1])
        if gate[0] == "CNOT":
            return self.__CNOT(qnum[0][0], qnum[0][1])
        if gate[0] == "CZ":
            return self.__CZ(qnum[0][0], qnum[0][1])


    def Make_Gate_Logic(self, inputs: list, name: str) -> np.ndarray:
        '''
        Defines a custom gate with a letter, in order to simploify the printing of the circuit.
        :param inputs: Input list of timesteps to build circuit.
        :param name: One letter to define the name of the gate.
        :return: A circuit.
        '''
        return self.Gate_Logic(inputs, add_gate_name = name)


    def Gate_Logic(self, inputs, add_gate_name = ""):
        """! What the class/method does
            @param list the parameters and what they do
            @return  what the function returns
            :param add_gate_history:
        """
        N = self.Register_Size
        step_n = len(inputs)

        # Add the gates to the gate history for printing later.
        if add_gate_name == "": # If not defining a custom name
            [self.__gate_histroy.append(i) for i in inputs]
        else: # If defining a gate with a custom Name
            self.__gate_histroy.append(([add_gate_name], [[0, N-1]]))

        M = []

        for i in range(0, step_n):
            for j in range(0, len(self.single_inputs)):
                if self.single_inputs[j] in inputs[i][0]:
                    M.append(self.__Single_Gates(inputs[i][0], inputs[i][1]))
            for j in range(0, len(self.double_inputs)):
                if self.double_inputs[j] in inputs[i][0]:
                    M.append(self.__Double_Gates(inputs[i][0], inputs[i][1]))

        m = M[0]
        for i in range(1, len(M)):
            m = np.matmul(m, M[i])

        return m

    def __ApplyOperator(self, matrix, qubits): #this function will produce the output but still working on how to do that
        """! What the class/method does
            @param list the parameters and what they do
            @return  what the function returns
        """
        qubits = self.Norm(np.matmul(matrix, qubits))
        return qubits

# main



# comp = Quantum_Computer(2)
# print(comp.Q)
#print(comp.__doc__)
#
# t1 = time.time()
# cnot = comp.CNOT(0, 1)
# t2 = time.time()
# print("CNOT gate", t2 - t1)
# print(cnot)
#
# # CNOT gate made using only CV and Hadamard gates:
# t3 = time.time()
# step1 = (["H"], [[1]])
# step2 = (["CV"], [[0, 1]])
# step3 = (["CV"], [[0, 1]])
# step4 = (["H"], [[1]])
#
# steps = [step1, step2, step3, step4]
#
# mat = comp.Gate_Logic(steps)
# t4 = time.time()
# print("CNOT gate using H and CV", t4 - t3)
# print(np.real(mat))
#
# # Toffoli gate made from only CV and Hadamard gates
# comp3 = QuantumComputer(3)
# t5 = time.time()
#
# toffoli_steps = [(["H"], [[2]]), (["CV"], [[1, 2]]),
#                  (["H"], [[1]]), (["CV"], [[0, 1]]), (["CV"], [[0, 1]]), (["H"], [[1]]),
#                  (["CV"], [[1, 2]]), (["CV"], [[1, 2]]), (["CV"], [[1, 2]]),
#                  (["H"], [[1]]), (["CV"], [[0, 1]]), (["CV"], [[0, 1]]), (["H"], [[1]]),
#                  (["CV"], [[0, 2]]), (["H"], [[2]])]
#
# toffoli_matrix = comp3.Gate_Logic(toffoli_steps)
# t6 = time.time()
# print("Toffoli gate using H and CV", t6 - t5)
# print(np.real(toffoli_matrix))
#
