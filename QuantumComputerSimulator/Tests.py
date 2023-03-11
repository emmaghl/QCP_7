from QuantumComputerSimulator.QuantumComputer import QuantumComputer
from QuantumComputerSimulator.mods.DenseMatrix import DenseMatrix

import numpy as np

class Test():
    '''
    This class contains functions that can be called by adding the --test arfument from the terminal:
`   python3 sample.py --test`
    which will verify that the circuit should be operating as expected. Functions were added throughout the duration of the project.
    '''
    def __init__(self):
        # Gets all methods to test
        list_methods = [method for method in dir(Test) if method.startswith('_') is False]

        # Loops through printing all methods and checking running them
        for method in list_methods:
            print(f"Checking: {method}...")
            func = getattr(Test, method) #gets function ready to call
            func(self)

    def __compare_vecs(self, vec1: np.array, vec2: np.array, precision: int = 2):
        if np.all(np.around(vec1, decimals=precision) == np.around(vec2, decimals=precision)):
            return True
        return False

    def __glue_circuits(self, matricies: np.array) -> np.ndarray:
        ''' Glues together circuits from left to right. In terms of matricies, `multiply_matricies([a, b, c])`, returns `c*b*a`.'''
        m = np.identity(len(matricies[0].matrix[0]))

        for matrix in np.flip(matricies, axis=0):
            m = np.matmul(m, matrix.matrix)
        return m

    def matrix_multiply_Dense(self):
        dm = DenseMatrix
        matrix1 = np.array([
            [-1, 5, 3,],
            [4.5, 0+5j, 1],
            [4, 6, 0+7j]
        ], dtype = complex)
        matrix2 = np.array([
            [1, 6, 4,],
            [2, 0+5j, 1],
            [4, 3, 4]
        ], dtype = complex)
        final = np.around(dm.matrix_multiply(matrix1, matrix2).matrix, decimals=1)
        final_compare = np.array([
            [21.0, 3.0+25.0j, 13.0,],
            [8.5+10.0j, 5.0, 22.0+5.0j],
            [16.0+28.0j, 24.0+51.0j, 22.0+28.0j]
        ], dtype = complex)
        assert np.all(final == final_compare), f"Multiplication of matricies from Dense class failed. The result should return -> \n {final_compare}\nInstead it returned -> \n {final}"

    def correct_basis_Dense(self):
        qc = QuantumComputer(3, 'Dense')

        init_states = [
            (["X"], [[1]]),
            (["X"], [[2]])
        ]

        circuits = [
            qc.gate_logic(init_states)
        ]

        # We should expect only the states |110> to be 100%
        glued_circuits = self.__glue_circuits(circuits)
        probs = qc.get_probabilities(glued_circuits)
        assert (probs['110'] == 1), f"Incorrect basis orientation of basis! It should be that |110> = 1, instead it's {probs['110']}. Most likely that |011> is measured instead: |001> = {probs['011']}"

    def catch_incorrect_user_input(self):
        '''Input an incorrect format for the matrix'''
        qc = QuantumComputer(2, "Dense")
        try:
            qc.gate_logic( (["H"], [[0]]) ) # Should be an array not tuple
        except Exception:
            pass
        else:
            assert (False), "Didn't catch user input error`1"

    def CNOT_gate_and_Tensor_Product(self):
        qc = QuantumComputer(3, 'Dense')

        init_states = [
            (["X"], [[0]]),
            (["CNOT"], [[0, 1]])
        ]

        circuits = [
            qc.gate_logic(init_states),
        ]

        # Prints the matrix representation of the circuits, and the output vector when the |00> is sent in. Should be able
        # to amplify the |11> states.
        glued_circuits = self.__glue_circuits(circuits)
        probs = qc.get_probabilities(glued_circuits)

        assert (probs['011'] == 1), "CNOT gate isn't working properly! Check CNOT_gate_and_Tensor_Product function in Test.py class for more details about the setup of the circuit."