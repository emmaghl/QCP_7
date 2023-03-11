'''Sample file for testing the QuantumComputerSimulator module, and showcasing the features. To run tests, add `--test` argument when running from the terminal.'''
from QuantumComputerSimulator import QuantumComputer, Test

import matplotlib.pyplot as plt
import sys
import numpy as np

def glue_circuits(matricies: np.array) -> np.ndarray:
    ''' Glues together circuits from left to right. In terms of matricies, `multiply_matricies([a, b, c])`, returns `c*b*a`.'''
    m = np.identity(len(matricies[0].matrix[0]))

    for matrix in np.flip(matricies, axis=0):
        m = np.matmul(m, matrix.matrix)
    return m

def CCnot(control_1, control_2, target) -> list:
    '''Defines the Toffoli gate, and an example for implementing other gates from the elementary ones.'''
    gate_built = [
        (["H"], [[target]]),
        (["CV"], [[control_2, target]]),
        (["H"], [[control_2]]),
        (["CV"], [[control_1, control_2]]),
        (["CV"], [[control_1, control_2]]),
        (["H"], [[control_2]]),
        (["CV"], [[control_2, target]]),
        (["CV"], [[control_2, target]]),
        (["CV"], [[control_2, target]]),
        (["H"], [[control_2]]),
        (["CV"], [[control_1, control_2]]),
        (["CV"], [[control_1, control_2]]),
        (["H"], [[control_2]]),
        (["CV"], [[control_1, target]]),
        (["H"], [[target]])
    ]
    return gate_built

def GroverAlgorithm_3Qubit(show_plots=False):
    '''A function implementing a two qubit version of Grover's algorithm.'''
    qc = QuantumComputer(3, 'Dense')

    # Defines the gates for grover's algorithm
    init_states = [
        (["H"], [[0]]),
        (["H"], [[1]]),
        (["H"], [[2]])
    ]

    oracle = [
        (["CZ"], [[0, 2]])
    ]

    half_of_amplification = [
        (["H"], [[0]]),
        (["H"], [[1]]),
        (["H"], [[2]]),
        (["X"], [[0]]),
        (["X"], [[1]]),
        (["X"], [[2]]),
        (["H"], [[2]])
    ]

    # Constructs circuit from pieces defined above (will be glued together later to give the complete matrix)
    circuits = [
        qc.gate_logic(init_states),
        qc.gate_logic(oracle),
        qc.gate_logic(half_of_amplification),
        qc.gate_logic(CCnot(0, 1, 2), "T"),
        qc.gate_logic(half_of_amplification[::-1]) # Reverses list
    ]


    # Prints circuit and matrix.
    qc.print_circuit()

    # Prints the matrix representation of the circuits, and the output vector when the |00> is sent in. Should be able
    # to amplify the |11> states.
    glued_circuits = glue_circuits(circuits)
    print("With the matrix representation:")
    print(glued_circuits)

    print("\nOutput probabilities:")
    probs = qc.get_probabilities(glued_circuits)
    [print(f"|{i}> : {probs[i]}") for i in probs.keys()]

    if show_plots:
        plt.bar(probs.keys(), probs.values(), 1)
        plt.show()

if __name__=="__main__":
    # Runs example algorithms if not testing contents if not testing
    if not '--test' in sys.argv:
        print('Enter the number beside the algorithm that you would like to run.')
        print('[1] 3 Qubit Grovers')
        user_input = input(">")
        selection = ['1', 'exit']
        while not user_input.lower() in ['1', 'exit']:
            print(f'Please select from: {selection}.')
            user_input = input(">")

        np.set_printoptions(linewidth=np.inf, precision=2, suppress=True)# Makes output from numpy arrays more pleasant.

        if user_input == '1':
            GroverAlgorithm_3Qubit(show_plots=False)
    else:
        print("Running tests...")
        Test()
        print("Tests completed successfully.")
        exit()