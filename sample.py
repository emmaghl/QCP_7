'''Sample file for testing the QuantumComputerSimulator module, and showcasing the features. To run tests, add `--test` argument when running from the terminal.'''
from QuantumComputerSimulator import QuantumComputer, Test

import matplotlib.pyplot as plt
import sys
import numpy as np
import time

def glue_circuits(matricies: object) -> np.ndarray:
    ''' Glues together circuits from left to right. In terms of matricies, `multiply_matricies([a, b, c])`, returns `c*b*a`.'''
    m = matricies[0]

    for matrix in matricies[::-1]:
        m = matrix.matrix_multiply(m,matrix)
    return m

def user_validation(msg: str, options: list[str]) -> str:
    print(msg)
    user_input = input('>')
    while not user_input.lower() in options:
        print(f'Please select from: {options}.')
        user_input = input(msg)
    return user_input

def glue_lists(*lists) -> list:
    '''Adds elments from list_2 to list_1.'''
    big_list = []
    [[big_list.append(i) for i in list] for list in lists]
    return big_list

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

def CCCnot(control_1, control_2, control_3, target, auxilary) -> np.array:
    return glue_lists(
        CCnot(control_1, control_3, auxilary),
        CCnot(control_2,auxilary, target),
        CCnot(control_1, control_3, auxilary)
    )

def GroverAlgorithm_3Qubit(show_plots=False):
    '''A function implementing a two qubit version of Grover's algorithm.'''
    qc = QuantumComputer(3, 'Lazy')

    # Defines the gates for grover's algorithm
    init_states = [
        (["H", "H", "H"], [[0], [1], [2]])
    ]

    oracle = [
        (["CZ"], [[0, 2]])
    ]

    half_of_amplification = [
        (["H", "H", "H"], [[0], [1], [2]]),
        (["X", "X", "X"], [[0], [1], [2]]),
        (["H"], [[2]])
    ]

    # Constructs circuit from pieces defined above (will be glued together later to give the complete matrix)
    circuits = [
        qc.add_gate_to_circuit(init_states),
        qc.add_gate_to_circuit(oracle),
        qc.add_gate_to_circuit(half_of_amplification),
        qc.add_gate_to_circuit(CCnot(0, 1, 2)),
        qc.add_gate_to_circuit(half_of_amplification[::-1]) # Reverses list
    ]

    # Prints circuit and matrix, and then builds it. Can call print circuit earlier to check it is what is wanted to
    qc.print_circuit()
    user = user_validation('Continue building with circuit? (y/n)', ['y', 'n'])
    if user == 'n':
        exit()

    circuit = qc.build_circuit()

    # Prints the matrix representation of the circuits, and the output vector when the |00> is sent in. Should be able
    # to amplify the |11> states.
    print("With the matrix representation:")
    print(circuit.matrix)

    print("\nOutput probabilities:")
    probs = qc.get_probabilities(circuit.matrix)
    [print(f"|{i}> : {probs[i]}") for i in probs.keys()]

    if show_plots:
        plt.bar(probs.keys(), probs.values(), 1)
        plt.show()

def GroverAlgorithm_Mini_Suduko(show_plots = False):
    num_qubits = 6
    qc = QuantumComputer(num_qubits)

    print("Initialised gates...")
    # Defines the gates for grover's algorithmv
    init_states = [
        qc.gate_logic([
            (["H"], [[0]]),
            (["H"], [[1]]),
            (["H"], [[2]])
        ]),
        qc.gate_logic([ (["X"], [[4]]) ]),
        qc.gate_logic([ (["H"], [[4]]) ])
    ]

    print("Adding oracle...")

    classical_chossing = [
        qc.gate_logic(CCCnot(0, 1, 2, 3, 5), "T4"),
        qc.gate_logic([
            (["CNOT"], [[0, 3]]),
            (["CNOT"], [[1, 3]]),
            (["CNOT"], [[2, 3]]),
        ])
    ]

    kick_back = [
        qc.gate_logic([
            (["CNOT"], [[3, 4]]),
        ])
    ]

    reset = [
        qc.gate_logic(CCCnot(0, 1, 2, 3, 5), "T4"),
        qc.gate_logic([
            (["CNOT"], [[0, 3]]),
            (["CNOT"], [[1, 3]]),
            (["CNOT"], [[2, 3]]),
        ])
    ]

    print("Adding amplitude amplification...")

    amplify_amplitude = [
        qc.gate_logic([
            (["H"], [[0]]),
            (["H"], [[1]]),
            (["H"], [[2]]),
            (["X"], [[0]]),
            (["X"], [[1]]),
            (["X"], [[2]]),
            (["H"], [[2]])
        ]),
        qc.gate_logic(CCnot(0, 1, 2), "Z"),
        qc.gate_logic([
            (["H"], [[2]]),
            (["X"], [[0]]),
            (["X"], [[1]]),
            (["X"], [[2]]),
            (["H"], [[0]]),
            (["H"], [[1]]),
            (["H"], [[2]])
        ])
    ]

    print("Gluing circuit")
    oracle = glue_lists(classical_chossing, kick_back, reset)
    # Constructs circuit from pieces defined above (will be glued together later to give the complete matrix)
    circuits = glue_lists(init_states, oracle, amplify_amplitude)
    # Prints circuit and matrix.
    qc.print_circuit()

    glued_circuits = glue_circuits(circuits)

    print("\nOutput probabilities:")
    probs = qc.get_probabilities(glued_circuits)
    [print(f"|{i}> : {probs[i]}") for i in probs.keys()]

    if show_plots:
        plt.bar(probs.keys(), probs.values(), 1)
        plt.show()



if __name__=="__main__":
    # Runs example algorithms if not testing contents if not testing
    if len(sys.argv) == 1:
        # Prints the options to the user
        options = [
            '[1] 3 qubit Grovers (Dense)',
            '[2] Single binary row of 3x3 sudoko (Dense)',
            '[3] A full row of 3x3 sudoko (Dense)',
        ]
        [print(f'{o}') for o in options]
        user_input = user_validation('Enter the number beside the algorithm that you would like to run.', ['1', '2', '3'])

        if user_input == 'exit':
            exit()

        np.set_printoptions(linewidth=np.inf, precision=2, suppress=True)# Makes output from numpy arrays more pleasant.
        print('Running, this may take a bit...')

        if user_input == '1':
            GroverAlgorithm_3Qubit(show_plots=False)
        elif user_input == '2':
            GroverAlgorithm_Mini_Suduko(show_plots=False)

    elif '--test' in sys.argv:
        print("Running tests...")
        Test()
        print("Tests completed successfully.")
        exit()
    else:
        print("The only argument available is '--test'. \nThis runs all functions in the class 'Test' (that don't have _ in front of their name). \nThe 'Tests' class is located in QuantumComputerSimulator/Tests.py.")