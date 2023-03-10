from sparse import Sparse_Quantum_Computer
#from QuantumComputer import Quantum_Computer
from QuantumComputerV2 import QuantumComputer, DenseMatrix
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=np.inf, precision=2, suppress=True)

def glue_circuits(matricies: list[DenseMatrix]) -> np.ndarray:
    ''' Glues together circuits from left to right. In terms of matricies, `multiply_matricies([a, b, c])`, returns `c*b*a`.'''
    m = np.identity(len(matricies[0].matrix[0]))

    for matrix in np.flip(matricies, axis=0):
        #print(matrix.matrix)
        m = np.matmul(m, matrix.matrix )
    return m

def glue_lists(*lists) -> list:
    '''Adds elments from list_2 to list_1.'''
    big_list = []
    [[big_list.append(i) for i in list] for list in lists]
    return big_list

def CCnot(control_1, control_2, target) -> np.array:
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

def test_CCCnot():
    qc = QuantumComputer(3, 'Dense')

    # Defines the gates for grover's algorithm
    init_states = [
        #(["X"], [[2]]),
        (["CNOT"], [[1, 2]])
        #(["CZ"], [[0, 1]])
    ]

    circuits = [
        qc.gate_logic(init_states),
        #qc.gate_logic(CCnot(0, 1, 2))
    ]

    qc.print_circuit()

    # Prints the matrix representation of the circuits, and the output vector when the |00> is sent in. Should be able
    # to amplify the |11> states.
    glued_circuits = glue_circuits(circuits)
    print("With the matrix representation:")
    print(glued_circuits)
    print("\nOutput probabilities:")
    probs = qc.measure_all(glued_circuits)
    print(probs)

    plt.bar(probs.keys(), probs.values(), 1)
    plt.show()

def GroverAlgorithm_Mini_Suduko():
    num_qubits = 6
    qc = QuantumComputer(num_qubits)

    # Defines the gates for grover's algorithm
    init_states = [
        qc.gate_logic([
            (["H"], [[0]]),
            (["H"], [[1]]),
            (["H"], [[2]])
        ]),
        qc.gate_logic([ (["H"], [[4]]) ]),
        qc.gate_logic([ (["X"], [[4]]) ])
    ]

    classical_chossing = [
        qc.Make_gate_logic(CCCnot(0, 1, 2, 3, 5), "T4"),
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


    amplify_amplitude = [
        qc.gate_logic([
            (["H"], [[0]]),
            (["H"], [[1]]),
            (["H"], [[2]])
        ]),
        qc.gate_logic([
            (["X"], [[0]]),
            (["X"], [[1]]),
            (["X"], [[2]])
        ]),
        qc.gate_logic(CCz(0, 1, 2), "Z"),
        qc.gate_logic([
            (["X"], [[0]]),
            (["X"], [[1]]),
            (["X"], [[2]])
        ]),
        qc.gate_logic([
            (["H"], [[0]]),
            (["H"], [[1]]),
            (["H"], [[2]])
        ]),
    ]

    oracle = glue_lists(classical_chossing, kick_back, reset)

    # Constructs circuit from pieces defined above (will be glued together later to give the complete matrix)
    circuits = glue_lists(init_states, oracle, amplify_amplitude)
    # Prints circuit and matrix.
    qc.print_circuit()

    glued_circuits = glue_circuits(circuits)
    print("With the matrix representation:")
    print(glued_circuits)
    print("\nOutput probabilities:")
    startVec = np.zeros((2**num_qubits))
    startVec[0]=1
    outVec = glued_circuits.dot(startVec)
    print(np.array([100*outVec[i]*np.conjugate(outVec[i]) for i in range(len(outVec))], dtype=np.float32))

def GroverAlgorithm_3Qubit():
    '''A function implementing a two qubit version of Grover's algorithm.'''
    qc = QuantumComputer(3, 'Dense')

    # Defines the gates for grover's algorithm
    init_states = [(["H"], [[0]]),
                   (["H"], [[1]]),
                   (["H"], [[2]]),
                   #(["X"], [[3]]), (["H"], [[3]])
                   ]

    oracle = [ (["CZ"], [[0, 2]]) ]

    # Constructs circuit from pieces defined above (will be glued together later to give the complete matrix)
    circuits = [
        qc.gate_logic(init_states),
        qc.gate_logic(oracle),
        qc.gate_logic([
            (["H"], [[0]]),
            (["H"], [[1]]),
            (["H"], [[2]]),
            (["X"], [[0]]),
            (["X"], [[1]]),
            (["X"], [[2]]),
            (["H"], [[2]]),
        ]),
        qc.gate_logic(CCnot(0, 1, 2), "T"),
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


    # Prints circuit and matrix.
    qc.print_circuit()

    # Prints the matrix representation of the circuits, and the output vector when the |00> is sent in. Should be able
    # to amplify the |11> states.
    glued_circuits = glue_circuits(circuits)

    print("\nOutput probabilities:")
    probs = qc.measure_all(glued_circuits)
    print(probs)

    plt.bar(probs.keys(), probs.values(), 1)
    plt.show()

def main():
    GroverAlgorithm_3Qubit()
    #test_CCCnot()


if __name__=="__main__":
    main()

