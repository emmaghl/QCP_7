from sparse import Sparse_Quantum_Computer
#from QuantumComputer import Quantum_Computer
from QuantumComputerV2 import QuantumComputer, DenseMatrix
import numpy as np
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

def GroverAlgorithm_3Qubit():
    '''A function implementing a two qubit version of Grover's algorithm.'''
    qc = QuantumComputer(3, 'Dense')

    # Defines the gates for grover's algorithm
    init_states = [(["H"], [[0]]),
                   (["H"], [[1]]),
                   (["H"], [[2]])]

    oracle = [(["CZ"], [[0, 2]])]

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
    print("With the matrix representation:")
    print(glued_circuits)
    print("\nOutput probabilities:")
    outVec = np.matmul(glued_circuits,[1, 0, 0, 0, 0, 0, 0, 0])
    print(np.array([100*outVec[i]*np.conjugate(outVec[i]) for i in range(len(outVec))], dtype=np.float32))

def main():
    GroverAlgorithm_3Qubit()


if __name__=="__main__":
    main()

