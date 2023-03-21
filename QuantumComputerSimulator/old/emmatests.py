from QuantumComputerSimulator import QuantumComputer

import matplotlib.pyplot as plt
import sys
import numpy as np

qc = QuantumComputer(3, 'Dense')

#
# # Defines the gates for grover's algorithm
# init_states = [
#     (["CNOT"], [[0, 1]])
# ]
#
# circuits = [
#     qc.gate_logic(init_states),
# ]
#
# qc.print_circuit()

# def glue_circuits(matricies: np.array) -> np.ndarray:
#     ''' Glues together circuits from left to right. In terms of matricies, `multiply_matricies([a, b, c])`, returns `c*b*a`.'''
#     m = np.identity(len(matricies[0].matrix[0]))
#
#     for matrix in np.flip(matricies, axis=0):
#         m = np.matmul(m, matrix.matrix)
#     return m
#
# def emma_test_two():
#     qc = QuantumComputer(3, 'Sparse')
#
#     # Defines the gates for grover's algorithm
#     init_states = [
#         (["CNOT"], [[0, 1]])
#     ]
#
#     circuits = [
#         qc.gate_logic(init_states),
#     ]
#
#     qc.print_circuit()
#
#     # Prints the matrix representation of the circuits, and the output vector when the |00> is sent in. Should be able
#     # to amplify the |11> states.
#     glued_circuits = glue_circuits(circuits)
#     print("test 2")
#     print("With the matrix representation:")
#     print(glued_circuits)
#     print("\nOutput probabilities:")
#     probs = qc.measure_all(glued_circuits)
#     print(probs)
#
#     # plt.bar(probs.keys(), probs.values(), 1)
#     # plt.show()
#
# emma_test_two()

