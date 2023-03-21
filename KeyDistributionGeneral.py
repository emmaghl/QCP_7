from QuantumComputerSimulator import QuantumComputer, Test
from QuantumComputerSimulator.mods.SparseMatrix import SparseMatrix
from QuantumComputerSimulator.mods.DenseMatrix import DenseMatrix
from QuantumComputerSimulator.mods.LazyMatrix import LazyMatrix

import numpy as np
import random

def encode_message(n, A_bases, A_bits, register):
    for i in range(n):
        j = A_bases[i]
        k = A_bits[i]
        if j == 0:
            if k == 0:
                pass
            else:
                circuit = qc.gate_logic([(["X"], [[i]])])
                circuit = circuit.matrix
                register = qc.Matrix.matrix_multiply(circuit, register)
                register = register.matrix
        if j == 1:
            if k == 0:
                circuit = qc.gate_logic([(["H"], [[i]])])
                circuit = circuit.matrix
                register = qc.Matrix.matrix_multiply(circuit, register)
                register = register.matrix
            else:
                circuit_1 = qc.gate_logic([(["X"], [[i]])])
                circuit_1 = circuit_1.matrix
                circuit_2 = qc.gate_logic([(["H"], [[i]])])
                circuit_2 = circuit_2.matrix
                circuit = qc.Matrix.matrix_multiply(circuit_2, circuit_1)
                register = qc.Matrix.matrix_multiply(circuit, register)
                register = register.matrix
    return register
def C_intercepts(n, C_bases, register):
    print('The random bases you measure the message with are =', C_bases)
    register_intercept = []
    for i in range(n):
        g = C_bases[i]
        if g == 0:
            result = qc.measure_any(i, 0, register)
            if qc.Matrix == DenseMatrix:
                register_intercept.append(result)
            elif qc.Matrix == SparseMatrix:
                register_intercept.append([0, i, result])
            elif qc.Matrix == LazyMatrix:
                print("oh no, not quite working for Lazy yet ... ")
        else:
            circuit = qc.gate_logic([(["H"], [[i]])])
            circuit = circuit.matrix

            register = qc.Matrix.matrix_multiply(circuit, register)
            register = register.matrix
            result = qc.measure_any(i, 0, register)
            if qc.Matrix == DenseMatrix:
                register_intercept.append(result)
            elif qc.Matrix == SparseMatrix:
                register_intercept.append([0, i, result])
            elif qc.Matrix == LazyMatrix:
                print("oh no, not quite working for Lazy yet ... ")
    print('Intercepted message =', register_intercept)
    zero_col = qc.Matrix("zerocol")
    one_col = qc.Matrix("onecol")

    for i in range(n):
        q = register_intercept[i]
        if i == 0:
            if q == 0:
                register = qc.Matrix.transpose(zero_col)
            else:
                register = qc.Matrix.transpose(one_col)
        else:
            if q == 0:
                register = qc.Matrix.tensor_prod(zero_col, register)
                register = register.matrix
            else:
                register = qc.Matrix.tensor_prod(one_col, register)
                register = register.matrix

    return register
def B_measure(n, B_bases, register):
    measurement = []
    for i in range(n):
        g = B_bases[i]
        if g == 0:

            result = qc.measure_any(i, 0, register)
            measurement.append(result)

        else:
            circuit = qc.gate_logic([(["H"], [[i]])])
            circuit = circuit.matrix
            register = qc.Matrix.matrix_multiply(circuit, register)
            register = register.matrix
            result = qc.measure_any(i, 0, register)
            measurement.append(result)
    return measurement

def garbage_function(n, bases_1, bases_2, bits_1):
    Key = []
    for i in range(n):
        a = bases_1[i]
        b = bases_2[i]
        c = bits_1[i]
        if a == b:
            Key.append(c)
        else:
            pass

    return Key

def user_validation(msg: str, options: list[str]) -> str:
    print(msg)
    user_input = input('>')
    while not user_input.lower() in options:
        print(f'Please select from: {options}.')
        print(msg)
        user_input = input('>')
    return user_input
def Q_Key_Distribution():
    print("You are acting as a communication channel for person A to send secret messages to person B.")
    t = str(input('What type of matrix object do you want to use? Type D for dense, S for sparse, L for lazy: '))
    n = int(input('How long would person A like their bit message to be?: '))

    global qc
    if t == "D" or t == "d":
        qc = QuantumComputer(n, 'Dense')
    if t == "S" or t == "s":
        qc = QuantumComputer(n, 'Sparse')
    if t == "L" or t == "l":
        qc = QuantumComputer(n, 'Lazy')

    '''Step 0 set up number of qubits '''
    register = qc.Matrix.quantum_register(n)

    ''' Step 1 random A bit message '''
    A_bits = np.random.randint(2, size=n)

    ''' Step 2 random A bases'''
    A_bases = np.random.randint(2, size=n)

    ''' Step 3 encode qubits using A bits and A bases'''
    register = encode_message(n, A_bases, A_bits, register)

    print('Person A has their secretly encoded message ready to send to person B.')

    ''' Step ! interception '''
    while True:
        y = str(input('Do you want to intercept and try and read their message?\n >'))
        if y == "Yes" or y == "yes" or y == "YES":
            C_bases = np.random.randint(2, size=n)
            register = C_intercepts(n, C_bases, register)
            break
        elif y == "No" or y == "no" or y == "NO":
            break
        else:
            print("Whoops, that was an incorrect input! Accepted inputs: Yes, No")

    ''' Step 4 '''
    B_bases = np.random.randint(2, size=n)
    measurement = B_measure(n, B_bases, register)

    print('Person B has measured the message.')
    print('Person B shares the bases which they measured the message with, and vice versa so that they can both create a key from the matching bases.')
    print('A bases =', A_bases)
    print('B bases =', B_bases)

    ''' Step 5 '''
    A_Key = garbage_function(n, A_bases, B_bases, A_bits)
    B_Key = garbage_function(n, A_bases, B_bases, measurement)

    print('After garbage collection the length of both keys is:', len(A_Key))

    if len(A_Key) == 0:
        print('No Key was created because none of the bases matched, try again.')
        exit()
    elif len(A_Key) == 1:
        print('No useful sample can be created because the key length was less then 2, try again.')
        exit()
    else:
        pass

    ''' Step 6'''
    print('Now a random sample is generated to test if the keys for person A and B are secure.')
    sample_A = []
    sample_B = []
    j = len(A_Key)
    number_of_samples = round(j * 0.5)
    s = random.sample(range(0, j), number_of_samples)

    for i in s:
        f = A_Key[i]
        h = B_Key[i]
        sample_A.append(f)
        sample_B.append(h)

    print('Person A and person B share their random sample of the message they measured using the bases already shared:')
    print('A random sample = ', sample_A)
    print('B random sample = ', sample_B)

    ''' Step 7 '''
    j = len(sample_A)

    for i in range(j):
        p = 1
        if sample_A[i] == sample_B[i]:
            p = p + 1
        else:
            print('You were caught listening!')
            exit()
    print('Secret Key is probably secure.')

    ''' Step 8 '''
    j = len(A_Key)
    t = range(0, j)
    t = list(t)

    for i in s:
        t.remove(i)

    A_secret_key = []
    B_secret_key = []

    for i in t:
        A_secret_key.append(A_Key[i])
        B_secret_key.append(B_Key[i])

    print('Both person A and person B have their secret keys now:')
    print('A Secret Key =', A_secret_key, 'These are not shared publicaly, but are used to encript messages.')
    print('B Secret Key =', B_secret_key, 'These are not shared publicaly, but are used to encript messages.')
def KeyDist_report_example():
    print("You are acting as a communication channel for person A to send secret messages to person B.")
    t = str(input('What type of matrix object do you want to use? Type D for dense, S for sparse, L for lazy: '))
    print('Person A would like to send a 5 qubit message')
    n = 5
    global qc
    if t == "D" or t == "d":
        qc = QuantumComputer(n, 'Dense')
    if t == "S" or t == "s":
        qc = QuantumComputer(n, 'Sparse')
    if t == "L" or t == "l":
        qc = QuantumComputer(n, 'Lazy')

    '''Step 0 set up number of qubits '''
    register = qc.Matrix.quantum_register(n)

    ''' Step 1 random A bit message '''

    A_bits = (0, 1, 0, 1, 1)
    print('A Bit message =', A_bits)

    ''' Step 2 random A bases'''
    A_bases = (0, 1, 1, 0, 1)
    print('A Bases =', A_bases)

    ''' Step 3 encode qubits using A bits and A bases'''
    register = encode_message(n, A_bases, A_bits, register)

    print('Person A has their secretly encoded message ready to send to person B.')

    ''' Step ! interception '''
    while True:
        y = str(input('Do you want to intercept and try and read their message?\n >'))
        if y == "Yes" or y == "yes" or y == "YES":
            C_bases = ( 0, 1, 1, 0, 0)
            register = C_intercepts(n, C_bases, register)
            break
        elif y == "No" or y == "no" or y == "NO":
            break
        else:
            print("Whoops, that was an incorrect input! Accepted inputs: Yes, No")

    ''' Step 4 '''
    B_bases = (0, 1, 1, 0, 0)
    measurement = B_measure(n, B_bases, register)

    print('Person B has measured the message.')
    print(
        'Person B shares the bases which they measured the message with, and vice versa so that they can both create a key from the matching bases.')
    print('A bases =', A_bases)
    print('B bases =', B_bases)

    ''' Step 5 '''
    A_Key = garbage_function(n, A_bases, B_bases, A_bits)
    B_Key = garbage_function(n, A_bases, B_bases, measurement)

    print('After garbage collection the length of both keys is:', len(A_Key))

    if len(A_Key) == 0:
        print('No Key was created because none of the bases matched, try again.')
        exit()
    elif len(A_Key) == 1:
        print('No useful sample can be created because the key length was less then 2, try again.')
        exit()
    else:
        pass

    ''' Step 6'''
    print('Now a random sample is generated to test if the keys for person A and B are secure.')
    sample_A = []
    sample_B = []
    s = (1, 3)

    for i in s:
        f = A_Key[i]
        h = B_Key[i]
        sample_A.append(f)
        sample_B.append(h)

    print(
        'Person A and person B share their random sample of the message they measured using the bases already shared:')
    print('A random sample = ', sample_A)
    print('B random sample = ', sample_B)

    ''' Step 7 '''
    j = len(sample_A)

    for i in range(j):
        p = 1
        if sample_A[i] == sample_B[i]:
            p = p + 1
        else:
            print('You were caught listening!')
            exit()
    print('Secret Key is probably secure.')

    ''' Step 8 '''
    j = len(A_Key)
    t = range(0, j)
    t = list(t)

    for i in s:
        t.remove(i)

    A_secret_key = []
    B_secret_key = []

    for i in t:
        A_secret_key.append(A_Key[i])
        B_secret_key.append(B_Key[i])

    print('Both person A and person B have their secret keys now:')
    print('A Secret Key =', A_secret_key, 'These are not shared publicaly, but are used to encript messages.')
    print('B Secret Key =', B_secret_key, 'These are not shared publicaly, but are used to encript messages.')


if __name__=="__main__":
    # Runs example algorithms if not testing contents if not testing
    options = [
            '[1] Report Example',
            '[2] Normal',
    ]
    [print(f'{o}') for o in options]
    user_input = user_validation('Enter the number beside the option that you would like to run.',
                                 ['1', '2', 'exit'])
    if user_input == '1':
        KeyDist_report_example()
    elif user_input == '2':
        Q_Key_Distribution()
    else:
        exit()








