# Quantum Computer Simulator
This python package can build, print and simulate a quantum computer. The simulated quantum computer can be chosen to implement dense (lazy and sparse?) methods.

## Python Version and OS
Tested with Python 3.11.2 on macOS Ventura 13.1. See `Requirements.txt` file to see the python libraries and versions needed to run this code. The package should also run for Linux and Windows operating systems; the only reason why it wouldn't is that `PrintCircuit.py` clears the screen using the terminal/cmd when the circuit is printed in ASCII. This command varies across operating systems, however the package should account for this. 

## Installation and Testing
### Using Git
Open terminal and change into a directory that you wish the repository to be downloaded in. This can be accomplished with the command
```angular2html
cd directory_name
```
To list the available directories on your machine (and files within the current directory you're in), use  the following command if you're running MacOs/Linux:
```angular2html
ls
```
or 
```angular2html
dir
```
for Windows. This can help you navigate between directories using the terminal. Then, create a new folder (such as `QuantumComputer`) with:
```angular2html
mkdir QuantumComputer
```
Enter this folder with
```angular2html
cd QuantumComputer
```
The git repository will be downloaded into this folder. To download the repository, first install git (https://github.com/git-guides/install-git), and then enter:
```angular2html
git clone https://github.com/emmaghl/Project-Bambi
```
Enter the cloned repository with the `cd` command, and locate the `sample.py` file. Then run:
```angular2html
python3 sample.py --test
```
This assumes that you have `python3` added to your path (https://realpython.com/add-python-to-path/). If successful, this should complete without any errors, and validate that the `QuantumComputerSimulator` package is working as it should be.

## Documentation
The documentation can be found in the [docs folder](QuantumComputerSimulator/docs/QuantumComputerSimulator). To view the documentation, open `index.html` in your favourite web browser. 

## Quick Start Guide
The `sample.py` file contains three implementations of Grover's algorithm (if opening with an IDE, it's best to collapse all the functions and open them one by one). The three functions in order of increasing qubits are: 
1. `GroverAlgorithm_3Qubit`
2. `GroverAlgorithm_SingleRow_BinaryCol_Suduko`
3. `GroverAlgorithm_SingleRow_Suduko`
 
Let's implement the first function, `GroverAlgorithm_3Qubit`, to learn how to interface with this package.

In a new python file, the `QuantumComputerSimulator` package is imported like so:
```python
from QuantumComputerSimulator import QuantumComputer
```
This import statement assumes that the new python file was created in the same directory as of `sample.py`. The import statement can be changed accordingly to where the python file was created relative to the `QuantumComputerSimulator` folder. Next, a 3 qubit quantum computer is instantiated:
```python
qc = QuantumComputer(3)
```
By default, the quantum computer will use dense methods which are slower than other techniques, however the way it works is more 'intuitive'. After instantiating the computer, the circuit can start to be built. For Grover's, the circuit needs to put the register into a superposition of equal weight. This can be achieved by acting three Hadamard gates to each qubit. The following code achieves this:
```python
init_states = [
    (["H", "H", "H"], [[0], [1], [2]])
]
qc.add_gate_to_circuit(init_states)
```
Let's unpack this. The list `init_states` holds a single *time step*. A time step is represented as a tuple, and can be thought of as a column on a quantum circuit diagram. Within this time step, three hadamard gates are defined in a list as the first element of our time step. The second element of our time step are the positions of those respective gates. The positions label each 'wire' from the top of the circuit to the bottom.

The list of time steps, `init_states`, is fed to `qc.add_gate_to_circuit`. This function will add the list of time steps onto a queue. This doesn't build anything; one can imagine that by calling this function a blue print is being constructed for this quantum circuit.

Next, an oracle is added:
```python
 oracle = [
    (["CZ"], [[0, 2]])
]
qc.add_gate_to_circuit(oracle)
```
This contains a time step that adds a double gate to the circuit. The control qubit is `0`, and the target qubit is `2`. The order of control and target qubits are the same for every double gate. The next step in Grover's algorithm is amplitude amplification. This can start to be implemented like so:
```python
half_of_amplification = [
    (["H", "H", "H"], [[0], [1], [2]]),
    (["X", "X", "X"], [[0], [1], [2]]),
    (["H"], [[2]])
]
qc.add_gate_to_circuit(half_of_amplification)
```
The `qc.add_gate_to_circuit` function can handle multiple time steps of a varying number of gates. 

For the amplitude amplification, a CCz gate needs to added. This can be done by attaching a hadamard gate to each side of the target qubit of a toffoli gate. The toffoli gate (also known as CCnot) can be implemented using CV and Hadamard gates. To do so requires a long chain of gates, so let's define a new function:
```python
def CCnot(control_1: int, control_2: int, target: int) -> list:
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
```
This can also be used as a template for other complicated gates that are built from elementary ones. The construction of the blue print for our circuit is finished by adding
```python
qc.add_gate_to_circuit(CCnot(0, 1, 2), add_gate_name="T"),
qc.add_gate_to_circuit(half_of_amplification[::-1]) 
```
The first line uses an additional parameter; this adds a gate name that will encapsulate all those gates defined in the CCnot function. This **cannot** be used for calling CCnot in the future, but it will condense all the gates defined in the CCnot function when printed to the screen. The final line adds the mirror image of the previously defined list, as amplitude amplification is symmetric about the CCnot gate.

To check that blue print of the circuit was constructed correctly, it's best to print the circuit to the terminal screen now. This is good practice before running the simulation of the circuit; especially if it's a large circuit that may take a long time to simulate. This is easily done by calling:
```python
qc.print_circuit()
```
The python file should be run from the terminal, as running from an IDE may cause issues with printing the circuit. The printed circuit should look like this:
```angular2html
    --H----•----H----X---------░---------X----H--
           |                   ░
    --H---------H----X---------T---------X----H--
           |                   ░
    --H----Z----H----X----H----░----H----X----H--
```
Note that the CCNot gate that was added with the label 'T' shows up in a hashed format. This will always happen for custom gates, and will always span the length of the wires. Now having consolidated that the blue prints are correct, the circuit can be built using
```python
qc.build_circuit()
```
Need to decide what happens with measuring...

## Key distribution
The file `KeyDistributionGeneral.py` contains the quantum key distrbution programme based on the BB84 protocol. This can be run in either your ide or command line as desceribed before.

Once initiated the programme will start by asking which matrix type you would like to use.
```python
t = str(input('What type of matrix object do you want to use? Type D for dense, S for sparse, L for lazy: '))
```
Next the programme needs to know how many qubits will be used in the encryption:
```python
n = int(input('How long would person A like their bit message to be?: '))
```
Now the programme will call the QuantumComputer class and pass the matrix type as well as the number of qubits to be used initalising this and storing it in the variable qc:
```python
    global qc
    if t == "D" or t == "d":
        qc = QuantumComputer(n, 'Dense')
    if t == "S" or t == "s":
        qc = QuantumComputer(n, 'Sparse')
    if t == "L" or t == "l":
        qc = QuantumComputer(n, 'Lazy')
```
The register is then called and both the bits and the basis are setup to be vectors filled with n random numbers.
