# Quantum Computer Simulator
This python library can build, print and simulate a quantum computer. The simulated quantum computer can be chosen to implement dense (lazy and sparse?) methods.

## Python Version and OS
Been tested to work with Python 3.11.2 on MacOS. See `Requirements.txt` file to see the python libraries and versions needed to run this code. The library should also run for Linux and Windows operating systems; the only reason why it wouldn't is that `PrintCircuit.py` clears the screen when the circuit is printed using the terminal/cmd. This command varies across operating systems, however the library should account for this. 

## Installation and Testing
### Using Git
Open terminal and change into a directory that you wish the library to be downloaded in. This can be accomplished with the command
```angular2html
cd directory_name
```
To list the available directories (and files within the current directory you're in), use  the following command if you're using for MacOs/Linux:
```angular2html
ls
```
or 
```angular2html
dir
```
for Windows. This can help you navigate between directories using the terminal. Then, create a new folder (such as called `QuantumComputer`) with:
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
Enter the repository with the `cd` command, ahd locate the `sample.py` file and run:
```angular2html
python3 sample.py --test
```
This assumes that you have `python3` added to your path (https://realpython.com/add-python-to-path/). If successful, this should complete without any errors, and validate that all functions in the `QuantumComputerSimulator` library are working as they should be.

## Documentation
The documentation can be found in the [docs folder](QuantumComputerSimulator/docs/QuantumComputerSimulator). To view the documentation, open `index.html` with your favourite web browser. 

## Quick Start Guide
The `sample.py` file contains three implementations of Grover's circuit. The three functions, in order of increasing qubits that are used, are called: 
1. `GroverAlgorithm_3Qubit`
2. `GroverAlgorithm_SingleRow_BinaryCol_Suduko`
3. `GroverAlgorithm_SingleRow_Suduko`
 
Let's implement the first function, `GroverAlgorithm_3Qubit`, to gain an appreciation in the thinking and logic behind using this library to simulate a quantum computer. 

In a new python file, we first want to import this library. If the new python file was created in the directory where `sample.py` is, then the import statement would be
```python
from QuantumComputerSimulator import QuantumComputer
```
This can be changed depending where the python file was created relative to the `QuantumComputerSimulator` folder. Next, a 3 qubit quantum computer is instantiated:
```python
qc = QuantumComputer(3)
```
By default, the quantum computer will use dense methods which are slower than other techniques, however this is more 'intuitive'. After instantiating the computer, the circuit can start to be built. For Grover's, the circuit needs to put the register into a superposition of equal weights. This can be achieved by acting three Hadamard gates to each qubit. The following code does this:
```python
init_states = [
    (["H", "H", "H"], [[0], [1], [2]])
]
qc.add_gate_to_circuit(init_states)
```
Let's unpack this. A list, `init_states`, is defined first which holds a single *time step*. A time step is represented as a tuple, and can be thought of as a column on a quantum circuit. Within this time step, three hadamard gates are defined in a list as the first element of our time step. The second element of our time step are the positions of those gates. The positions label each 'wire' from the top of the circuit to the bottom. Then, the list of time steps, `init_states`, is fed to `qc.add_gate_to_circuit(init_states)`. This function will add a list of time steps onto a queue. This doesn't build anything; one can imagine that by calling this function a blue print is being constructed for this quantum circuit.

Next we add an oracle:
```python
 oracle = [
    (["CZ"], [[0, 2]])
]
qc.add_gate_to_circuit(oracle)
```
This contains a time step that adds a double gate to the circuit. The control bit is `0`, and the target qubit is `1`. This is the same order of control and target qubits for every double gate. Following Grover's algorithm, the amplitude amplification can start to be added,
```python
half_of_amplification = [
    (["H", "H", "H"], [[0], [1], [2]]),
    (["X", "X", "X"], [[0], [1], [2]]),
    (["H"], [[2]])
]
qc.add_gate_to_circuit(half_of_amplification)
```
As you can see, the `qc.add_gate_to_circuit` function can handle multiple time steps of varying number of gates. For the amplitude amplification, a CCz gate needs to added. This can be done by attatching a hadamard gate to each side of the target qubit of a toffoli gate. The toffoli gate (also known as CCnot) can be implemented using CV and Hadamard gates. This is a long chain of commands, so we choose to put this in a function:
```python
def CCnot(control_1, control_2, target) -> list:
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
This can also be used as a template for other gates that can be built from elementary ones. Now, the blue prints for our circuit can be finished by adding,
```python
qc.add_gate_to_circuit(CCnot(0, 1, 2), add_gate_name="T"),
qc.add_gate_to_circuit(half_of_amplification[::-1]) 
```
The first line uses an additional parameter; this adds a gate name that will encapsulate all those gates defined in thre CCnot function. This can't be used for calling CCnot in the future, but it will condense the circuit when printed to the screen. The final line adds the mirror image of the previously defined list, as amplitude amplification is symmetric about the CCnot gate.

To check that the circuit is what is expected, it's best to print the circuit to the terminal screen now; before commiting to running the simulation. This is easily done by calling:
```python
qc.print_circuit()
```
The python file should be run from the terminal now, to make sure that the circuit has been implemented correctly! The printed circuit should look like this:
```angular2html
|0> --H----•----H----X---------░---------X----H--
           |                   ░
|0> --H---------H----X---------T---------X----H--
           |                   ░
|0> --H----Z----H----X----H----░----H----X----H--
```
Note, that the CCNot gate that was added with the label 'T' shows up in a hashed format. This will always happen for custom gates, and will always span the length of the wires. If the output is correct, the circuit can be built using
```python
qc.build_circuit()
```
Need to decide what happens with measuring...
