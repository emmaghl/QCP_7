# Quantum Computer Simulator
This python library allows functionality to build, print and simulate a quantum computer. The simulated quantum computer can be chosen to implement dense and lazy methods.

## Installation and Testing
###Using Git
Open terminal and change into a directory that you wish the library to be downloaded in with the command
```angular2html
cd directory_name
```
You can list the available directories (and files within the current directory you're in) by entering the command (for MacOs/Linux):
```angular2html
ls
```
or 
```angular2html
dir
```
for windows. Then, create a new folder (such as called `QuantumComputer`) with:
```angular2html
mkdir QuantumComputer
```
Enter this folder with
```angular2html
cd QuantumComputer
```
This folder is where the git repository will be downloaded into. This can be done by installing git (https://github.com/git-guides/install-git) and entering the command:
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
The `sample.py` file contains three implementations of Grover's circuit, with increasing number of qubits. The three functions, in order of increasing qubits needed, are called: 
1. `GroverAlgorithm_3Qubit`
2. `GroverAlgorithm_SingleRow_BinaryCol_Suduko`
3. `GroverAlgorithm_SingleRow_Suduko`
 
Let's walk through the first function, `GroverAlgorithm_3Qubit`, to gain appreciation in thinking and logic behind using this library to simulate a quantum computer.

First, in a new python file, we want to import this library. If the new python file was created in the directory where `sample.py` is, then the import statement would be
```python
from QuantumComputerSimulator import QuantumComputer
```
This can be changed depending where the python file was created relative to the `QuantumComputerSimulator' folder. Next, we want to instantiate our a new quantum computer using 3 qubits.
```python
qc = QuantumComputer(3)
```
By default, the quantum computer will use a dense methods which are slower than other techniques, however this is more intuitive from a quantum mechanics point of view as the gates are interpreted as matricies. Now we can start to build our circuit. For Grover's, we need to put the circuit into a superposition of equal weights. This can be achieved by acting three Hadamard gates to each qubit. This can with:
```python
init_states = [
    (["H", "H", "H"], [[0], [1], [2]])
]
qc.add_gate_to_circuit(init_states)
```
Let's unpack this code. We first define a list, `init_states`, which holds a single *time step*. A time step is represented as a tuple. Within this time step, we define three hadamard gates in a list, as the first element of our time step. The second element of our time step are the positions of those gates. The postitions label each 'wire' from the top of the circuit to the bottom. Then, the list of time steps, `init_states`, is fed to `qc.add_gate_to_circuit(init_states)`. This function will add these gates onto a queue. This hasn't built anything yet; one can imagine that we're simply building a blue print for this quantum circuit, and the `qc.add_gate_to_circuit(init_states)` function adds gates to the blue print.

Next we add an oracle:
```python
 oracle = [
    (["CZ"], [[0, 2]])
]
qc.add_gate_to_circuit(oracle)
```
This contains a time step that adds a double gate to the circuit. The control bit is `0`, and the target qubit is `1`. This is the same ordering of control and target qubits for every double gate.

Following Grover's algorithm, we add the section where the amplitudes amplify,
```python
half_of_amplification = [
    (["H", "H", "H"], [[0], [1], [2]]),
    (["X", "X", "X"], [[0], [1], [2]]),
    (["H"], [[2]])
]
qc.add_gate_to_circuit(half_of_amplification)
```
As you can see, the `qc.add_gate_to_circuit` function can handle multiple time steps of varying number of gates. Adding to the amplitude amplification, we need to implement a CCz gate. This can be done with attatching a hadamard gate to each side of the target qubit of a toffoli gate. The toffoli gate (also known as CCnot) can be implemented using CV and Hadarmard gates. This is a long chain of commands, so we choose to put this in a function:
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
This can also be used as a template for other gates that can be built from elementary ones. Now, we can finish building the blue prints for our circuit,
```python
qc.add_gate_to_circuit(CCnot(0, 1, 2), add_gate_name="T"),
qc.add_gate_to_circuit(half_of_amplification[::-1]) 
```
The first line uses an additional parameter; this adds a gate name that will encapsulate all those gates. This can't be used for calling CCnot in the future, but it will condense the circuit when printed to the screen. The final line adds the mirror image of the previously defined list, as amplitude amplification is symmetric about the CCnot gate.

Now, it's best to check if we've built the circuit as we've wanted. This is easily done by calling:
```python
qc.print_circuit()
```
You should run the python file from the terminal, and make sure that we've ended up with the gate that we wanted! The printed circuit should look like this:
```angular2html
|0> --H----•----H----X---------░---------X----H--
           |                   ░
|0> --H---------H----X---------T---------X----H--
           |                   ░
|0> --H----Z----H----X----H----░----H----X----H--
```
If we're happy, with this blue print, we can now buil the circuit by calling
```python
qc.build_circuit()
```
Need to decide what happens with measuring...

## Python Version and OS
Been tested to work with Python 3.11.2 on MacOS. See `Requirements.txt` file to see the python libraries and versions needed to run this code. The library should also run for Linux and Windows operating systems; the only reason why it wouldn't is that `PrintCircuit.py` clears the screen when the circuit is printed using the terminal/cmd. This command varies across operating systems, however the library should account for this. 

