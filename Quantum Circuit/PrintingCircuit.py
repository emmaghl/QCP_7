import os
import sys

class PrintingCircuit():

    def __init__(self, circuit: list, num_qubits: int, SPACE_HOZ: int = 5, SPACE_HOZ_MIDDLE: int = 3):
        '''WARNING: Need to call `print_circuit_ascii` from terminal/cmd and will clear the terminal screen.
        Prints the quantum circuit in an ascii format on the terminal.

        @param circuit The quantum circuit that is given by the Quantum Computer.py class
        @param num_qubits The number of qubits in the circuit.
        @param SPACE_HOZ The horizontal spaces between the gates.
        @param SPACE_HOZ_MIDDLE The horizontal space that the gate is placed from the emnd of the other gate. Gives
            the effect of centering the gate.
        '''
        if SPACE_HOZ_MIDDLE >= SPACE_HOZ:
            raise Exception(f"Error thrown due to {SPACE_HOZ_MIDDLE} >= {SPACE_HOZ}. "
                            f"Horizontal space between gates must be larger than setting the horizontal middle.")

        self.circuit = circuit
        self.num_qubits = num_qubits

        self.__SPACE_HOZ = SPACE_HOZ
        self.__SPACE_HOZ_MIDDLE = SPACE_HOZ_MIDDLE

    def print_circuit_ascii(self):
        '''WARNING: Needs to be run from terminal/cmd, and clears the screen.'''
        self.__clear_screen()

        circuit_string = ""
        circuit_length = len(self.circuit)

        # Creates the horizontal wires (not most efficient way to concatenate strings)
        for i in range(self.num_qubits):
            circuit_string += f"|0> " + "-"*(circuit_length*5) + "\n\n"
        print(circuit_string)

        # Go through each time step, and print characters
        for x in range(circuit_length):
            gate = self.circuit[x] # Get gate operator
            if len(gate[1][0]) > 1: # Check if double or single gate
                self.__print_doubleGate(gate, x)
            else:
                self.__print_singleGate(gate, x)

    def __print_at(self, x: int, y:int , text: str):
        '''Moves the terminal cursor to (x,y) and prints 'text'.'''
        sys.stdout.write("\x1b7\x1b[%d;%df%s\x1b8" % (y+1, x+4, text))
        sys.stdout.flush()

    def __clear_screen(self, ):
        os.system("cls") if os.name == "nt" else os.system("clear")

    def __print_singleGate(self, time_step: list, x: int):
        for i in range(len(time_step[0])): #Loop through all single gates
            self.__print_at(x*self.__SPACE_HOZ+self.__SPACE_HOZ_MIDDLE, time_step[1][i][0]*2, time_step[0][i][0])

    def __connect_nodes(self, x_terminal_pos: int, y_pos: int, control: str, target: str = "\u2295"):
        self.__print_at(x_terminal_pos, y_pos[1]*2, target)
        self.__print_at(x_terminal_pos, y_pos[0]*2, control)
        for i in range(abs(y_pos[0] - y_pos[1])): # Create line connecting target and control
            self.__print_at(x_terminal_pos, min(y_pos[0],y_pos[1])*2 + 2*i + 1, "|")

    def __print_doubleGate(self, time_step: list, x: int):
        x_terminal_position = x*self.__SPACE_HOZ+self.__SPACE_HOZ_MIDDLE
        y_positions = time_step[1][0]

        if time_step[0][0] == "CNOT":
            self.__connect_nodes(x_terminal_position, y_positions, "\u2022")
        elif time_step[0][0][0] == "C": # Get first letter of another Controlled gate, and print second letter
            self.__connect_nodes(x_terminal_position, y_positions, time_step[0][0][1:])
        else:
            for i in range(abs(y_positions[0] - y_positions[1])*2+1): # Create shaded block to repreent function.
                self.__print_at(x_terminal_position, min(y_positions[0],y_positions[1]) + i, "\u2591")
            self.__print_at(x_terminal_position,
                     int(abs(y_positions[0]-y_positions[1])/2)+min(y_positions[0],y_positions[1])+1,
                     time_step[0][0])




