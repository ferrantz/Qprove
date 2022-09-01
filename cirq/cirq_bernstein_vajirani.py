import cirq
import numpy as np

def func_bit_pattern(num_qubits):

    '''Crea i parametri della funzione oracle'''

    bit_pattern = []
    for i in range(num_qubits):
        bit_pattern.append(np.random.randint(0, 2))
    print(f"Function bit pattern: \
        {''.join([str(x) for x in bit_pattern])}")
    return bit_pattern

def oracle(input_qubits, target_qubit, circuit, num_qubits, bit_pattern):

    '''Definisce l'oracolo'''

    for i in range(num_qubits):
        if bit_pattern[i] == 1:
            circuit.append(cirq.CNOT(input_qubits[i], target_qubit))
    return circuit

def BV_algorithm(num_qubits, bit_pattern):
    input_qubits = [cirq.LineQubit(i) for i in range(num_qubits)]
    target_qubit = cirq.LineQubit(num_qubits)
    circuit = cirq.Circuit()
    circuit.append([cirq.H(input_qubits[i]) for i in range(num_qubits)])
    circuit.append([cirq.X(target_qubit), cirq.H(target_qubit)])
    circuit = oracle(input_qubits, target_qubit, circuit, num_qubits, bit_pattern)
    circuit.append([cirq.H(input_qubits[i]) for i in range(num_qubits)])
    circuit.append(cirq.measure(*input_qubits, key = 'Z'))
    print('Diagramma del circuito di Bernstein Vajirani')
    print(circuit)
    sim = cirq.Simulator()
    results = sim.run(circuit, repetitions = 1000)
    results = dict(results.histogram(key = 'Z'))
    print(results)
    results_binary = {}
    for k in results.keys():
        results_binary["{0:b}".format(k)] = results[k]
    print('Distribuzione di bit pattern per BV')
    print(results_binary)

def main(num_qubits = 6, bit_pattern = None):
    if bit_pattern is None:
        bit_pattern = func_bit_pattern(num_qubits)
    BV_algorithm(num_qubits, bit_pattern)

if __name__ == '__main__':
    main()