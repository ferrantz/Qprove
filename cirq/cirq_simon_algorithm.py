import cirq
import numpy as np

def oracle(input_qubits, target_qubits, circuit):

    '''Oracolo per il codice segreto 110'''

    circuit.append(cirq.CNOT(input_qubits[2], target_qubits[1]))
    circuit.append(cirq.X(target_qubits[0]))
    circuit.append(cirq.CNOT(input_qubits[2], target_qubits[0]))
    circuit.append(cirq.CCNOT(input_qubits[0], input_qubits[1], target_qubits[0])) # CCNOT è il Toffoli Gate -> se i primi due sono |11>, allora flippa il terzo qubit
    circuit.append(cirq.X(input_qubits[0]))
    circuit.append(cirq.X(input_qubits[1]))
    circuit.append(cirq.CCNOT(input_qubits[0], input_qubits[1], target_qubits[0]))
    circuit.append(cirq.X(input_qubits[0]))
    circuit.append(cirq.X(input_qubits[1]))
    circuit.append(cirq.X(target_qubits[0]))
    return circuit

def simons_algorithm_circuit(num_qubits = 3, copies = 1000):

    '''Costruisce il circuito per l'Algoritmo di Simon
    :param num_qubits:
    :return: cirq circuit
    '''

    input_qubits = [cirq.LineQubit(i) for i in range(num_qubits)]
    target_qubits = [cirq.LineQubit(k) for k in range(num_qubits, 2*num_qubits)]
    circuit = cirq.Circuit()
    # crea il superposition state uguale per i qubit in input attraverso la trasformata di Hadamard
    circuit.append([cirq.H(input_qubits[i]) for i in range(num_qubits)])
    # passa il superposition state attraverso l'oracolo
    circuit = oracle(input_qubits, target_qubits, circuit)
    # applica la trasformata di Hadamard sugli angoli dell'input
    circuit.append([cirq.H(input_qubits[i]) for i in range(num_qubits)])
    # misura i qubit input e target
    circuit.append(cirq.measure(*(input_qubits + target_qubits), key = 'Z'))
    print("Diagramma del circuito dell'Algoritmo di Simon")
    print(circuit)
    # simula l'algoritmo
    sim = cirq.Simulator()
    result = sim.run(circuit, repetitions = copies)
    out = dict(result.histogram(key = 'Z'))
    out_result = {}
    for k in out.keys():
        new_key = "{0:b}".format(k)
        if len(new_key) < 2*num_qubits:
            new_key = (2*num_qubits - len(new_key))*'0' + new_key
        out_result[new_key] = out[k]
    print(out_result)

if __name__ == '__main__':
    simons_algorithm_circuit()