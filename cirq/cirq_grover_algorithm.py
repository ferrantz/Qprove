import cirq
import numpy as np

def oracle(input_qubits, target_qubit, circuit, secret_element = '01'):
    print('Elemento da cercare: {secret_element}')
    # flippa i qubit corrispondenti ai bit contenenti 0 
    for i, bit in enumerate(secret_element):
        if int(bit) == 0:
            circuit.append(cirq.X(input_qubits[i]))
    # fa un conditional NOT usando tutti tutti gli input qubit come control qubit
    circuit.append(cirq.TOFFOLI(*input_qubits, target_qubit))
    # reversa gli input qubit allo stato iniziale prima del flipping
    for i, bit in enumerate(secret_element):
        if int(bit) == 0:
            circuit.append(cirq.X(input_qubits[i]))
    return circuit

def grovers_algorithm(num_qubits = 2, copies = 1000):
    # definisci input e target qubit
    input_qubits = [cirq.LineQubit(i) for i in range(num_qubits)]
    target_qubit = cirq.LineQubit(num_qubits)
    # definisci il circuito quantistico
    circuit = cirq.Circuit()
    # crea un equal superposition state
    circuit.append([cirq.H(input_qubits[i]) for i in range(num_qubits)])
    # porta il target qubit al |-> state
    circuit.append([cirq.X(target_qubit), cirq.H(target_qubit)])
    # passa il qubit attraverso l'oracolo
    circuit = oracle(input_qubits, target_qubit, circuit)
    # costruisci l'operatore di Grover
    circuit.append(cirq.H.on_each(*input_qubits))
    circuit.append(cirq.X.on_each(*input_qubits))
    circuit.append(cirq.H.on(input_qubits[1]))
    circuit.append(cirq.CNOT(input_qubits[0], input_qubits[1]))
    circuit.append(cirq.H.on(input_qubits[1]))
    circuit.append(cirq.X.on_each(*input_qubits))
    circuit.append(cirq.H.on_each(*input_qubits))
    # misura il risultato
    circuit.append(cirq.measure(*input_qubits, key = 'Z'))
    print('Algoritmo di Grover')
    print(circuit)
    sim = cirq.Simulator()
    result = sim.run(circuit, repetitions = copies)
    out = result.histogram(key = 'Z')
    out_result = {}
    for k in out.keys():
        new_key = "{0:b}".format(k)
        if len(new_key) < num_qubits:
            new_key = (num_qubits - len(new_key))*'0' + new_key
        out_result[new_key] = out[k]
    print(out_result)

if __name__ == '__main__':
    grovers_algorithm(2)