# bug

import cirq
import numpy as np

def oracle(data_reg, y_reg, circuit, is_balanced = True):
    if is_balanced:
        print('data_reg = ' + str(data_reg))
        print('y_reg = ' + str(y_reg))
        print('data_reg[0] = ' + str(data_reg[0]))
        print('data_reg[1] = ' + str(data_reg[1]))
        print('y_reg = ' + str(y_reg))
        circuit.append([cirq.CNOT(data_reg[0], y_reg), cirq.CNOT(data_reg[1], y_reg)])
    return circuit

def deutch_jozsa(domain_size: int, func_type_to_simulate: str = 'balanced', copies: int = 100):
    '''
    :param domain_size: numero di input nella funzione
    :param oracle: oracolo che simula la funzione
    :return: se la funzione è bilanciata o costante
    '''
    # definisci il data register e il qubit target
    reqd_num_qubits = int(np.ceil(np.log2(domain_size))) # qubit necessari (log2 di n)
    # definisci i qubit di input
    data_reg = [cirq.LineQubit(c) for c in range(reqd_num_qubits)]
    # definisci i qubit target
    y_reg = cirq.LineQubit(reqd_num_qubits)
    # definisci il circuito
    circuit = cirq.Circuit()
    # definisci lo stato di superposizione uguale per i qubit in input
    circuit.append(cirq.H(data_reg[c]) for c in range(reqd_num_qubits))
    # definisci il superposition state - (meno)
    circuit.append(cirq.X(y_reg)) # prima faccio CNOT...
    circuit.append(cirq.H(y_reg)) # ... poi H
    # check per la natura della funzione: bilanciato/costante per simulare ed implementare l'oracolo
    if func_type_to_simulate == 'balanced':
        is_balanced = True
    else:
        is_balanced = False
    circuit = oracle(data_reg, y_reg, circuit, is_balanced=is_balanced)
    # applica la trasformata di Hadamard ad ognuno dei qubit di input
    circuit.append(cirq.H(data_reg[c] for c in range(reqd_num_qubits)))
    # misura i qubit di input
    circuit.append(circuit.measure(*data_reg, key = 'z'))
    print('Diagramma del circuito')
    print(circuit)
    sim = cirq.Simulator()
    result = sim.run(circuit, repetitions = copies)
    print(result.histogram(key = 'z'))

if __name__ == '__main__':
    print('Esegui Deutsch-Josza per una funzione bilanciata di domain size 4')
    deutch_jozsa(domain_size=4, func_type_to_simulate='balanced', copies = 1000)
    print('Esegui Deutsch Josza per una funzione costante con domain size 4')
    deutch_jozsa(domain_size=4, func_type_to_simulate='', copies = 1000)