# bug

import cirq
import numpy as np

def random_number_generator(low = 0, high = 2**10, m = 10):
    '''
    :param low: limite inferiore di numeri da generare
    :param high: limite superiore di numeri da generare
    :param number m: numero di numeri casuali da restituire
    :return stringa di numeri casuali 
    '''
    # determina il numero di qubit richiesti
    qubits_required = int(np.ceil(np.log2(high - low)))
    print('I qubit richiesti sono ' + str(qubits_required))
    # definisce i qubit
    Q_reg = [cirq.LineQubit(c) for c in range(qubits_required)]
    # definisce il circuito
    circuit = cirq.Circuit()
    circuit.append(cirq.H(Q_reg[c]) for c in range(qubits_required))
    circuit.append(cirq.measure(*Q_reg, key = 'z'))
    print(circuit)
    # simula il circuito
    sim = cirq.Simulator()
    num_gen = 0
    output = []
    while num_gen < m:
        result = sim.run(circuit, repetitions = 1)
        rand_number = result.data.get_values()[0][0] + low # 'DataFrame' object has no attribute 'get_values'
        if rand_number < high:
            num_gen += 1
    return output

if __name__ == '__main__':
    output = random_number_generator()
    print(output)