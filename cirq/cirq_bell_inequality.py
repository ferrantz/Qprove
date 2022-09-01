import cirq
import numpy as np

def bell_inequality_test_circuit():
    '''
    Definisce 4 qubit
    qubit 0 - Alice
    qubit 1 - contiene il bit mandato ad Alice dall'arbitro
    qubit 2 - qubit di Bob
    qubit 3 - contiene il bit mandato a Bob dall'arbitro
    :return: cirq circuit
    '''
    qubits = [cirq.LineQubit(i) for i in range(4)]
    circuit = cirq.Circuit()
    # entangla Alice e Bob al Bell state
    circuit.append([cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[2])])
    # applica X^(-0.25) al qubit di Alice
    circuit.append([cirq.X(qubits[0])**(-0.25)])
    # applica H ai qubit dell'arbitro per Alice e Bob. Si fa per randomizzare il qubit
    circuit.append([cirq.H(qubits[1]), cirq.H(qubits[3])])
    # performa un X^0.5 sui qubit di Alice e Bob basandoti sui qubit dell'arbitro
    circuit.append([cirq.CNOT(qubits[1], qubits[0])**0.5])
    circuit.append([cirq.CNOT(qubits[3], qubits[2])**0.5])
    # misura tutti i qubit
    circuit.append(cirq.measure(qubits[0], key = 'A'))
    circuit.append(cirq.measure(qubits[1], key = 'r_A'))
    circuit.append(cirq.measure(qubits[2], key = 'B'))
    circuit.append(cirq.measure(qubits[3], key = 'r_B'))
    return circuit

def main(iters = 1000):

    '''Costruisce il circuito del Bell inequality test'''

    circuit = bell_inequality_test_circuit()
    print('Circuito')
    print(circuit)
    # simula più iterazioni
    sim = cirq.Simulator()
    result = sim.run(circuit, repetitions = iters)
    A = result.measurements['A'][:, 0]
    r_A = result.measurements['r_A'][:, 0]
    B = result.measurements['B'][:, 0]
    r_B = result.measurements['r_B'][:, 0]
    win = (np.array(A) + np.array(B)) % 2 == (np.array(r_A) & np.array(r_B))
    print(f"Alice e Bob hanno vinto {100*np.mean(win)} % of times")

if __name__ == '__main__':
    main()