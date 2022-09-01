import cirq

def quantum_teleportation(qubit_to_sed_op = 'H', num_copies = 100):
    Q1, Q2, Q3 = [cirq.LineQubit(i) for i in range(3)]
    cirquit = cirq.Circuit()
    '''
    Q1: Stato del qubit che Alida vuole inviare a Bob
    Q2: Control qubit di Alice
    Q3: Control qubit di Bob
    '''
    if qubit_to_sed_op == 'H':
        cirquit.append(cirq.H(Q1)) # metà 0 e metà 1 più o meno
    elif qubit_to_sed_op == 'X':
        cirquit.append(cirq.X(Q1)) # Counter({1: 100})
    elif qubit_to_sed_op == 'Y':
        cirquit.append(cirq.Y(Q1)) # Counter({1: 100})
    elif qubit_to_sed_op == 'I':
        cirquit.append(cirq.I(Q1)) # Counter({0: 100})
    else:
        raise NotImplementedError('Ancora da implementare')

    # fa entanglement dei qubit di controllo di Alice e Bob (Q2 e Q3)
    cirquit.append(cirq.H(Q2))
    cirquit.append(cirq.CNOT(Q2, Q3))
    # CNOT i dati di Alice del qubit Q1 con il control qubit Q2
    cirquit.append(cirq.CNOT(Q1, Q2))
    # trasforma i dati di Alice del qubit Q1 in base +/- usando una trasformata di Hadamard
    cirquit.append(cirq.H(Q1))
    # misura i qubit di Alice Q1 e Q2
    cirquit.append(cirq.measure(Q1, Q2))  
    # fa un CNOT sul qubit di Bob Q3 usando il controlo qubit Q2 di Alice dopo la misurazione
    cirquit.append(cirq.CNOT(Q2, Q3))
    # fa una Z condizionata sul qubit di Bob Q3 usando il bit di controllo di Alice Q1 dopo la misurazione
    cirquit.append(cirq.CZ(Q1, Q3))
    # misura lo stato trasmesso a Bob in Q3
    cirquit.append(cirq.measure(Q3, key = 'Z'))
    print('Circuito')
    print(cirquit)
    sim = cirq.Simulator()
    output = sim.run(cirquit, repetitions = num_copies)
    print('Output della misurazione')
    print(output.histogram(key = 'Z'))

if __name__ == '__main__':
    quantum_teleportation(qubit_to_sed_op = 'H')