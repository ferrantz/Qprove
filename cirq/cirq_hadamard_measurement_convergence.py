import cirq
from matplotlib import pyplot as plt

def hadamard_state_measurements(copies):
    qubit = cirq.GridQubit(0, 0)
    circuit = cirq.Circuit([cirq.H(qubit), cirq.measure(qubit, key = 'm')])
    print('Circuito:')
    print(circuit)
    sim = cirq.Simulator()
    output = sim.run(circuit, repetitions = copies)
    res = output.histogram(key = 'm')
    prob_0 = dict(res)[0]/copies
    print(prob_0)
    return prob_0

def main(copies_low = 10, copies_high = 1000):
    probability_for_zero_state_trace = []
    copies_trace = []
    for n in range(copies_low, copies_high):
        copies_trace.append(n)
        prob_0 = hadamard_state_measurements(n)
        probability_for_zero_state_trace.append(prob_0)
    plt.plot(copies_trace, probability_for_zero_state_trace)
    plt.xlabel('Numero di misurazioni')
    plt.ylabel('Probabilità di misurare lo stato 0')
    plt.title('Sequenza di convergenza della probabilità per lo stato 0')
    plt.show()

if __name__ == '__main__':
    main()        