# Misurazione dopo un Hadamard su un qubit

import cirq

qubit = cirq.GridQubit(0, 0)

circuit = cirq.Circuit([cirq.H(qubit), cirq.measure(qubit, key= 'm')])
print(circuit)

sim = cirq.Simulator()
output = sim.run(circuit, repetitions = 100)
print('Output della misurazione: ')
#print(output)
print('Istogramma:')
print(output.histogram(key = 'm'))