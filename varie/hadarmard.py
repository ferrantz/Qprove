import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram

simulator = Aer.get_backend('qasm_simulator')

circuit = QuantumCircuit(1, 1) # crea un quantum circuit con 1 qubit
circuit.h(0) # aggiunge un H a al qubit 0
circuit.measure([0], [0]) # mappa la misurazione quantistica al register classico
job = execute(circuit, simulator, shots = 100) # esegue il circuito sul simulatore qasm
result = job.result()
counts = result.get_counts(circuit)
print('Totale di 0 e 1: ' + str(counts))
print(circuit.draw(output = 'text'))