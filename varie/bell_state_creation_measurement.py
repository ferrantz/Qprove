import numpy as np
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram

simulator = Aer.get_backend('qasm_simulator')
circuit = QuantumCircuit(2, 2)
circuit.h(0) # aggiunge un H gate sul qubit 0
circuit.cx(0, 1) # aggiunge un CNOT gate sul qubit di controllo 0 e su quello target 1
circuit.measure([0, 1], [0, 1]) # mappa la misurazione quantistica in bit classici
job = execute(circuit, simulator, shots = 100)
result = job.result()
counts = result.get_counts(circuit)
print('\n Totale per 00 e 11:', counts)
print(circuit.draw(output = 'text'))