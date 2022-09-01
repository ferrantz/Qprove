# questo esercizio mostra il fenomeno dell'interferenza mettendo tra un gate di Pauli due gate H

# Pauli X (|0> diventa |1> e |1> diventa |0>) -> facendo il 'sandwich' di H |0> diventa 0 e |1> diventa 1
# Pauli Y (|0> diventa i|1> e |1> diventa -i|0>) -> facendo il 'sandwich' di H |0> diventa 1 e |1> diventa 0
# Pauli Z (|0> resta |0> e |1> diventa -|1>) -> facendo il 'sandwich' di H |0> diventa 1 e 1 e |1> diventa 0

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi
from qiskit.providers.aer import AerSimulator

sim = AerSimulator()
qreg_q = QuantumRegister(1, 'q')
creg_c = ClassicalRegister(1, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)
circuit.x(0)
circuit.h(0)
circuit.rz(pi, 0)
circuit.h(0)
circuit.measure(0, 0)
print(circuit)
job = sim.run(circuit)      
result = job.result() 
result.get_counts()   
print(result.get_counts())