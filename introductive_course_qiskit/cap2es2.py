# non mi trovo col risultato, per me fa 01

# Problema dell'1+1

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi
from qiskit.providers.aer import AerSimulator

sim = AerSimulator()
qreg_q = QuantumRegister(4, 'q')
creg_c = ClassicalRegister(2, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.x(qreg_q[0])
circuit.x(qreg_q[1])
circuit.cx(qreg_q[0], qreg_q[2])
circuit.cx(qreg_q[1], qreg_q[2])
circuit.ccx(qreg_q[0], qreg_q[1], qreg_q[3])
circuit.measure(qreg_q[2], creg_c[0])
circuit.measure(qreg_q[3], creg_c[1])
print(circuit)
job = sim.run(circuit)      
result = job.result() 
result.get_counts()   
print(result.get_counts())