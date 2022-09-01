# lo state vector di |00> è Statevector([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j], dims=(2, 2))
# con h e poi cx riesco a creare un entangled state il cui statevector è [0.70710678+0.j, 0+0.j, 0+0.j, 0.70710678+0.j]

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.providers.aer import AerSimulator

sim = AerSimulator()
qc = QuantumCircuit(2)

# entanglement con cx

# qc.h(1)
# qc.cx(1,0)
# ket = Statevector(qc)
# print(qc)
# print(ket)


qc.h(0)
qc.h(1)
qc.z(0)
qc.cx(1,0)
ket = Statevector(qc)
print(qc)
print(ket)