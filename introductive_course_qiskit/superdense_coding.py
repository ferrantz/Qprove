from qiskit import Aer
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

backend = Aer.get_backend('aer_simulator')

# The message
MESSAGE = '00'

# Alice encodes the message
qc_alice = QuantumCircuit(2, 2)
if MESSAGE[-1]=='1':
    qc_alice.x(0)
if MESSAGE[-2]=='1':
    qc_alice.x(1)

# then she creates entangled states
qc_alice.h(1)
qc_alice.cx(1,0)

ket = Statevector(qc_alice)
print(ket) # state vector 'entangolato'. Bob deve 'deantangolarlo'
qc_bob = QuantumCircuit(2,2)
# Bob unentangles
qc_bob.cx(0,1)
qc_bob.h(0)
# Then measures
qc_bob.measure([0,1],[0,1])

print(backend.run(qc_alice.compose(qc_bob)).result().get_counts())