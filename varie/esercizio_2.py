# da: https://www.youtube.com/watch?v=tBnWG_95F9c

from qiskit import *
from qiskit.tools.visualization import plot_bloch_multivector, plot_histogram
from matplotlib import pyplot as plt

circuit = QuantumCircuit(2, 2)
circuit.x(0)
simulator = Aer.get_backend('statevector_simulator')
result = execute(circuit, backend = simulator).result()
statevector = result.get_statevector()
print(statevector)
circuit.draw(output = 'mpl')
plot_bloch_multivector(statevector)
# plt.show()
circuit.measure([0], [0])
backend = Aer.get_backend('qasm_simulator')
result = execute(circuit, backend = backend, shots = 1024).result()
counts = result.get_counts()
plot_histogram(counts)
plt.show()

# per ottenere la rappresentazione matriciale
circuit = QuantumCircuit(1, 1)
circuit.x(0)
simulator = Aer.get_backend('unitary_simulator')
result = execute(circuit, backend = simulator).result()
unitary = result.get_unitary()
print(unitary)
