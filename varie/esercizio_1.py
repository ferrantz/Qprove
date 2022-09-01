# da: https://www.youtube.com/watch?v=RrUTwq5jKM4
# per usare il matrix product state: https://qiskit.org/documentation/tutorials/simulators/7_matrix_product_state_method.html
# maggiori info su simulatori e backend: https://quantum-computing.ibm.com/lab/docs/iql/manage/account/ibmq

from qiskit import *
from matplotlib import pyplot as plt # bisogna importarlo per fare il .draw()
from qiskit.tools.visualization import plot_histogram
from qiskit.tools.monitor import job_monitor

qr = QuantumRegister(2)
cr = ClassicalRegister(2)
circuit = QuantumCircuit(qr, cr)
circuit.h(qr[0])
circuit.cx(qr[0], qr[1])
circuit.measure(qr, cr)
print(circuit)
# circuit.draw(output='mpl') # per mostrare fi
# plt.show() 

# utilizzo simulatore locale
simulator = Aer.get_backend('qasm_simulator')  # Aer = simulatore in locale
result = execute(circuit, backend = simulator).result() # esegue il simulatore e restituisce i risultati nella variabile result
plot_histogram(result.get_counts(circuit))
# plt.show() 

# utilizzo risorse remote
IBMQ.load_account()
provider = IBMQ.get_provider('ibm-q')
qcomp = provider.get_backend('ibmq_quito')
job = execute(circuit, backend = qcomp)
job_monitor(job) # il job viene messo in coda con gli altri che arrivano da altre parti del mondo
result = job.result()
plot_histogram(result.get_counts(circuit))
plt.show()