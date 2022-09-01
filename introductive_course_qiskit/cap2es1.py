# non capisco come avviene l'ordine della misurazione sui bit classici

from qiskit import QuantumCircuit
from matplotlib import pyplot as plt
from qiskit.providers.aer import AerSimulator
sim = AerSimulator()  # make new simulator object
qc = QuantumCircuit(3, 3)
qc.x([0, 1])
qc.i(2)
qc.measure([0, 1, 2], [1, 2, 0]) # perché così i cbit? NON MI TROVO
# qc.draw()  
# plt.show()
print(qc)
job = sim.run(qc)      
result = job.result() 
result.get_counts()   
print(result.get_counts())

# qc_2 = QuantumCircuit(3, 3)
# qc_2.x([0])
# # qc_2.measure([0, 1, 2], [2, 0, 1]) # perché così i cbit? NON MI TROVO
# # qc_2.measure(0, 0)
# # qc_2.measure(1, 1)
# # qc_2.measure(2, 2)
# print(qc_2)
# job_2 = sim.run(qc_2)      
# result_2 = job_2.result() 
# result_2.get_counts()   
# print(result_2.get_counts())