from qiskit import *
import math
from numpy import pi

IBMQ.load_account()
# provider_reale = IBMQ.get_provider(hub = 'partner-cnr', group = 'icar-napoli', project = 'ferrante')
provider_reale = IBMQ.get_provider(hub = 'ibm-q', group = 'open', project = 'main')
backend_reale = provider_reale.get_backend('ibmq_belem') # prova a vedere che succede se cambi il backend
job = backend_reale.retrieve_job('62c6aa32beddc2043c2c0edb')
job_stats = job.result().get_counts()
freq_job_stats = {k: v / 5000 for k, v in job_stats.items()}
variabile_a = freq_job_stats['00']**2 + freq_job_stats['11']**2 
del freq_job_stats['00']
del freq_job_stats['11']
variabile_b = sum({k: v**2 for k, v in freq_job_stats.items()}.values())
variabile_c = len(freq_job_stats.keys())
potenza_segnale = variabile_a / variabile_c
potenza_rumore = variabile_b / variabile_c
SNR = potenza_segnale / potenza_rumore
if SNR != 0:
    SNR_db = 10 * math.log(SNR, 10)
else:
    SNR_db = 'NaN'

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from numpy import pi

qreg_q = QuantumRegister(5, 'q')
creg_c = ClassicalRegister(2, 'c')
circuit = QuantumCircuit(qreg_q, creg_c)

circuit.rz(1.5707963267948966, qreg_q[0])
circuit.sx(qreg_q[0])
circuit.rz(1.5707963267948966, qreg_q[0])
circuit.cx(qreg_q[0], qreg_q[1])
circuit.measure(qreg_q[0], creg_c[0])
circuit.measure(qreg_q[1], creg_c[1])

print('depth: ' + str(circuit.depth()))
print('SNR_db: ' + str(SNR_db))