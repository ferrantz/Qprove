from qiskit import *
import math
from numpy import pi

IBMQ.load_account()
provider_reale = IBMQ.get_provider(hub = 'partner-cnr', group = 'icar-napoli', project = 'ferrante')
# provider_reale = IBMQ.get_provider(hub = 'ibm-q', group = 'open', project = 'main')
backend_reale = provider_reale.get_backend('ibmq_montreal') # prova a vedere che succede se cambi il backend
job = backend_reale.retrieve_job('62c6eeb81e73ff1eeaf613f1')

job_stats = job.result().get_counts()
freq_job_stats = {k: v / 5000 for k, v in job_stats.items()}

# print(freq_job_stats)
variabile_a = freq_job_stats['00010']**2 + freq_job_stats['11100']**2 + freq_job_stats['11111']**2 + freq_job_stats['00001']**2 + freq_job_stats['11000']**2 + freq_job_stats['00000']**2 + freq_job_stats['11011']**2 + freq_job_stats['00100']**2 + freq_job_stats['00011']**2 + freq_job_stats['11110']**2 + freq_job_stats['00110']**2 + freq_job_stats['00111']**2 + freq_job_stats['11001']**2 + freq_job_stats['00101']**2 + freq_job_stats['11101']**2 + freq_job_stats['11010']**2 
del freq_job_stats['00010']
del freq_job_stats['11100']
del freq_job_stats['11111']
del freq_job_stats['00001']
del freq_job_stats['11000']
del freq_job_stats['00000']
del freq_job_stats['11011']
del freq_job_stats['00100']
del freq_job_stats['00011']
del freq_job_stats['11110']
del freq_job_stats['00110']
del freq_job_stats['00111']
del freq_job_stats['11001']
del freq_job_stats['00101']
del freq_job_stats['11101']
del freq_job_stats['11010']
variabile_b = sum({k: v**2 for k, v in freq_job_stats.items()}.values())
variabile_c = 32 # len(freq_job_stats.keys())
potenza_segnale = variabile_a / variabile_c
potenza_rumore = variabile_b / variabile_c
SNR = potenza_segnale / potenza_rumore
if SNR != 0:
    SNR_db = 10 * math.log(SNR, 10)
else:
    SNR_db = 'NaN'

### circuito transpilato qui


# print('depth: ' + str(circuit.depth()))
print('SNR_db: ' + str(SNR_db))