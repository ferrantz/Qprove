import numpy as np
# import pandas as pd
# import yfinance as yf
# import matplotlib.pyplot as plt
# from qiskit.utils import algorithm_globals
# from qiskit.algorithms import VQE
# from qiskit.algorithms.optimizers import COBYLA
# from qiskit import Aer
# from qiskit.circuit.library import TwoLocal
# from qiskit.utils import QuantumInstance
# from qiskit_optimization.algorithms import MinimumEigenOptimizer
# from qiskit_finance.applications.optimization import PortfolioOptimization
# from qiskit_optimization.converters import QuadraticProgramToQubo
# from qiskit import IBMQ, execute, transpile, assemble
# import time 
import ast



from qiskit import *

# 2 ASSET

with open(r"C:/Users/italo/Desktop/Qprove/nove_asset") as f:
    lines = f.readlines()
    str_lines_1, str_lines_2 = str(lines).replace('[', '').replace(']', '').replace('"', '').replace('[', '').split('}')
    str_lines_1 = str_lines_1 + '}'
    dict_r_p = ast.literal_eval(str_lines_1)
    str_lines_2 = float(str_lines_2)
    dict_r_p['tempo_esecuzione'] = str_lines_2

IBMQ.load_account()
provider_reale = IBMQ.get_provider(hub = 'partner-cnr', group = 'icar-napoli', project = 'ferrante')
backend_reale = provider_reale.get_backend('ibmq_montreal') # prova a vedere che succede se cambi il backend
job = backend_reale.retrieve_job('62b9608cd8901d485b4b9751')
risultati_intermedi = str(job.result()).split('counts=')[1].split('),')[0]
n_qubit = int(np.log2(risultati_intermedi.count(':')))
lista_statevector = []
# print('n_qubit: ' + str(n_qubit))
for numero_decimale in range(2**(n_qubit)):
    numero_binario = bin(numero_decimale).replace("0b", "")
    if len(numero_binario) < n_qubit:
        zeri_da_appendere = n_qubit - len(numero_binario)
        numero_binario = zeri_da_appendere * '0' + numero_binario
    lista_statevector.append(numero_binario)

dict_r_p_2 = {}
risultati_intermedi_2 = ast.literal_eval(risultati_intermedi)
for y in range(len(lista_statevector)):
    dict_r_p_2[lista_statevector[y]] = list(risultati_intermedi_2.items())[y][1] # [1] perché è il valore

shots = int(str(job.result()).split('shots=')[1].split(',')[0])
lista_valori_divisi = []
for v in dict_r_p_2.values():
    rapporto = v / shots
    lista_valori_divisi.append(rapporto)

for key, val in zip(dict_r_p_2, lista_valori_divisi): # assegna il rapporto a 1000 dei valori
    dict_r_p_2[key] = val

dict_r_p_2['tempo_esecuzione'] = (job.time_per_step()['COMPLETED'] - job.time_per_step()['RUNNING']).total_seconds()

print(dict_r_p)
print('----------------------------------------------------')
print(dict_r_p_2)