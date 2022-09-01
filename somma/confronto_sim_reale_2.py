from qiskit import *
import ast
import numpy as np
import matplotlib.pyplot as plt

IBMQ.load_account()
provider_reale = IBMQ.get_provider(hub = 'partner-cnr', group = 'icar-napoli', project = 'ferrante')
backend_reale = provider_reale.get_backend('ibmq_montreal') # prova a vedere che succede se cambi il backend
job = backend_reale.retrieve_job('62bc0c0b5ef42c2332b292fd')

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
dizionario_finale = dict_r_p_2.copy()
del dizionario_finale['tempo_esecuzione']
risultato_simulatore = {'10111': 1}  # 2500
# t_2 = 8.63475489616394 

for item in list(set(dizionario_finale) - set(risultato_simulatore)):
    risultato_simulatore[item] = 0

risultato_simulatore = dict(sorted(risultato_simulatore.items()))


risultato_transpiler = {'10111': 1}
for item in list(set(dizionario_finale) - set(risultato_transpiler)):
    risultato_transpiler[item] = 0
risultato_transpiler = dict(sorted(risultato_transpiler.items()))

valore_highest = max(dizionario_finale, key=dizionario_finale.get)
dec_valore_highest = int(valore_highest, 2)
x = list(dizionario_finale.keys())
lista_colori_reali = ['blue' for _ in range(len(x))]
lista_colori_reali[dec_valore_highest] = 'red'

fig, axs = plt.subplots(3)
fig.suptitle('Sum 3: 1000 + 1111')
axs[0].bar(list(risultato_simulatore.keys()), list(risultato_simulatore.values()), color = 'blue')
axs[1].bar(list(risultato_transpiler.keys()), list(risultato_transpiler.values()), color = 'blue')
axs[2].bar(list(dizionario_finale.keys()), list(dizionario_finale.values()), color = 'blue')
for val in x:
    # if val == '101':
    axs[0].text(val, risultato_simulatore[val] + 0.05, risultato_simulatore[val], ha='center',fontsize=8)
    axs[1].text(val, risultato_transpiler[val] + 0.05, risultato_transpiler[val], ha='center',fontsize=8)
    axs[2].text(val, dizionario_finale[val] + 0.05, dizionario_finale[val], ha='center',fontsize=8)
    # if val == valore_highest:
    #     axs[0].text(val, dizionario_finale[valore_highest] + 0.05, dizionario_finale[valore_highest], ha='center',fontsize=9)
axs[0].set_ylim([0, 1.2])
axs[1].set_ylim([0, 1.2])
axs[2].set_ylim([0, 1.2])
fig.tight_layout(pad=2.0)
axs[0].title.set_text('Simulator on original circuit')
axs[1].title.set_text('Simulator on transpiled circuit')
axs[2].title.set_text('Real quantum machine transpiled circuit')
plt.show()