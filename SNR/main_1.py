from qiskit import *
from matplotlib import pyplot as plt
import math
import pandas as pd

IBMQ.load_account()
provider_reale = IBMQ.get_provider(hub = 'partner-cnr', group = 'icar-napoli', project = 'ferrante')
backend_reale = provider_reale.get_backend('ibmq_montreal') # prova a vedere che succede se cambi il backend
job_2_qubit = backend_reale.retrieve_job('62bc0c0b5ef42c2332b292fd')
job_3_qubit = backend_reale.retrieve_job('62bc0c69d6549771e39783d5')
job_4_qubit = backend_reale.retrieve_job('62bc0ca05ef42c4518b29305')
job_5_qubit = backend_reale.retrieve_job('62bd7518d529f2395f2cb772')
job_6_qubit = backend_reale.retrieve_job('62bd7cd3c9053f9436b47b3e')
job_7_qubit = backend_reale.retrieve_job('62bd763d716a310364562ad2')

dizionario_job_risposta = {job_2_qubit: hex(int('10', 2) + int('11', 2)), job_3_qubit: hex(int('100', 2) + int('111', 2)),
        job_4_qubit: hex(int('1000', 2) + int('1111', 2)), job_5_qubit: hex(int('10000', 2) + int('11111', 2)),
            job_6_qubit: hex(int('100000', 2) + int('111111', 2)), job_7_qubit: hex(int('1000000', 2) + int('1111111', 2))}

lista_snr = []
esponente = 3
contatore_risposta_esatta = 0
# gates_executed = []
for job, risposta in dizionario_job_risposta.items():
    dizionario_risposte = job.result().results[0].data.counts
    shots = sum(dizionario_risposte.values())
    gates_executed = job._data['summary_data_']['summary']['gates_executed']
    if job._name != '1000000+1111111':
        correct_answer = dizionario_risposte[list(dizionario_job_risposta.values())[contatore_risposta_esatta]] / shots
    else: 
        correct_answer = 0
    valore_medio_risultati = 1 / len(list(dizionario_risposte.keys()))
    max_percentuale = max([numero_assoluto / shots for numero_assoluto in list(dizionario_risposte.values())])
    variabile_c = len(dizionario_risposte.keys())
    if job._name != '1000000+1111111':
        variabile_a = (dizionario_risposte[risposta] / shots)**2
    else:
        variabile_a = 0
    dizionari_quadrati_sbagliati = {}
    if job._name != '1000000+1111111': 
        del dizionario_risposte[risposta]
    else:
        pass
    for k, v in dizionario_risposte.items():
        dizionari_quadrati_sbagliati[k] = (v / shots)**2
    variabile_b = sum(dizionari_quadrati_sbagliati.values())
    potenza_segnale = variabile_a / variabile_c
    potenza_rumore = variabile_b / variabile_c
    SNR = potenza_segnale / potenza_rumore
    contatore_risposta_esatta += 1
    df = pd.DataFrame()
    d= {}
    d['gates_executed'] = gates_executed
    d['SNR'] = SNR
    if SNR != 0:
        d['10log10_SNR'] = 10 * math.log(SNR, 10)
    else:
        d['10log10_SNR'] = 'NaN'
    d['%_corrected_sample'] = correct_answer * 100
    d['valore_medio_risultati'] = valore_medio_risultati * 100
    d['possibili_risultati'] = 2**(esponente)
    esponente += 1
    d['max_percentuale'] = max_percentuale * 100
    print(d)
    df = df.append(d, ignore_index=True)



#     dizionari_quadrati_sbagliati = {}
#     for k, v in dizionario_risposte.items():
#         dizionari_quadrati_sbagliati[k] = (v / shots)**2
#     variabile_c = len(dizionario_risposte.keys())
#     if job._name != '1000000+1111111':
#         variabile_a = (dizionario_risposte[risposta] / shots)**2
#     else:
#         variabile_a = 0
#     variabile_b = sum(dizionari_quadrati_sbagliati.values())
#     potenza_segnale = variabile_a / variabile_c
#     potenza_rumore = variabile_b / variabile_c
#     SNR = potenza_segnale / potenza_rumore
#     lista_snr.append(SNR)
#     gates_executed.append(job._data['summary_data_']['summary']['gates_executed'])

# lista_snr = [0.27444287056882644, 0.056240674416523785, 0.00441684647706809, 0.0011949507235243732, 1.554055220762178e-05, 0.0]
# lista_snr_2 = [-0.5615480467655313, -1.2499494800647146, -2.3548876959014247, -2.9226500034545, -4.808548985535104]
# lista_snr_3 = []
# for valore in lista_snr_2:
#     lista_snr_3.append(10 * valore)
# # lista_depth = [12, 17, 22, 27, 32] # , 37
# # lista_depth = [100, 191, 300, 347, 448]
# ax = plt.gca()

# plt.plot(gates_executed[:-1], lista_snr_3)
# ax.set_title('SNR curve')
# plt.xlabel("Depth of (transpiled) circuit")
# plt.ylabel("10log10(SNR)")
# plt.xticks(gates_executed[:-1])
# plt.show()