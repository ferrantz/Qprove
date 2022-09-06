# TO DO: 
# Aggiungi colonna h ed fval.
# ----------------------
# prova a minimizzare con funzione COBYLA di scipy dando VQE in input


# scopo: confrontare risultati di qiskit con quelli del metodo classico esaustivo

from qiskit.utils import algorithm_globals
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import COBYLA, SPSA, P_BFGS
from qiskit import Aer
from qiskit.circuit.library import TwoLocal
from qiskit.utils import QuantumInstance
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit import IBMQ, Aer
import time, sys
from utils.utils_operations import *
from utils.utils_vqe import *
from utils.utils_classic import *
from utils.utils_tabella import *
from iperparametri import *

# BACKEND_SELEZIONATO = input('Si scelga il backend del calcolo ' + '\n' + '0: calcolo su macchina locale'+ '\n' + '1: calcolo su simulatore IBM' + '\n' + 
#     '2: calcolo su macchina reale IBM' + '\n' + 'Backend scelto: ')

if BACKEND_SELEZIONATO != '0' and BACKEND_SELEZIONATO != '1' and BACKEND_SELEZIONATO != '2':
    print('Valore immesso non conforme. Script interrotto')
    dataframe_combinazioni_fval = []
    df_classico_esaustivo = []
    sys.exit()

# QISKIT
def main_qiskit(bounds_):
    print('Inizio risoluzione problema portafoglio con VQE')
    tic_qiskit = time.time()
    algorithm_globals.random_seed = 1234
    qp = crea_problema_quadratico(assets, cov_matrix, Q, BUDGET, lista_prezzi, bounds_, num_assets)
    if OTTIMIZZATORE == 'cobyla':
        optimizer = COBYLA()
        optimizer.set_options(maxiter=500)
    elif OTTIMIZZATORE == 'spsa':
        optimizer = SPSA()
        optimizer.set_options(maxiter=500)
    elif OTTIMIZZATORE == 'p_bfgs':
        optimizer = P_BFGS()
        optimizer.set_options(maxiter=500)
    ry = TwoLocal(num_assets, "ry", "cz", reps=1, entanglement="full")
    # print('Numero di qubit del circuito ry: ' + str(ry.num_qubits))
    # print('Depth del circuito ry: ' + str(ry.depth()))
    # print('Circuito: ' + '\n')
    # print(ry.decompose().draw())
    if BACKEND_SELEZIONATO == '0':        
        backend = Aer.get_backend(BACKEND_LOCALE) # statevector_simulator con 16 qubit è troppo dispendioso
    elif BACKEND_SELEZIONATO == '1':
        IBMQ.load_account()
        provider_reale = IBMQ.get_provider(hub = 'ibm-q')
        backend = provider_reale.get_backend(BACKEND_SIMULATORE_IBM) 
    elif BACKEND_SELEZIONATO == '2':
        provider_reale = IBMQ.get_provider(hub = 'partner-cnr', group = 'icar-napoli', project = 'ferrante')
        backend = provider_reale.get_backend(BACKEND_MACCHINA_REALE_IBM) 
    quantum_instance = QuantumInstance(backend=backend, seed_transpiler=6954)
    vqe_mes = VQE(ry, optimizer=optimizer, quantum_instance=quantum_instance)
    # print('vqe_mes.ansatz.num_qubits: ' + str(vqe_mes.ansatz.num_qubits))
    # print('vqe_mes.ansatz.depth(): ' + str(vqe_mes.ansatz.depth()))
    # print('Ansatz di vqe_mes: ' + '\n')
    # print(vqe_mes.ansatz.decompose().draw())
    vqe = MinimumEigenOptimizer(vqe_mes) # trasforma il problema in QUBO. Il QUBO diviene Hamiltoniana il cui autovettore minimo e corrispondente eigenstate corrispondono alla soluzione ottimale del problema di ottimizzazione originale 
    result = vqe.solve(qp)
    print(result)
    combinazioni, fval = liste_combinazioni_e_fval(result)
    toc_qiskit = time.time() - tic_qiskit
    print('Il calcolo delle migliori combinazioni è durato ' + str(round(float(toc_qiskit), 3)) + ' secondi') 
    dataframe_combinazioni_fval = pd.DataFrame(list(zip(combinazioni, fval)))
    if dataframe_combinazioni_fval.empty:
        print('Non sono state trovate combinazioni valide (status = SUCCESS)')
    else:
        dataframe_combinazioni_fval.columns = ['combinazione', 'fval']
        dataframe_combinazioni_fval['deviazione_standard'] = varianza_portafoglio(dataframe_combinazioni_fval, cov_matrix)
    n_comb_vqe = conta_combinazioni_quantistiche(result) # numero di combinazioni calcolate dalla vqe
    if fval == []:
        best_fval = None
    else:
        best_fval = fval[0]
    return dataframe_combinazioni_fval, n_comb_vqe, ry.num_qubits, ry.depth(), toc_qiskit, best_fval

# CLASSICO (ITERAZIONI SU TUTTE LE POSSIBILI COMBINAZIONI)
def main_classico(bounds_):
    bounds_espanso = espandi_bounds(bounds_)
    tic_classico = time.time()
    df_classico_esaustivo, h_migliore_classico = iterazione_classica(bounds_espanso, BUDGET, np.array(lista_prezzi), log_apprezzamenti_quotidiani, cov_matrix, Q)
    toc_classico = time.time() - tic_classico
    print("L'iterazione classica esaustiva è durata " + str(round(float(toc_classico), 3)) + ' secondi')
    df_classico_esaustivo.columns = ['combinazione', 'h']
    df_classico_esaustivo['deviazione_standard'] = varianza_portafoglio(df_classico_esaustivo, cov_matrix)
    df_classico_esaustivo = df_classico_esaustivo.sort_values(by=['h'])
    n_comb_classico = conta_combinazioni_classiche(bounds_espanso)
    return df_classico_esaustivo, n_comb_classico, toc_classico

# PRINTO I RISULTATI
def risultati(dataframe_combinazioni_fval, df_classico_esaustivo, logaritmi_degli_apprezzamenti_giornalieri, matrice_covarianze, propensione_al_rischio):
    print('Dataframe con combinazioni, fval e calcolati dalla VQE con le varianze calcolate dati i pesi e la matrice di covarianza: ' + '\n')
    print(dataframe_combinazioni_fval)
    print('\n' + 'Dataset esaustivo calcolato col metodo classico:' + '\n')
    print(df_classico_esaustivo)
    # calcola h della migliore combinazione calcolata da VQE (H_VQE - H_CLASSICO)
    w = dataframe_combinazioni_fval.iloc[0].combinazione # combinazione migliore calcolata dalla VQE
    h_classico = df_classico_esaustivo.iloc[0].h
    h_VQE = -(np.matmul(w, logaritmi_degli_apprezzamenti_giornalieri.iloc[-1, :].T)) + np.matmul(np.matmul(w.T, matrice_covarianze), w) * propensione_al_rischio ####q*w.T*cov_matrix*w
    return h_VQE - h_classico, h_VQE, h_classico

def crea_tabella(n_vqe, n_classico, limiti, n_qubit, depth, diff): 

    '''Crea la tabella chiesta da Francesco il 05/09''' # cambia il docstring

    return n_vqe, n_classico, limiti, n_qubit, depth, diff

lista_ottimizzatore_in_uso = ['cobyla', 'cobyla', 'cobyla', 'cobyla', 'cobyla', 'cobyla', 'spsa', 'spsa', 'spsa', 'spsa', 'spsa', 'spsa', 'p_bfgs', 'p_bfgs', 'p_bfgs', 'p_bfgs', 'p_bfgs', 'p_bfgs']
lista_numero_asset_in_uso = []
lista_differenze_h = []
lista_combinazioni_vqe = []
lista_numero_qubit = []
lista_depth_circuito = []
lista_combinazione_metodo_classico = []
lista_tempo_vqe = []
lista_tempo_classico = []
lista_bounds = []
lista_coi_prezzi = []
lista_migliori_fval = []
lista_migliori_h_classici = []
lista_migliori_h_quantistici = []

for ott in LISTA_OTTIMIZZATORI:
    OTTIMIZZATORE = ott
    lista_ottimizzatore_in_uso.append(OTTIMIZZATORE)
    print('\n')
    print("OTTIMIZZATORE: " + str(ott))
    for NUMERO_ASSET in range(2, 8): 
        lista_numero_asset_in_uso.append(NUMERO_ASSET)
        print('\n')
        print('----------------------------------------------------------------------------------------------')
        print('Sto considerando ' + str(NUMERO_ASSET) + ' asset')
        dataframe, lista_prezzi, log_apprezzamenti_quotidiani, cov_matrix, medie_diversi_er, ann_sd, assets, num_assets = elementi_utili(STRINGA_PERCORSO_CSV, NUMERO_ASSET, PREZZO_ASSET_1, PREZZO_ASSET_2, PREZZO_ASSET_3, PREZZO_ASSET_4, PREZZO_ASSET_5, PREZZO_ASSET_6, PREZZO_ASSET_7)
        vincoli_acquisto = slicing_bounds(BOUNDS, NUMERO_ASSET)
        df_vqe, numero_combinazioni_vqe, numero_qubit, depth_circuito, tempo_qiskit, fval_migliore_combinazione = main_qiskit(vincoli_acquisto)
        df_classico, numero_combinazioni_classico, tempo_classico = main_classico(vincoli_acquisto)
        try:
            differenza_h, h_migliore_VQE, h_migliore_combinazione_classica = risultati(df_vqe, df_classico, log_apprezzamenti_quotidiani, cov_matrix, Q)
        except: # l'algoritmo quantistico potrebbe anche non trovare nulla
            print("La VQE non è stata in grado di trovare una soluzione")
            differenza_h, h_migliore_VQE, h_migliore_combinazione_classica = None, None, None
        lista_differenze_h.append(differenza_h)
        lista_combinazioni_vqe.append(numero_combinazioni_vqe)
        lista_numero_qubit.append(numero_qubit)
        lista_depth_circuito.append(depth_circuito)
        lista_combinazione_metodo_classico.append(numero_combinazioni_classico)
        lista_tempo_vqe.append(tempo_qiskit)
        lista_tempo_classico.append(tempo_classico)
        bounds_in_uso = BOUNDS[:NUMERO_ASSET]
        lista_bounds.append(bounds_in_uso)
        lista_coi_prezzi.append(lista_prezzi)
        lista_migliori_fval.append(fval_migliore_combinazione)
        lista_migliori_h_classici.append(h_migliore_combinazione_classica)
        lista_migliori_h_quantistici.append(h_migliore_VQE)
        # crea_tabella(numero_combinazioni_vqe, numero_combinazioni_classico, BOUNDS, numero_qubit, depth_circuito, differenza_h)

dataframe = pd.DataFrame(
    {'ottimizzatore':lista_ottimizzatore_in_uso,
    'numero_asset': lista_numero_asset_in_uso,
    'bounds_usati (in ordine)': lista_bounds,
    'prezzi_usati (in ordine)': lista_coi_prezzi,
    'n_combinazioni_VQE': lista_combinazioni_vqe,
    'n_combinazioni_metodo_classico': lista_combinazione_metodo_classico,
    'n_qubit': lista_numero_qubit,
    'depth_circuito': lista_depth_circuito,
    'durata_VQE (in secondi)': lista_tempo_vqe,
    'durata_metodo_classico (in secondi)': lista_tempo_classico,
    'fval_migliore_combinazione': lista_migliori_fval,
    'h_migliore_combinazione_quantistica': lista_migliori_h_quantistici,
    'h_migliore_combinazione_classica': lista_migliori_h_classici,
    'delta_h': lista_differenze_h}
    )

values = {'delta_h': '/'} # dove c'è NaN nella colonna delta_h sostituisco con '/'
dataframe = dataframe.fillna(value = values)
print(dataframe)
dataframe.to_excel('ottimizzazione_portafoglio/confronto_qiskit_metodo_classico/dataframe_ciclo.xlsx')
# SINGLE SHOT
# if __name__ == '__main__':
#     dataframe, lista_prezzi, log_apprezzamenti_quotidiani, cov_matrix, medie_diversi_er, ann_sd, assets, num_assets = elementi_utili(STRINGA_PERCORSO_CSV, NUMERO_ASSET, PREZZO_ASSET_1, PREZZO_ASSET_2, PREZZO_ASSET_3, PREZZO_ASSET_4, PREZZO_ASSET_5, PREZZO_ASSET_6, PREZZO_ASSET_7)
#     vincoli_acquisto = slicing_bounds(BOUNDS, NUMERO_ASSET)
#     df_vqe, numero_combinazioni_vqe, numero_qubit, depth_circuito = main_qiskit(vincoli_acquisto)
#     df_classico, numero_combinazioni_classico = main_classico(vincoli_acquisto)
#     differenza_h = risultati(df_vqe, df_classico)
#     crea_tabella(numero_combinazioni_vqe, numero_combinazioni_classico, BOUNDS, numero_qubit, depth_circuito, differenza_h)