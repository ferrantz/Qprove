# scopo: confrontare risultati di qiskit con quelli del metodo classico esaustivo

from qiskit.utils import algorithm_globals
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit import Aer
from qiskit.circuit.library import TwoLocal
from qiskit.utils import QuantumInstance
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit import IBMQ, Aer
import time, sys
from utils.utils_operations import *
from utils.utils_vqe import *
from utils.utils_classic import *
from iperparametri import *

# BACKEND_SELEZIONATO = input('Si scelga il backend del calcolo ' + '\n' + '0: calcolo su macchina locale'+ '\n' + '1: calcolo su simulatore IBM' + '\n' + 
#     '2: calcolo su macchina reale IBM' + '\n' + 'Backend scelto: ')

if BACKEND_SELEZIONATO != '0' and BACKEND_SELEZIONATO != '1' and BACKEND_SELEZIONATO != '2':
    print('Valore immesso non conforme. Script interrotto')
    dataframe_combinazioni_fval = []
    df_classico_esaustivo = []
    sys.exit()

# QISKIT
def main_qiskit():
    print('Inizio risoluzione problema portafoglio con VQE')
    tic_qiskit = time.time()
    algorithm_globals.random_seed = 1234
    qp = crea_problema_quadratico(assets, cov_matrix, Q, BUDGET, lista_prezzi, BOUNDS, num_assets)
    cobyla = COBYLA()
    cobyla.set_options(maxiter=500)
    ry = TwoLocal(num_assets, "ry", "cz", reps=1, entanglement="full")
    print('Numero di qubit del circuito ry: ' + str(ry.num_qubits))
    print('Depth del circuito ry: ' + str(ry.depth()))
    print('Circuito: ' + '\n')
    print(ry.decompose().draw())
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
    vqe_mes = VQE(ry, optimizer=cobyla, quantum_instance=quantum_instance)
    print('vqe_mes.ansatz.num_qubits: ' + str(vqe_mes.ansatz.num_qubits))
    print('vqe_mes.ansatz.depth(): ' + str(vqe_mes.ansatz.depth()))
    print('Ansatz di vqe_mes: ' + '\n')
    print(vqe_mes.ansatz.decompose().draw())
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
    return dataframe_combinazioni_fval

# CLASSICO (ITERAZIONI SU TUTTE LE POSSIBILI COMBINAZIONI)
def main_classico():
    bounds_espanso = espandi_bounds(BOUNDS)
    tic_classico = time.time()
    df_classico_esaustivo = iterazione_classica(bounds_espanso, BUDGET, np.array(lista_prezzi), log_apprezzamenti_quotidiani, cov_matrix, Q)
    toc_classico = time.time() - tic_classico
    print("L'iterazione classica esaustiva è durata " + str(round(float(toc_classico), 3)) + ' secondi')
    df_classico_esaustivo.columns = ['combinazione', 'h']
    df_classico_esaustivo['deviazione_standard'] = varianza_portafoglio(df_classico_esaustivo, cov_matrix)
    df_classico_esaustivo = df_classico_esaustivo.sort_values(by=['h'])
    return df_classico_esaustivo

# PRINTO I RISULTATI
def printa_risultati(dataframe_combinazioni_fval, df_classico_esaustivo):
    print('Dataframe con combinazioni, fval e calcolati dalla VQE con le varianze calcolate dati i pesi e la matrice di covarianza: ' + '\n')
    print(dataframe_combinazioni_fval)
    print('\n' + 'Dataset esaustivo calcolato col metodo classico:' + '\n')
    print(df_classico_esaustivo)

if __name__ == '__main__':
    dataframe, lista_prezzi, log_apprezzamenti_quotidiani, cov_matrix, medie_diversi_er, ann_sd, assets, num_assets = elementi_utili(STRINGA_PERCORSO_CSV, ARGOMENTO_PARSING_CSV, PREZZO_ASSET_1, PREZZO_ASSET_2, PREZZO_ASSET_3, PREZZO_ASSET_4)
    df_vqe = main_qiskit()
    df_classico = main_classico()
    printa_risultati(df_vqe, df_classico)