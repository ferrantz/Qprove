# ref: https://www.machinelearningplus.com/machine-learning/portfolio-optimization-python-example/

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import time
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit import Aer
from qiskit.circuit.library import TwoLocal
from qiskit.utils import QuantumInstance
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.utils import algorithm_globals
from qiskit import IBMQ
import ast 
import os


BACKEND = 'ibmq_montreal'

def selection_to_picks(num_assets, selection):
    purchase = []
    for i in range(num_assets):
        if selection[i] == 1:
            purchase.append(test.columns[i])
    return purchase

def index_to_selection(i, num_assets):
    s = "{0:b}".format(i).rjust(num_assets)
    x = np.array([1 if s[i] == "1" else 0 for i in reversed(range(num_assets))])
    return x

def print_result(result):
    selection = result.x
    value = result.fval
    print("Optimal: selection {}, value {:.4f}".format(selection, value))

    eigenstate = result.min_eigen_solver_result.eigenstate
    eigenvector = eigenstate if isinstance(eigenstate, np.ndarray) else eigenstate.to_matrix()
    probabilities = np.abs(eigenvector) ** 2
    i_sorted = reversed(np.argsort(probabilities))
    print("\n----------------- Full result ---------------------")
    print("selection\tvalue\t\tprobability")
    print("---------------------------------------------------")
    contatore_giri = 0
    for i in i_sorted:
        x = index_to_selection(i, num_assets)
        if contatore_giri == 0:
            da_ritornare = x
        value = QuadraticProgramToQubo().convert(qp).objective.evaluate(x)
        # value = portfolio.to_quadratic_program().objective.evaluate(x)
        probability = probabilities[i]
        print("%10s\t%.4f\t\t%.4f" % (x, value, probability))
        contatore_giri += 1
    return da_ritornare
def remove_exponent(value):
    decial = value.split('e')
    ret_val = format(((float(decial[0]))*(10**int(decial[1]))), '.8f')
    return ret_val

NUMERO_ASSET = 8
# payload=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
# first_table = payload[0]
# second_table = payload[1]
# df = first_table
# symbols = df['Symbol'].values.tolist()
# print(len(symbols))
# test = yf.download(symbols, period='1y')
# test.head()
# test = test['Adj Close'] # prendo solo i prezzi di chiusura aggiustati
# test = test.drop(columns = ['BF.B', 'BRK.B'])
# test = test.apply(lambda x: x.fillna(x.mean()),axis=0)
# test = test.sample(n = NUMERO_ASSET, axis = 'columns')

test = pd.read_csv(r"C:\\Users\\italo\\Desktop\\Qprove\\dataframe_per_confronto_8_asset.csv", index_col='Date')


log_apprezzamenti_quotidiani = test.pct_change().apply(lambda x: np.log(1+x))
cov_matrix = test.pct_change().apply(lambda x: np.log(1+x)).cov()
corr_matrix = test.pct_change().apply(lambda x: np.log(1+x)).corr()
medie_diversi_er = log_apprezzamenti_quotidiani.mean()
ann_sd = test.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(test.shape[0]))
assets = pd.concat([medie_diversi_er, ann_sd], axis=1) # Creating a table for visualising returns and volatility of assets
assets.columns = ['Returns', 'Volatility']
p_ret = [] # array dei ritorni del portfolio
p_vol = [] # array della volatilità del portfolio
p_weights = [] # array dei pesi degli asset
num_assets = len(test.columns)

if NUMERO_ASSET == 3:
    w = [[x, y, z] for x in [0,1] for y in [0,1] for z in [0,1] if [x, y, z] != [0, 0, 0]] # elimino la combinazione [0, 0, 0] perché è quella naturalmente a varianza minima

if NUMERO_ASSET == 4:
    w = [[x, y, z, a] for x in [0,1] for y in [0,1] for z in [0,1] for a  in [0,1] if [x, y, z, a] != [0, 0, 0, 0]] # elimino la combinazione [0, 0, 0] perché è quella naturalmente a varianza minima

if NUMERO_ASSET == 8:
    w = [[x, y, z, a, b, c, d, e] for x in [0,1] for y in [0,1] for z in [0,1] for a in [0,1] 
        for b in [0,1] for c in [0,1] for d in [0,1] for e in [0,1]
            if [x, y, z, a] != [0, 0, 0, 0]]

for portfolio in range(len(w)):
    t = time.time()
    weights = w[portfolio]
    weights = weights/np.sum(weights)
    returns = np.dot(weights, medie_diversi_er)
    p_ret.append(returns)
    p_weights.append(weights)
    var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
    sd = np.sqrt(var) # Daily standard deviation
    ann_sd = sd*np.sqrt(250) # Annual standard deviation = volatility
    p_vol.append(ann_sd)
    # print('Tempo per un singolo portfolio: ' + str(time.time() - t))  

data = {'Returns':p_ret, 'Volatility':p_vol}
for counter, symbol in enumerate(test.columns.tolist()):
    # print(counter, symbol)
    data[symbol +' weight'] = [w[counter] for w in p_weights]
portfolios  = pd.DataFrame(data)
min_vol_port = portfolios.iloc[portfolios['Volatility'].idxmin()]
print('portfolio a varianza minima: ')                             
# print(min_vol_port)
array = np.array(min_vol_port.iloc[2:])
print(list(np.where(array > 0, 1, array)))
# e plottiamolo
# plt.subplots(figsize=[10,10])
# plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=10, alpha=0.3)
# plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='*', s=500)
# plt.show()

# SIMULATORE

q = 0
num_assets = NUMERO_ASSET
budget = (min_vol_port.iloc[2:] > 0).value_counts()[True] # per fare confronti seleziono il budget uguale al numero di asset che acquisterei col metodo classico a varianza minima aggiustato 
portfolio = PortfolioOptimization(
    expected_returns=medie_diversi_er.to_numpy(), covariances=cov_matrix.to_numpy(), budget=budget, risk_factor=q)
qp = portfolio.to_quadratic_program()
print(qp)
algorithm_globals.random_seed = 1234
backend = Aer.get_backend("statevector_simulator")
cobyla = COBYLA()
cobyla.set_options(maxiter=500)
ry = TwoLocal(num_assets, "ry", "cz", reps=3, entanglement="full")
quantum_instance = QuantumInstance(backend=backend, seed_simulator=1587, seed_transpiler=6954)
vqe_mes = VQE(ry, optimizer=cobyla, quantum_instance=quantum_instance)
vqe = MinimumEigenOptimizer(vqe_mes)
t = time.time()
result = vqe.solve(qp)
t_2 = time.time() - t
print('Il simulatore per le ' + str(assets.shape[0]) + ' variabili ha impiegato ' + str(t_2) + ' secondi')
# print_result(result)
migliore_combinazione = print_result(result)
dataframe_intermedio = pd.concat([portfolios.iloc[:, :2], np.sign(portfolios.iloc[:, 2:])], axis=1)
# indice_simulatore = portfolios.loc[(dataframe_intermedio.iloc[:, 2] == migliore_combinazione[0]) & (dataframe_intermedio.iloc[:, 3] == migliore_combinazione[1]) & (dataframe_intermedio.iloc[:, 4] == migliore_combinazione[2])].index.to_list()
indice_simulatore = int(np.array2string(migliore_combinazione).replace('[', '').replace(' ', '').replace(']', ''), 2) - 1 # -1 perché ho tolto la combinazione con tutti 0


# MACCHINA REALE
# IBMQ.load_account()
# provider_reale = IBMQ.get_provider(hub = 'ibm-q')
# provider_reale = IBMQ.get_provider(hub = 'partner-cnr', group = 'icar-napoli', project = 'ferrante')
# backend_reale = provider_reale.get_backend(BACKEND) 
# quantum_instance_reale = QuantumInstance(backend=backend_reale, seed_transpiler=1414) # leva seed_simulator=1587
# vqe_mes_reale = VQE(ry, optimizer=cobyla, quantum_instance=quantum_instance_reale)
# vqe_reale = MinimumEigenOptimizer(vqe_mes_reale)
# print('budget = ' + str(budget))
# print('q = ' + str(q))
# print('numero di asset = ' + str(num_assets))
# print('Mando in coda')
# result = vqe_reale.solve(qp)


# # -----------------------------------------------------------------------------------------------------------------
# IBMQ.load_account()
# provider_reale = IBMQ.get_provider(hub = 'partner-cnr', group = 'icar-napoli', project = 'ferrante')
# backend_reale = provider_reale.get_backend('ibmq_montreal') # prova a vedere che succede se cambi il backend

# id con 3 asset: 62b9ce1ad4d0c21a711df6c0
# id con 4 asset: 
# id con 8 asset: 62bb0fe0e1377d48fe0987e4

# job = backend_reale.retrieve_job('')
# risultati_intermedi = str(job.result()).split('counts=')[1].split('),')[0]
# n_qubit = int(np.log2(risultati_intermedi.count(':')))
# lista_statevector = []
# print('n_qubit: ' + str(n_qubit))
# for numero_decimale in range(2**(n_qubit)):
#     numero_binario = bin(numero_decimale).replace("0b", "")
#     if len(numero_binario) < n_qubit:
#         zeri_da_appendere = n_qubit - len(numero_binario)
#         numero_binario = zeri_da_appendere * '0' + numero_binario
#     lista_statevector.append(numero_binario)

# dict_r_p_2 = {}
# risultati_intermedi_2 = ast.literal_eval(risultati_intermedi)
# for y in range(len(lista_statevector)):
#     dict_r_p_2[lista_statevector[y]] = list(risultati_intermedi_2.items())[y][1] # [1] perché è il valore

# shots = int(str(job.result()).split('shots=')[1].split(',')[0])
# lista_valori_divisi = []
# for v in dict_r_p_2.values():
#     rapporto = v / shots
#     lista_valori_divisi.append(rapporto)

# for key, val in zip(dict_r_p_2, lista_valori_divisi): # assegna il rapporto a 1000 dei valori
#     dict_r_p_2[key] = val

# dict_r_p_2['tempo_esecuzione'] = (job.time_per_step()['COMPLETED'] - job.time_per_step()['RUNNING']).total_seconds()

# dizionario_subset_visto_budget = {key: value for key, value in dict_r_p_2.items() if key.count('1') == budget}
# migliore_combinazione_reale = max(dizionario_subset_visto_budget, key = dizionario_subset_visto_budget.get)
# dataframe_intermedio = pd.concat([portfolios.iloc[:, :2], np.sign(portfolios.iloc[:, 2:])], axis=1)
# indice_reale = int(migliore_combinazione_reale, 2) - 1
# plt.subplots(figsize=[10,10])
# plt.scatter(portfolios['Volatility'], portfolios['Returns'],marker='o', s=100, alpha=0.3, color = 'black')
# plt.scatter(min_vol_port[1], min_vol_port[0], color='r', marker='d', s=200, label = 'Rivisitazione Markovitz pesi prestabili')
# plt.scatter(portfolios.iloc[indice_simulatore].iloc[1], portfolios.iloc[indice_simulatore].iloc[0], color='orange', marker='x', s=200, label = 'Simulatore')
# plt.scatter(portfolios.iloc[indice_reale].iloc[1], portfolios.iloc[indice_reale].iloc[0], color='blue', marker='*', s=200, label = 'Macchina reale')
# plt.legend()
# titolo_figura = 'rischio_' + str(q).replace(',', '') + '_' + 'budget_' + str(budget) + '.png'
# plt.title('q = ' + str(q) + ', Budget = ' + str(budget) + ', numero di asset: ' + str(NUMERO_ASSET))
# plt.tight_layout()
# plt.savefig(os.path.join(r"C:\\Users\\italo\\Desktop\\Qprove\\ottimizzazione_portafoglio\\figure_markowitz", titolo_figura))
# plt.show()