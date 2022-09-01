# ref: https://www.machinelearningplus.com/machine-learning/portfolio-optimization-python-example/

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from qiskit.utils import algorithm_globals
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit import Aer
from qiskit.circuit.library import TwoLocal
from qiskit.utils import QuantumInstance
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_optimization.converters import QuadraticProgramToQubo
import time 
from qiskit import IBMQ, execute, transpile, assemble

NOME_FILE = 'otto_asset'
NUMERO_DI_ASSET_ORDINATO = 8
BACKEND = 'ibmq_montreal'

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
    for i in i_sorted:
        x = index_to_selection(i, num_assets)
        value = QuadraticProgramToQubo().convert(qp).objective.evaluate(x)
        # value = portfolio.to_quadratic_program().objective.evaluate(x)
        probability = probabilities[i]
        print("%10s\t%.4f\t\t%.4f" % (x, value, probability))

def remove_exponent(value):
    decial = value.split('e')
    ret_val = format(((float(decial[0]))*(10**int(decial[1]))), '.8f')
    return ret_val

payload=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
first_table = payload[0]
second_table = payload[1]
df = first_table
symbols = df['Symbol'].values.tolist()
print(len(symbols))
test = yf.download(symbols, period='1y')
test.head()
test = test['Adj Close'] # prendo solo i prezzi di chiusura aggiustati
test = test.drop(columns = ['BF.B', 'BRK.B'])
test = test.apply(lambda x: x.fillna(x.mean()),axis=0)
test = test.sample(n = NUMERO_DI_ASSET_ORDINATO, axis = 'columns')
log_apprezzamenti_quotidiani = test.pct_change().apply(lambda x: np.log(1+x))
cov_matrix = test.pct_change().apply(lambda x: np.log(1+x)).cov()
corr_matrix = test.pct_change().apply(lambda x: np.log(1+x)).corr()
medie_diversi_er = log_apprezzamenti_quotidiani.mean()
ann_sd = test.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(test.shape[0]))
assets = pd.concat([medie_diversi_er, ann_sd], axis=1) # Creating a table for visualising returns and volatility of assets
assets.columns = ['Returns', 'Volatility']
# assets.plot.scatter(x='Volatility', y='Returns', marker='o', s=10, alpha=0.3, grid=True, figsize=[10,10])

# SIMULATORE

q = 0.5
num_assets = len(test.columns)
budget = num_assets // 2 
portfolio = PortfolioOptimization(
    expected_returns=medie_diversi_er.to_numpy(), covariances=cov_matrix.to_numpy(), risk_factor=q, budget=budget
)
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
print_result(result)

dict_r_p = {}
for i in range(len(result.raw_samples)):
    r = str(result.raw_samples[i]).split('SolutionSample(x=array(')[1].split(')')[0] # risposta
    p = str(result.raw_samples[i]).split('probability')[1].replace('=', '').split(',')[0] # probabilità associata alla risposta
    if 'e' in p: # check per vedere se è un numero esponenziale
        p_pulito = remove_exponent(p)
        dict_r_p[r] = p_pulito
    else:
        dict_r_p[r] = p

for old_key in dict_r_p.keys():
    new_key = old_key.replace('[', '').replace(']', '').replace(' ', '').replace(',', '')
    dict_r_p[new_key] = dict_r_p.pop(old_key)

for old_value in dict_r_p.values():
    new_value = float(old_value)
    for keys in dict_r_p:
        dict_r_p[keys] = float(dict_r_p[keys])

for n in range(2**test.shape[1]): # 2^numero di variabili (asset)
    binary = bin(n).replace('0b', '')
    if len(binary) < test.shape[1]:
        zeri_da_aggiungere = test.shape[1] - len(binary)
        binary = zeri_da_aggiungere * '0' + binary
    if binary not in dict_r_p.keys():
        dict_r_p[binary] = 0.0
dict_r_p = dict(sorted(dict_r_p.items()))

with open(NOME_FILE, "w") as file:
    file.write(str(dict_r_p))
    file.write(str(t_2))

# MACCHINA REALE 

IBMQ.load_account()
# provider_reale = IBMQ.get_provider(hub = 'ibm-q')
provider_reale = IBMQ.get_provider(hub = 'partner-cnr', group = 'icar-napoli', project = 'ferrante')
backend_reale = provider_reale.get_backend(BACKEND) 
quantum_instance_reale = QuantumInstance(backend=backend_reale, seed_transpiler=6954) # leva seed_simulator=1587
vqe_mes_reale = VQE(ry, optimizer=cobyla, quantum_instance=quantum_instance_reale)
vqe_reale = MinimumEigenOptimizer(vqe_mes_reale)
print('Mando in coda')
result = vqe_reale.solve(qp)