# scopo: debuggare il file per entrare dentro la libreria docplex e vedere il calcolo che fa in .to_quadratic_program()
# righe estratte da confronto_sim_macc.ipynb

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
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, IBMQ, execute, Aer

from qiskit_optimization.converters import QuadraticProgramToQubo


def index_to_selection(i, num_assets):
    s = "{0:b}".format(i).rjust(num_assets)
    x = np.array([1 if s[i] == "1" else 0 for i in reversed(range(num_assets))])
    return x


def print_result(result):
    selection = result.x
    value = result.fval
    print("Optimal: selection {}, value {:.4f}".format(selection, value))
    eigenstate = result.min_eigen_solver_result.eigenstate 
    #eigenvector = eigenstate if isinstance(eigenstate, np.ndarray) else eigenstate.to_matrix()
    eigenvector = np.array(list(eigenstate.items()))[:, 1].astype('float')
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

# payload=pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
# first_table = payload[0]
# second_table = payload[1]
# df = first_table
# symbols = df['Symbol'].values.tolist()
# print(len(symbols))
# stocks = ['AAPL', 'GOOG', 'INTC', 'WFC'] # ,  'ADCP' 
# test = yf.download(symbols, period='1y')
# test.head()
# test_2 = test[test.columns[1:5]]
# print('capisci i type delle colonne')

PREZZO_ASSET_1 = 50
PREZZO_ASSET_2 = 70
PREZZO_ASSET_3 = 25
PREZZO_ASSET_4 = 30
lista_prezzi = [PREZZO_ASSET_1, PREZZO_ASSET_2, PREZZO_ASSET_3, PREZZO_ASSET_4]

test_2 = pd.read_csv('ottimizzazione_portafoglio/dati_confronto_4_asset.csv', parse_dates = ['Unnamed: 0'])[3:]
test_2 = test_2.rename(columns = {'Unnamed: 0': 'Date', 'Adj Close': 'Asset_1', 'Adj Close.1': 'Asset_2', 'Adj Close.2': 'Asset_3', 'Adj Close.3': 'Asset_4'})
test_2 = test_2.drop(columns = ['Date'])
numero_di_asset_ordinato = 4
#test_2 = test_2['Adj Close']
#test_2['Date'] = pd.to_datetime(test_2['Date']) 
test_2 = test_2.apply(pd.to_numeric)
test = test_2.apply(lambda x: x.fillna(x.mean()),axis=0)
#test = test.sample(n = numero_di_asset_ordinato, axis = 'columns')
log_apprezzamenti_quotidiani = test.pct_change().apply(lambda x: np.log(1+x))
cov_matrix = test.pct_change().apply(lambda x: np.log(1+x)).cov()
corr_matrix = test.pct_change().apply(lambda x: np.log(1+x)).corr()
medie_diversi_er = log_apprezzamenti_quotidiani.mean()
ann_sd = test.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(test.shape[0]))
assets = pd.concat([medie_diversi_er, ann_sd], axis=1) # Creating a table for visualising returns and volatility of assets
assets.columns = ['Returns', 'Volatility']
q = 1
num_assets = 4 #len(test.columns)
budget = 300


# con bounds = [(0, 3), (0, 3), (0, 3), (0, 3)] e budget = 525 -> SUCCESS
# con bounds = [(0, 1), (0, 1), (0, 2), (0, 3)] e budget = 180 -> SUCCESS
# con bounds = [(0, 1), (0, 1), (0, 2), (0, 3)] e budget = 160 -> SUCCESS {con bounds = [(0, 6), (0, 6), (0, 6), (0, 6)] ottengo un altro risultato SUCCESS}
bounds = [(0, 1), (0, 1), (0, 1), (0, 2)] 
portfolio = PortfolioOptimization(
    expected_returns=medie_diversi_er.to_numpy(), covariances=cov_matrix.to_numpy(), risk_factor=q, budget=budget
   , bounds = bounds
)
#qp = portfolio.to_quadratic_program() # ENTRO DENTRO (non ci riesco col debugger)

from docplex.mp.advmodel import AdvModel
from qiskit_optimization.translators import from_docplex_mp
mdl = AdvModel(name="Portfolio optimization")

if bounds is not None:
    x = [
                    mdl.integer_var(lb=bounds[i][0], ub=bounds[i][1], name=f"x_{i}")
                    for i in range(num_assets)
                ]
else:
    x = [mdl.binary_var(name=f"x_{i}") for i in range(num_assets)] 
quad = mdl.quad_matrix_sum(cov_matrix, x) # performa il calcolo x’Qx (per la seconda parte della funzione oggetto)
linear = np.dot(assets['Returns'], x) # per la prima parte della funzione oggetto. 
mdl.minimize(q * quad - linear) # vincolo: minimizza questa roba. Q AGISCE SULLE COVARIANZE E SI CAMBIA IL SEGNO DEI RITORNI
#mdl.add_constraint(mdl.sum(x[i] for i in range(num_assets)) == budget) # vincolo: la somma dei titoli da acquistare deve essere pari al budget
mdl.add_constraint(mdl.sum(x[i]*lista_prezzi[i] for i in range(num_assets)) <= budget)
#mdl.add_constraint(mdl.sum(x[i]*lista_prezzi[i] for i in range(num_assets)) >= 100)
qp = from_docplex_mp(mdl) # traduce "in termini di qiskit" un mp problem in un problema quadratico
print(qp)

algorithm_globals.random_seed = 1234
backend = Aer.get_backend("qasm_simulator") # statevector_simulator con 16 qubit è troppo dispendioso
cobyla = COBYLA()
cobyla.set_options(maxiter=500)
ry = TwoLocal(num_assets, "ry", "cz", reps=1, entanglement="full")
print('Numero di qubit del circuito ry: ' + str(ry.num_qubits))
print('Depth del circuito ry: ' + str(ry.depth()))
print('Circuito: ' + '\n')
print(ry.decompose().draw())
# -----------------------------------------
#IBMQ.load_account()
# provider_reale = IBMQ.get_provider(hub = 'ibm-q')
# BACKEND = 'ibmq_qasm_simulator'
# provider_reale = IBMQ.get_provider(hub = 'partner-cnr', group = 'simulators', project = 'simulators')
#provider_reale = IBMQ.get_provider(hub = 'ibm-q', group = 'open', project = 'main')
# provider_reale = IBMQ.get_provider(hub = 'partner-cnr', group = 'icar-napoli', project = 'ferrante')
#BACKEND = 'ibmq_quito'
#backend = provider_reale.get_backend(BACKEND) 
# -----------------------------------------
quantum_instance = QuantumInstance(backend=backend, seed_transpiler=6954)
vqe_mes = VQE(ry, optimizer=cobyla, quantum_instance=quantum_instance)
print('vqe_mes.ansatz.num_qubits: ' + str(vqe_mes.ansatz.num_qubits))
print('vqe_mes.ansatz.depth(): ' + str(vqe_mes.ansatz.depth()))
print('Ansatz di vqe_mes: ' + '\n')
print(vqe_mes.ansatz.decompose().draw())


vqe = MinimumEigenOptimizer(vqe_mes) # trasforma il problema in QUBO. Il QUBO diviene Hamiltoniana il cui autovettore minimo e 
# corrispondente eigenstate corrispondono alla soluzione ottimale del problema di ottimizzazione originale 



result = vqe.solve(qp)
print(result)



# BACKEND = 'ibmq_montreal' 
# provider_reale = IBMQ.get_provider(hub = 'partner-cnr', group = 'icar-napoli', project = 'ferrante')
# # provider_reale = IBMQ.get_provider(hub = 'ibm-q', group = 'open', project = 'main')
# # provider_reale = IBMQ.get_provider(hub = 'partner-cnr', group = 'simulators', project = 'simulators') # per simulatori
# backend_reale = provider_reale.get_backend(BACKEND) 
# job_reale = execute(circuit, backend = backend_reale, shots=num_shots)




print(assets)
print_result(result)