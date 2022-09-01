import warnings
warnings.filterwarnings("ignore")
from qiskit import IBMQ
import yfinance as yf
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
from qiskit_finance.applications.optimization import PortfolioOptimization
# from utils import*
from qiskit.algorithms import VQE, QAOA, NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.utils import algorithm_globals
from qiskit import Aer
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import TwoLocal
from qiskit.utils import QuantumInstance

def index_to_selection(i, num_assets):
    s = "{0:b}".format(i).rjust(num_assets)
    x = np.array([1 if s[i] == "1" else 0 for i in reversed(range(num_assets))])
    return x


def print_result(result):
    selection = result.x
    value = result.fval
    print("Optimal: selection {}, value {:.4f}".format(selection, value))

    eigenstate = result.min_eigen_solver_result.eigenstate # eigenstate Ψ(θ) tale per cui λmin ≤ ≡ λ(θ) <Ψ(θ)|H|Ψ(θ)>
    eigenvector = eigenstate if isinstance(eigenstate, np.ndarray) else eigenstate.to_matrix() # trasforma in numpy array
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

stocks = ['AAPL', 'GOOG', 'INTC', 'WFC']
# stocks = ['AAPL', 'GOOG', 'INTC']
data = yf.download(stocks, period='1y')

# data = data.iloc[:, data.columns.get_level_values(0)=='Adj Close']
print('La shape del dataset è: ' + str(data.shape))
# per ottenere deviazione standard, medie e covarianza uso le greche
closes = np.transpose(np.array(data.Close)) # matrix of daily closing prices
absdiff = np.diff(closes)                   # change in closing price each day
reldiff = np.divide(absdiff, closes[:,:-1]) # relative change in daily closing price
delta = np.mean(reldiff, axis=1)            # mean price change
print('delta: ' + str(delta))
sigma = np.cov(reldiff)                     # covariance (standard deviations)
std = np.std(reldiff, axis=1)               # standard deviation

print('METODO CLASSICO')
m = gp.Model('portfolio') # scopro che è Continuous instance portfolio chiamando m.getVars

# Add matrix variable for the stocks
x = m.addMVar(len(stocks))

# Objective is to minimize risk (squared).  This is modeled using the
# covariance matrix, which measures the historical correlation between stocks
portfolio_risk = x @ sigma @ x # covarianze per ogni coppia
m.setObjective(portfolio_risk, GRB.MINIMIZE)

# Fix budget with a constraint
m.addConstr(x.sum() == 1, 'budget')

# Verify model formulation
m.write('portfolio_selection_optimization.lp')

# Optimize model to find the minimum risk portfolio
m.optimize()
minrisk_volatility = sqrt(m.ObjVal)
minrisk_return = delta @ x.X # equivalente di np.dot
print(pd.DataFrame(data=np.append(x.X, [minrisk_volatility, minrisk_return]),
             index=stocks + ['Volatility', 'Expected Return'],
             columns=['Minimum Risk Portfolio']))
portfolio_return = delta @ x
target = m.addConstr(portfolio_return == minrisk_return, 'target')

# Solve for efficient frontier by varying target return
frontier = np.empty((2,0))
for r in np.linspace(delta.min(), delta.max(), 25):
    target[0].rhs = r
    m.optimize()
    frontier = np.append(frontier, [[sqrt(m.ObjVal)],[r]], axis=1)
fig, ax = plt.subplots(figsize=(10,8))

# Plot volatility versus expected return for individual stocks
ax.scatter(x=std, y=delta,
           color='Blue', label='Individual Stocks')
for i, stock in enumerate(stocks):
    ax.annotate(stock, (std[i], delta[i]))

# Plot volatility versus expected return for minimum risk portfolio
ax.scatter(x=minrisk_volatility, y=minrisk_return, color='DarkGreen')
ax.annotate('Minimum\nRisk\nPortfolio', (minrisk_volatility, minrisk_return),
            horizontalalignment='right')

# Plot efficient frontier
ax.plot(frontier[0], frontier[1], label='Efficient Frontier', color='DarkGreen')

# Format and display the final plot
ax.axis([frontier[0].min()*0.7, frontier[0].max()*1.3, delta.min()*1.2, delta.max()*1.2])
ax.set_xlabel('Volatility (standard deviation)')
ax.set_ylabel('Expected Return')
ax.legend()
ax.grid()
plt.show()

print('METODO QUANTISTICO')
num_assets = len(delta)
seed = 123
mu = delta.copy()
# plt.imshow(sigma, interpolation="nearest")
# plt.show()
q = 0.005  # set risk factor
# budget = num_assets // 2  # set budget
budget = 1
portfolio = PortfolioOptimization(
    expected_returns=mu, covariances=sigma, risk_factor=q, budget=budget, bounds = [(0, budget), (0, budget)]
)
qp = portfolio.to_quadratic_program()
print(qp)
exact_mes = NumPyMinimumEigensolver()
exact_eigensolver = MinimumEigenOptimizer(exact_mes)
algorithm_globals.massive=True
result = exact_eigensolver.solve(qp)
fvals = [s.fval for s in result.samples]
probabilities = [s.probability for s in result.samples]
print('ritorno atteso: ' + str(np.mean(fvals)))
print('dev st: ' + str(np.std(fvals)))
print_result(result)


algorithm_globals.random_seed = 1234
backend = Aer.get_backend("statevector_simulator")
cobyla = COBYLA()
cobyla.set_options(maxiter=500)
ry = TwoLocal(num_assets, "ry", "cz", reps=3, entanglement="full")
quantum_instance = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)
vqe_mes = VQE(ry, optimizer=cobyla, quantum_instance=quantum_instance)
vqe = MinimumEigenOptimizer(vqe_mes)
result = vqe.solve(qp)
fvals = [s.fval for s in result.samples]
probabilities = [s.probability for s in result.samples]
print('ritorno atteso: ' + str(np.mean(fvals)))
print('dev st: ' + str(np.std(fvals)))
print_result(result)