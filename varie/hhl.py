import numpy as np
from qiskit.algorithms.linear_solvers.numpy_linear_solver import NumPyLinearSolver
from qiskit.algorithms.linear_solvers.hhl import HHL
from qiskit.algorithms.linear_solvers.matrices.tridiagonal_toeplitz import TridiagonalToeplitz

# matrix = np.array([[1, -1/3], [-1/3, 1]])
matrix = np.array([0, 2])
matrix = np.expand_dims(matrix, axis=0)
vector = np.array([1, 0])

naive_hhl_solution = HHL().solve(matrix, vector)
# print(naive_hhl_solution)
classical_solution = NumPyLinearSolver().solve(matrix, vector / np.linalg.norm(vector))
# print(classical_solution)
tridi_matrix = TridiagonalToeplitz(1, 1, -1 / 3)
# print(tridi_matrix)
tridi_solution = HHL().solve(tridi_matrix, vector)
# print(tridi_solution)
print('classical state:', classical_solution.state)