from re import X
import seaborn as sns
from matplotlib import pyplot as plt
import ast
import numpy as np

sns.set_style("white")

with open(r"C:/Users/italo/Desktop/Qprove/due_asset") as f:
    lines_due = f.readlines()
with open(r"C:/Users/italo/Desktop/Qprove/tre_asset") as f:
    lines_tre = f.readlines()
with open(r"C:/Users/italo/Desktop/Qprove/quattro_asset") as f:
    lines_quattro = f.readlines()
with open(r"C:/Users/italo/Desktop/Qprove/cinque_asset") as f:
    lines_cinque = f.readlines()
with open(r"C:/Users/italo/Desktop/Qprove/sei_asset") as f:
    lines_sei = f.readlines()
with open(r"C:/Users/italo/Desktop/Qprove/sette_asset") as f:
    lines_sette = f.readlines()
with open(r"C:/Users/italo/Desktop/Qprove/otto_asset") as f:
    lines_otto = f.readlines()

d_simulatore = ast.literal_eval(lines_quattro[0])
d_simulatore = {x: d_simulatore[x] for x in d_simulatore if x not in 'tempo_esecuzione'}
d_reale = ast.literal_eval(lines_quattro[2])
d_reale = {x: d_reale[x] for x in d_reale if x not in 'tempo_esecuzione'}
x = list(d_simulatore.values())
y = list(d_reale.values())
# print(list(d_simulatore.keys())[:len(y)])

width = 0.8
fig, axs = plt.subplots(2)
fig.suptitle('Bar chart caso con ' + str(int(np.log2(len(d_simulatore.keys())))) + ' qubit')
axs[0].bar(list(d_simulatore.keys())[:len(y)], list(d_simulatore.values())[:len(y)], width=width, 
        color='b', label='Valori simulatore')
axs[1].bar(list(d_simulatore.keys())[:len(y)], list(d_reale.values()), 
        width= width, color='r', alpha=0.5, label='Valori macchina reale')
plt.xticks(range(len(d_reale.keys())), list(d_reale.keys()))
axs[0].set_yticks(np.arange(0, 1.1, 0.1))
axs[1].set_yticks(np.arange(0, 1.1, 0.1))
plt.tight_layout()
plt.show()