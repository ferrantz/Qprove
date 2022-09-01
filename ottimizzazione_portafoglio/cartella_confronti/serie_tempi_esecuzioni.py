import ast
from matplotlib import pyplot as plt

lista_simulatore = []
lista_reale = []

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

lista_simulatore.append(ast.literal_eval(lines_due[0])['tempo_esecuzione'])
lista_reale.append(ast.literal_eval(lines_due[2])['tempo_esecuzione'])
lista_simulatore.append(ast.literal_eval(lines_tre[0])['tempo_esecuzione'])
lista_reale.append(ast.literal_eval(lines_tre[2])['tempo_esecuzione'])
lista_simulatore.append(ast.literal_eval(lines_quattro[0])['tempo_esecuzione'])
lista_reale.append(ast.literal_eval(lines_quattro[2])['tempo_esecuzione'])
lista_simulatore.append(ast.literal_eval(lines_cinque[0])['tempo_esecuzione'])
lista_reale.append(ast.literal_eval(lines_cinque[2])['tempo_esecuzione'])
lista_simulatore.append(ast.literal_eval(lines_sei[0])['tempo_esecuzione'])
lista_reale.append(ast.literal_eval(lines_sei[2])['tempo_esecuzione'])
lista_simulatore.append(ast.literal_eval(lines_sette[0])['tempo_esecuzione'])
lista_reale.append(ast.literal_eval(lines_sette[2])['tempo_esecuzione'])
lista_simulatore.append(ast.literal_eval(lines_otto[0])['tempo_esecuzione'])
lista_reale.append(ast.literal_eval(lines_otto[2])['tempo_esecuzione'])

print(lista_simulatore)
print(lista_reale)

x = lista_simulatore.copy()
y = lista_reale.copy()
z = [_ for _ in range(2, 9)]
print(z)
fig = plt.figure()
# ax = fig.add_axes([2, 2, 2, 1])
plt.plot(z, x, label = "Tempo simulatore")
plt.plot(z, y, label = "Tempo macchina reale")
plt.xticks(z)
plt.xlabel("Numero asset")
plt.ylabel("Tempo in secondi")
plt.title("Tempi di esecuzione al crescere degli asset")
plt.legend()
plt.show()