from qiskit.utils import algorithm_globals
import numpy as np
from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit import Aer
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit import BasicAer, execute
import matplotlib.pyplot as plt
from qiskit.algorithms.optimizers import SPSA
from matplotlib.lines import Line2D

algorithm_globals.random_seed = 3142
np.random.seed(algorithm_globals.random_seed)
train_data, train_labels, test_data, test_labels= (
    ad_hoc_data(training_size=20, test_size=5, n=2, gap=0.3, one_hot=False)) # 2 features per ogni classe

# uso ZZFeatureMap per fare data encoding, e TwoLocal come circuito variazionale

adhoc_feature_map = ZZFeatureMap(feature_dimension=2, reps=2)
adhoc_var_form = TwoLocal(2, ['ry', 'rz'], 'cz', reps=2)
adhoc_circuit = adhoc_feature_map.compose(adhoc_var_form)
adhoc_circuit.measure_all()
adhoc_circuit.decompose().draw()

# funzione che associa i dati alla feature map ed i parametri variazionali al circuito variazionale
# si fa per assicurare che i parametri del circuito siano associati alle giuste quantità

def circuit_parameters(data, variational):
    parameters = {}
    for i, p in enumerate(adhoc_feature_map.ordered_parameters):
        parameters[p] = data[i]
    for i, p in enumerate(adhoc_var_form.ordered_parameters):
        parameters[p] = variational[i]
    return parameters

# creiamo una funzione per calcolare la parity di una data bitstring. Se la parity è pari ritorna 1, altrimenti 0

def assign_label(bitstring):
    hamming_weight = sum([int(k) for k in list(bitstring)])
    odd = hamming_weight & 1
    if odd:
        return 0
    else:
        return 1

# creiamo una funzione che ritorna la distribuzione di probabilità sulle classi della label, dato il count sperimentale
# del circuito quantistico lanciato più volte

def label_probability(results):
    shots = sum(results.values())
    probabilities = {0: 0, 1: 0}
    for bitstring, counts in results.items():
        label = assign_label(bitstring)
        probabilities[label] += counts / shots
    return probabilities

def classification_probability(data, variational):
    circuits = [adhoc_circuit.assign_parameters(
        circuit_parameters(d,variational)) for d in data]
    backend = BasicAer.get_backend('qasm_simulator')
    results =  execute(circuits, backend).result()
    classification = [label_probability(results.get_counts(c)) for c in circuits]
    return classification

# per l'addestramento creiamo loss e cost function

def cross_entropy_loss(predictions, expected):
    p = predictions.get(expected)
    return -(expected*np.log(p)+(1-expected)*np.log(1-p))
   
def cost_function(data, labels, variational):
    classifications = classification_probability(data, variational)
    cost = 0
    for i, classification in enumerate(classifications):
        cost += cross_entropy_loss(classification, labels[i])
    cost /= len(data)
    return cost

# Callback function for optimiser for plotting purposes
def store_intermediate_result(evaluation, parameter, cost, 
                              stepsize, accept):
    evaluations.append(evaluation)
    parameters.append(parameter)
    costs.append(cost)

# Set up the optimization

parameters = []
costs = []
evaluations = []

optimizer = SPSA(maxiter=100, callback=store_intermediate_result)

#initial_point = np.random.random(adhoc_var_form.num_parameters)
initial_point = np.array([3.28559355, 5.48514978, 5.13099949,
                          0.88372228, 4.08885928, 2.45568528,
                          4.92364593, 5.59032015, 3.66837805,
                          4.84632313, 3.60713748, 2.43546])

objective_function = lambda variational: cost_function(train_data,
                                                       train_labels,
                                                       variational)

# Run the optimization
opt_var, opt_value, _ = optimizer.optimize(len(initial_point), objective_function, initial_point=initial_point)
fig = plt.figure()
plt.plot(evaluations, costs)
plt.xlabel('Steps')
plt.ylabel('Cost')
plt.show()

# funzione score per valutare il nostro classificatore
def score_classifier(data, labels, variational):
    probability = classification_probability(data, variational)
    prediction = [0 if p[0] >= p[1] else 1 for p in probability]
    accuracy = 0
    for i, p in enumerate(probability):
        if (p[0] >= p[1]) and (labels[i] == 0):
            accuracy += 1
        elif (p[1]) >= p[0] and (labels[i] == 1):
            accuracy += 1
    accuracy /= len(labels)
    return accuracy, prediction

accuracy, prediction = score_classifier(test_data, test_labels, opt_var)
accuracy


plt.figure(figsize=(9, 6))

for feature, label in zip(train_data, train_labels):
    marker = 'o' 
    color = 'C0' if label == 0 else 'C1'
    plt.scatter(feature[0], feature[1],
                marker=marker, s=100, color=color) 
for feature, label, pred in zip(test_data, test_labels, prediction):
    marker = 's' 
    color = 'C0' if pred == 0 else 'C1'
    plt.scatter(feature[0], feature[1],
                marker=marker, s=100, color=color)
    if label != pred:  # mark wrongly classified
        plt.scatter(feature[0], feature[1], marker='o', s=500,
                    linewidths=2.5, facecolor='none', edgecolor='C3')
legend_elements = [
    Line2D([0], [0], marker='o', c='w', mfc='C0', label='A', ms=10),
    Line2D([0], [0], marker='o', c='w', mfc='C0', label='B', ms=10),
    Line2D([0], [0], marker='s', c='w', mfc='C1', label='predict A',
           ms=10),
    Line2D([0], [0], marker='s', c='w', mfc='C0', label='predict B',
           ms=10),
    Line2D([0], [0], marker='o', c='w', mfc='none', mec='C3',
           label='wrongly classified', mew=2, ms=15)
]
plt.legend(handles=legend_elements, bbox_to_anchor=(1, 1),
           loc='upper left')
plt.title('Training & Test Data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()