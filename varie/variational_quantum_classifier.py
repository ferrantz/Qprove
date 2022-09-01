from qiskit import BasicAer
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import TwoLocal, ZZFeatureMap, ZFeatureMap, PauliFeatureMap, RealAmplitudes
from qiskit_machine_learning.algorithms import VQC
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt

start_time = time.time()
seed = 542
algorithm_globals.random_seed = seed

feature_dim = 4  # dimension of each data point
training_size = 120
test_size = 30

data=pd.read_csv(r'C:\\Users\\italo\\Desktop\\Qprove\\varie\\iris.csv')
data.loc[data["variety"]=="Setosa","variety"]=0
data.loc[data["variety"]=="Versicolor","variety"]=1
data.loc[data["variety"]=="Virginica","variety"]=2
data=data.iloc[np.random.permutation(len(data))]
X=data.iloc[:,:4].values
y=data.iloc[:,4].values
total_length=len(data)
train_length=int(0.8*total_length)
test_length=int(0.2*total_length)
training_features=X[:train_length]
test_features=X[train_length:]
training_labels=y[:train_length]
test_labels=y[train_length:]

print('training_features.shape' + str(training_features.shape))
print('training_labels.shape' + str(training_labels.shape))
print('test_features.shape' + str(test_features.shape))
print('test_labels.shape' + str(test_labels.shape))

enc = OneHotEncoder()
training_labels = enc.fit_transform(training_labels[:, np.newaxis]).toarray()
test_labels = enc.fit_transform(test_labels[:, np.newaxis]).toarray()


# capire se aumentare le reps persegue il no-cloning principle (probabilmente no, anche perché i risultati sembrano peggiorare 
# a differenza di https://www.frontiersin.org/articles/10.3389/fphy.2020.00297/full)
feature_map = ZFeatureMap(feature_dimension=feature_dim, reps=3) #, entanglement="full"
# feature_map = PauliFeatureMap(feature_dimension=feature_dim, reps=2) #, entanglement="full"
print('feature_map: \n')
feature_map.decompose().draw()
print('feature_map.num_qubits: ' + str(feature_map.num_qubits))
# ansatz = TwoLocal(feature_map.num_qubits, ['ry', 'rz'], 'cz', reps=3)
# ansatz = TwoLocal(feature_map.num_qubits, 'ry', 'cx', 'full', reps=2, insert_barriers=True) # fatto
ansatz=RealAmplitudes(feature_map.num_qubits, reps=1, insert_barriers=True)
print('ansatz: \n')
print(ansatz.decompose().draw())
vqc = VQC(feature_map=feature_map,
          ansatz=ansatz,
          optimizer=COBYLA(maxiter=100),
          quantum_instance=QuantumInstance(BasicAer.get_backend('statevector_simulator'), # statevector_simulator
                                           shots=1024,
                                           seed_simulator=seed,
                                           seed_transpiler=seed)
          )
vqc.fit(training_features, training_labels)

score = vqc.score(test_features, test_labels)
print(f"Testing accuracy: {score:0.2f}")
print("--- %s seconds ---" % np.round((time.time() - start_time), 5))