# bug

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA  
from qiskit import Aer 
from qiskit.aqua.components.feature_maps import SecondOrderExpansions, FirstOrderExpansions
from qiskit.aqua.algorithms import QSVM
import numpy as np
import matplotlib.pyplot as plt

class QSVM_routine:
    def __init__(self, feature_dim = 2, feature_depth = 2, train_test_split = 0.3, train_samples = 5, test_sample = 2, seed = 0, copies = 5):
        self.feature_dim = feature_dim
        self.feature_depth = feature_depth
        self.train_test_split = train_test_split
        self.train_samples = train_samples
        self.test_sample = test_sample
        self.seed = seed
        self.copies
    
    # crea i dataset di train e test
    def train_test_datasets(self):
        self.class_labels = [r'A', r'B']
        data, target = datasets.load_breast_cancer(True)
        train_X, test_X, train_Y, test_Y = train_test_split(data, target, test_size = self.train_test_split, random_state = self.seed)
        # normalizzazione
        self.z_scale = StandardScaler().fit(train_X)
        self.train_X_norm = self.z_scale.transform(train_X)
        self.test_X_norm = self.z_scale.transform(test_X)
        # proietta i dati in un numero di dimensioni uguale al numero di qubit
        self.pca = PCA(n_components = self.feature_dim).fit(self.train_X_norm)
        self.train_X_norm = self.pca.transform(self.train_X_norm)
        self.test_X_norm = self.pca.transform(self.test_X_norm)
        # scala nel range (-1, 1)
        X_all = np.append(self.train_X_norm, self.test_X_norm, axis = 0)
        minmax_scale = MinMaxScaler((-1, 1)).fit(X_all)
        self.train_X_norm = minmax_scale.transform(self.train_X_norm)
        self.test_X_norm = minmax_scale.transform(self.test_X_norm)
        # Prendi training e test number del data point
        self.train = {key: (self.train_X_norm[train_y == k, :]) [:self.train_samples] for k, key in enumerate(self.class_labels)}
        self.test = {key: (self.test_X_norm[test_y == k, :]) [:self.test_samples] for k, key in enumerate(self.class_labels)}

        