# https://medium.com/mlearning-ai/classification-using-quantum-kernels-tutorial-8a2f442fd188
# per il clustering con quantum kernel vedi: https://medium.com/mlearning-ai/clustering-using-quantum-kernels-tutorial-dcd90bf6440c
from imports import *

seed = 123456
h = 0.1  # step size in the mesh
names = ["Linear SVM", "RBF SVM", "QKernel_Default", "QKernel_Eq8", "QKernel_Eq9", "QKernel_Eq10", \
                "QKernel_Eq11", "QKernel_Eq12"]
rng = np.random.RandomState(2)
i = 1
# perché in ogni parametro si moltiplica per 2 (a prescindere della feature dimension)
qfm_default = PauliFeatureMap(feature_dimension=2, 
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=2, entanglement='full')
qfm_eq8 = PauliFeatureMap(feature_dimension=2, 
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=2, entanglement='full', data_map_func=data_map_eq8)
qfm_eq9 = PauliFeatureMap(feature_dimension=2, 
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=2, entanglement='full', data_map_func=data_map_eq9)
qfm_eq10 = PauliFeatureMap(feature_dimension=2, 
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=2, entanglement='full', data_map_func=data_map_eq10)
qfm_eq11 = PauliFeatureMap(feature_dimension=2, 
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=2, entanglement='full', data_map_func=data_map_eq11)
qfm_eq12 = PauliFeatureMap(feature_dimension=2, 
                                    paulis = ['ZI','IZ','ZZ'],
                                 reps=2, entanglement='full', data_map_func=data_map_eq12)

qcomp_backend = QuantumInstance(BasicAer.get_backend('qasm_simulator'), shots=1024,
                                seed_simulator=seed, seed_transpiler=seed)

qkern_default = QuantumKernel(feature_map=qfm_default, quantum_instance=qcomp_backend)
qkern_eq8 = QuantumKernel(feature_map=qfm_eq8, quantum_instance=qcomp_backend)
qkern_eq9 = QuantumKernel(feature_map=qfm_eq9, quantum_instance=qcomp_backend)
qkern_eq10 = QuantumKernel(feature_map=qfm_eq10, quantum_instance=qcomp_backend)
qkern_eq11 = QuantumKernel(feature_map=qfm_eq11, quantum_instance=qcomp_backend)
qkern_eq12 = QuantumKernel(feature_map=qfm_eq12, quantum_instance=qcomp_backend)

classifiers = [
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    SVC(kernel=qkern_default.evaluate), # .evaluate costruisce una quantum matrix per dei dati ed una feature map dati
    SVC(kernel=qkern_eq8.evaluate),
    SVC(kernel=qkern_eq9.evaluate),
    SVC(kernel=qkern_eq10.evaluate),
    SVC(kernel=qkern_eq11.evaluate),
    SVC(kernel=qkern_eq12.evaluate),
              ]

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)

print('rng: ' + str(rng))
X += 2 * rng.uniform(size=X.shape)

linearly_separable = (X, y)

datasets = [make_moons(noise=0.05, random_state=0),
            make_circles(noise=0.05, factor=0.5, random_state=1),
            linearly_separable,
            ]

figure = plt.figure(figsize=(24, 12))

for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(f'Callable kernel classification test score for {name}: {score}')

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name,size=18)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=18, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()