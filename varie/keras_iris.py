import keras #library for neural network
from keras.models import Sequential 
from keras.layers import Dense,Activation,Dropout 
# from keras.layers.normalization import BatchNormalization 
from keras.utils import np_utils
import pandas as pd #loading data in table form   
import matplotlib.pyplot as plt #visualisation
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import normalize #machine learning algorithm library
import time
import statistics

start_time = time.time()
data=pd.read_csv(r'iris.csv')
data.loc[data["variety"]=="Setosa","variety"]=0
data.loc[data["variety"]=="Versicolor","variety"]=1
data.loc[data["variety"]=="Virginica","variety"]=2
print(data.head())
X=data.iloc[:,:4].values
y=data.iloc[:,4].values
total_length=len(data)
train_length=int(0.8*total_length)
test_length=int(0.2*total_length)
X_train=X[:train_length]
X_test=X[train_length:]
y_train=y[:train_length]
y_test=y[train_length:]

print("Length of train set x:",X_train.shape[0],"y:",y_train.shape[0])
print("Length of test set x:",X_test.shape[0],"y:",y_test.shape[0])

y_train=np_utils.to_categorical(y_train,num_classes=3)
y_test=np_utils.to_categorical(y_test,num_classes=3)

lista_accuracy = []

for i in range(5):
    model=Sequential()
    model.add(Dense(1000,input_dim=4,activation='relu'))
    model.add(Dense(500,activation='relu'))
    model.add(Dense(300,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3,activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()

    model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=20,epochs=50,verbose=1)

    prediction=model.predict(X_test)
    length=len(prediction)
    y_label=np.argmax(y_test,axis=1)
    predict_label=np.argmax(prediction,axis=1)

    accuracy=np.sum(y_label==predict_label)/length * 100 
    lista_accuracy.append(accuracy)
    # print("Accuracy of the dataset",accuracy)

print('Accuracy media: ' + str(sum(lista_accuracy) / len(lista_accuracy)))
print('Deviazione standard: ' + str(statistics.stdev(lista_accuracy)))
print("--- %s seconds ---" % np.round((time.time() - start_time), 5))