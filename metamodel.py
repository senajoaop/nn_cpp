import numpy as np 
import matplotlib.pyplot as plt
import sys, os
import csv

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix




class MetaModelNN:
    def __init__(self):
        myDict = {}
        with open("res_opt_500.txt", 'r') as file:
            spamreader = csv.reader(file, delimiter=',')
            next(spamreader)
            for row in spamreader:
                dic = {}

                res = row[2:-10]
                r = []
                for i in range(len(res)//4):
                    r.append(res[i*4:i*4+4])
                    
                r = [[float(i) for i in item] for item in r]
                for i in range(len(r)):
                    r[i][0] = int(r[i][0])

                dic['nsol'] = int(row[1]) 
                dic['prop'] = np.array([float(item) for item in row[-10:]])
                dic['sols'] = np.array(r)

                myDict[row[0]] = dic.copy()


        keyToDel = []
        counter = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0}
        for key, value in myDict.items():
            n = str(value['sols'].shape[0])
            counter[n] += 1

            if n=='0':
                keyToDel.append(key)

        for key in keyToDel:
            myDict.pop(key, None)

        for key, value in myDict.items():
            myDict[key]['sols'] = myDict[key]['sols'][0, :]


        self.DS = myDict


    def data_organization(self):
        data, target = [], []
        for key, value in self.DS.items():
            data.append(value['prop'])
            target.append(value['sols'][1])

        data = np.array(data)
        target = np.array(target).reshape((-1, 1))

        if True:
            scaler = StandardScaler()
            data = scaler.fit_transform(data)

        self.data = data
        self.target = target

        self.trainX, self.testX, self.trainY, self.testY = train_test_split(data, target, test_size=0.2, train_size=0.8)

        print(self.trainX.shape)
        print(self.trainY.shape)
        print(data.shape[1])


        print(target)

        sys.exit()




    def training(self):
        self.model = Sequential()
        self.model.add(keras.layers.Dense(10, input_shape=(self.data.shape[1],), activation='relu'))
        # self.model.add(keras.layers.Activation('relu'))
        # self.model.add(keras.layers.Dropout(0.5))
        ###second layer
        self.model.add(keras.layers.Dense(200, activation='relu'))
        # self.model.add(keras.layers.Activation('relu'))
        # self.model.add(keras.layers.Dropout(0.5))
        ###third layer
        self.model.add(keras.layers.Dense(200, activation='relu'))
        # self.model.add(keras.layers.Activation('relu'))
        # self.model.add(keras.layers.Dropout(0.5))
        self.model.add(keras.layers.Dense(200, activation='relu'))
        # self.model.add(keras.layers.Activation('relu'))
        # self.model.add(keras.layers.Dropout(0.5))
        ###final layer
        self.model.add(keras.layers.Dense(1))
        # self.model.add(keras.layers.Activation('softmax'))

        ep = 100

        es = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='min', verbose=1)


        # self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


        print(self.model.summary())



        # plot_model(self.model, 'netstruct.png', show_dtype=True, show_shapes=True)

        # sys.exit()

        H = self.model.fit(self.trainX, self.trainY, epochs=ep, validation_data=(self.testX, self.testY))
        # H = self.model.fit(self.trainX, self.trainY, epochs=ep, validation_data=(self.testX, self.testY), callbacks=[es])

        test_loss, test_acc = self.model.evaluate(self.testX,  self.testY, verbose=2)

        y_pred = self.model.predict(self.testX)
            




















if __name__=="__main__":
    nn = MetaModelNN()
    nn.data_organization()
    nn.training()





