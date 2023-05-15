import numpy as np 
import matplotlib.pyplot as plt
import sys, os
import csv

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import LSTM
from keras.regularizers import l1, l2
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import SGD
import keras
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix
from scipy.io import savemat, loadmat
import kerastuner as kt




class MetaModelNN:
    def __init__(self):
        myDict = {}
        with open("/../data/res_opt_20000.txt", 'r') as file:
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

        self.scaler = StandardScaler()
        data = self.scaler.fit_transform(data)

        # self.data = data
        # self.target = target
        
        self.trainX, self.testX, self.trainY, self.testY = train_test_split(data, target, test_size=0.2, train_size=0.8)
       
        # print(self.trainX.shape)
        # print(self.trainY.shape)
        # print(data.shape[1])

        self.normalizer = tf.keras.layers.Normalization(axis=-1)
        self.normalizer.adapt(np.array(self.trainX))


        savemat("models/trainX.mat", {'trainX': self.trainX})
        savemat("models/testX.mat", {'testX': self.testX})
        savemat("models/trainY.mat", {'trainY': self.trainY})
        savemat("models/testY.mat", {'testY': self.testY})

        # print(self.normalizer.mean.numpy())

        # print(self.trainY)

        # sys.exit()

        # print(normalizer.mean.numpy())

        # print(normalizer(self.trainX[:1]).numpy())
        # sys.exit()




    def training(self):
        self.model = Sequential()
        self.model.add(keras.layers.Dense(10, input_shape=(self.data.shape[1],), activation='relu'))
        # self.model.add(keras.layers.Activation('relu'))
        # self.model.add(keras.layers.Dropout(0.5))
        ###second layer
        self.model.add(keras.layers.Dense(64, activation='relu'))
        # self.model.add(keras.layers.Activation('relu'))
        # self.model.add(keras.layers.Dropout(0.5))
        ###third layer
        self.model.add(keras.layers.Dense(64, activation='relu'))
        # self.model.add(keras.layers.Activation('relu'))
        # self.model.add(keras.layers.Dropout(0.5))
        # self.model.add(keras.layers.Dense(200, activation='relu'))
        # self.model.add(keras.layers.Activation('relu'))
        # self.model.add(keras.layers.Dropout(0.5))
        ###final layer
        self.model.add(keras.layers.Dense(1))
        # self.model.add(keras.layers.Activation('softmax'))

        ep = 100

        es = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='min', verbose=1)


        # self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # self.model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['accuracy'])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='mean_absolute_error')


        print(self.model.summary())



        # plot_model(self.model, 'netstruct.png', show_dtype=True, show_shapes=True)

        # sys.exit()

        H = self.model.fit(self.trainX, self.trainY, epochs=ep, validation_data=(self.testX, self.testY))
        # H = self.model.fit(self.trainX, self.trainY, epochs=ep, validation_data=(self.testX, self.testY), callbacks=[es])

        test_loss, test_acc = self.model.evaluate(self.testX,  self.testY, verbose=2)

        y_pred = self.model.predict(self.testX)
            



    def training_2(self):
        

        model = keras.Sequential([
            # self.normalizer,
            keras.layers.Input(self.trainX.shape[1]),
            keras.layers.Dense(100, activation='relu', kernel_regularizer=l2(0.001)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(400, activation='relu', kernel_regularizer=l2(0.001)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(800, activation='relu', kernel_regularizer=l2(0.001)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(400, activation='relu', kernel_regularizer=l2(0.001)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(1),
        ])

        model.compile(loss='mean_absolute_error',
                      optimizer=tf.keras.optimizers.Adam(0.0001))

        model.summary()

        history = model.fit(
            self.trainX,
            self.trainY,
            validation_split=0.2,
            # verbose=0,
            epochs=50)

        results = model.evaluate(self.testX, self.testY)


        test_predictions = model.predict(self.testX).flatten()

        a = plt.axes(aspect='equal')
        plt.scatter(self.testY, test_predictions)
        plt.xlabel('True Values [R]')
        plt.ylabel('Predictions [R]')
        lims = [80000, 180000]
        plt.xlim(lims)
        plt.ylim(lims)
        _ = plt.plot(lims, lims)

        plt.show()



        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        # plt.ylim([0, 10])
        plt.xlabel('Epoch')
        plt.ylabel('Error [R]')
        plt.legend()
        plt.grid(True)

        plt.show()


    def hyperparameter(self):
        def build_model_comp(hp):
            model = Sequential()
            model.add(self.normalizer)
            counter = 0
            for i in range(hp.Int('num_layers', min_value=1, max_value=6)):
                model.add(Dense(hp.Int('units' + str(i), min_value=64, max_value=640, step=64),
                                activation= hp.Choice('activation' + str(i), values=['relu','tanh','sigmoid']),
                                kernel_initializer=hp.Choice(f'kernel_init{i}', values=['he_normal', 'random_uniform']),
                                kernel_regularizer=l2(hp.Choice(f'regularizer{i}', values=[0.001, 0.0001, 0.00001, 0.000001]))))
                model.add(Dropout(hp.Choice('dropout' + str(i), values=[0.05, 0.1, 0.2, 0.3])))
                counter+=1
            model.add(Dense(1))
            model.compile(optimizer=hp.Choice('optimizer', values=['rmsprop','adam','sgd','nadam','adadelta']),
                      loss='mean_absolute_error')


            return model


        def build_model(hp):
            model = Sequential()
            model.add(self.normalizer)
            model.add(Dense(64,activation='relu'))
            model.add(Dense(64,activation='relu'))
            model.add(Dense(1))
            optimizer = hp.Choice('optimizer', values = ['adam','sgd','rmsprop','adadelta'])
            model.compile(optimizer=optimizer, loss='mean_absolute_error')

            return model
        

        tuner = kt.RandomSearch(build_model_comp,
                        objective='val_loss',
                        max_trials=100,
                        directory='keras_tuner_data',
                        project_name='nncpp_tuner',
                        overwrite=True)

        tuner.search(self.trainX, self.trainY, epochs=100, validation_data=(self.testX, self.testY))
        tuner.get_best_hyperparameters()[0].values
        print(tuner.get_best_hyperparameters()[0].values)

        model = tuner.get_best_models(num_models=1)[0]
        model.summary()

        model.save("models/hyper_model")
        model.save("models/hyper_model.h5")








if __name__=="__main__":
    nn = MetaModelNN()
    nn.data_organization()
    # nn.training()
    # nn.training_2()
    nn.hyperparameter()





