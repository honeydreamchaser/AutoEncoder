import csv
from enum import auto
from json import decoder
from keras.regularizers import l1
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from keras import backend as K
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Reshape, Conv1DTranspose
from keras.models import Model

def load_data(file_path):
    #Import CSV data
    with open(file_path, 'r') as input_file:
        input_reader = csv.reader(input_file)
        
        input_data = []
        for row in input_reader:
            input_data.append(np.array(list(map(float, row[1:69]))).reshape(68, ))
        input_data = np.array(input_data)
    return input_data, input_data

X_train, X_test = load_data("Retailer1.csv")

input_form = Input(shape=(68, ))
hidden_size = 128
code_size = 17

hidden_1 = Dense(hidden_size, activation='relu')(input_form)
code = Dense(code_size, activation='relu', activity_regularizer=l1 (10e-6))(hidden_1)
hidden_2 = Dense(hidden_size, activation='relu')(code)
output = Dense(68, activation='sigmoid')(hidden_2)

autoencoder = Model(input_form, output)
autoencoder.compile(optimizer='adam', loss='mae')
autoencoder.fit(X_train, X_train, epochs=50, batch_size=512)
result = autoencoder.predict(X_train)
print(X_train[1], result[1])
print(result.shape)
