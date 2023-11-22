import csv
from json import decoder
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from keras import backend as K
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Reshape, Conv1DTranspose, AveragePooling1D
from keras.models import Model, Sequential

def load_data(file_path):
    #Import CSV data
    input_form = Input(shape=(68, 1))
    with open(file_path, 'r') as input_file:
        input_reader = csv.reader(input_file)
        
        input_data = []
        for row in input_reader:
            input_data.append(np.array(list(map(float, row[1:69]))).reshape(68, 1))
        input_data = np.array(input_data)
    return input_data, input_form

X_train, input_form = load_data("Retailer1.csv")

#Building the encorder of Autoencorder
input_sig = Input(shape=(68,1))
x = Conv1D(8,3, activation='relu', padding='same')(input_sig)
x1 = MaxPooling1D(2)(x)
x2 = Conv1D(4,3, activation='relu', padding='same')(x1)
x3 = MaxPooling1D(2)(x2)
# x4 = AveragePooling1D()(x3)
flat = Flatten()(x3)
encoded = Dense(68)(flat)
d1 = Dense(68)(encoded)
d2 = Reshape((17,4))(d1)
d3 = Conv1D(4,1,strides=1, activation='relu', padding='same')(d2)
d4 = UpSampling1D(2)(d3)
d5 = Conv1D(8,1,strides=1, activation='relu', padding='same')(d4)
d6 = UpSampling1D(2)(d5)
decoded = Conv1D(1,1,strides=1, activation='sigmoid', padding='same')(d6)
model= Model(input_sig, decoded)

model.summary()

model.compile(optimizer="adam", loss='mse', metrics=["accuracy"])

model.fit(X_train, X_train, epochs=50, batch_size=340)

result = model.predict(X_train)

print(result)