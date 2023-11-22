import csv
import numpy as np
from aefordr_nn import *

def load_data(file_path):
    #Import CSV data
    input_form = Input(shape=(68, 1))
    for i in range(1, 7):
        with open(file_path + str(i) + ".csv", 'r') as input_file:
            input_reader = csv.reader(input_file)        
            input_data = []
            for row in input_reader:
                input_data.append(np.array(list(map(float, row[1:69]))).reshape(68, 1))
        input_file.close()
    input_data = np.array(input_data)
    return input_data, input_form

callbacks_list=[tf.keras.callbacks.EarlyStopping(
monitor='loss',patience=7),
                tf.keras.callbacks.ReduceLROnPlateau(monitor = "loss", factor = 0.1, patience = 6)
               ]

optimizer_ = tf.keras.optimizers.Adam(learning_rate=0.001)

# Model Fit:
X_train, input_form = load_data("Retailer")
auto_enc,enc = auto_encoder(X_train.shape[1],[64, 16],'sigmoid')
auto_enc.compile(optimizer=optimizer_,loss='cosine_similarity',metrics=['cosine_similarity'])
auto_enc_fitted = auto_enc.fit(X_train, X_train, epochs=20, batch_size=64, callbacks=callbacks_list)

for i in range(1, 7):
    with open("Retailer" + str(i) + ".csv", 'r') as input_file:
        input_reader = csv.reader(input_file)        
        input_data = []
        for row in input_reader:
            input_data.append(np.array(list(map(float, row[1:69]))).reshape(68, 1))
        input_data = np.array(input_data)
        encoded_features=enc.predict(input_data)
        with open("Retailer" + str(i) + "_AutoEn.csv", 'w', newline="") as output_file:
            out_writer = csv.writer(output_file)
            out_writer.writerows([row for row in encoded_features])
        print(encoded_features)
        output_file.close()
    input_file.close()