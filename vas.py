import csv
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA

from sklearn.metrics import mean_squared_error, silhouette_score


import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

cols = ['#1FC17B', '#78FECF', '#555B6E', '#CC998D', '#429EA6',
        '#153B50', '#8367C7', '#EE6352', '#C287E8', '#F0A6CA', 
        '#521945', '#361F27', '#828489', '#9AD2CB', '#EBD494', 
        '#53599A', '#80DED9', '#EF2D56', '#446DF6', '#AF929D']

def load_data(file_path):
    #Import CSV data
    with open(file_path, 'r') as input_file:
        input_reader = csv.reader(input_file)
        
        input_data = []
        for row in input_reader:
            input_data.append(np.array(list(map(float, row[1:69]))))
        input_data = np.array(input_data)
    return input_data, input_data

X, y = load_data("Retailer1.csv")

print(X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=17)
print(X_train.shape)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
pca = PCA(n_components=2)
pca.fit(X_train)

res_pca = pca.transform(X_test)
print("_________RES_PCA___________")
print(res_pca.shape)

unique_labels = np.unique(y_test)

# for index, unique_label in enumerate(unique_labels):
#     X_data = res_pca[y_test==unique_label]
#     plt.scatter(X_data[:,0], X_data[:,1], alpha=0.3, c=cols[index])
    
# plt.xlabel('Principal Component #1')
# plt.ylabel('Principal Component #2')
# plt.title('PCA Results')

autoencoder = MLPRegressor(alpha=1e-15, 
                           hidden_layer_sizes=(50, 100, 50, 2, 50, 100, 50), 
                           random_state=1, max_iter=20000)

autoencoder.fit(X_train, X_train)

W = autoencoder.coefs_
biases = autoencoder.intercepts_
encoder_weights = W[0:4]
encoder_biases = biases[0:4]
def encoder(encoder_weights, encoder_biases, data):
    res_ae = data
    for index, (w, b) in enumerate(zip(encoder_weights, encoder_biases)):
        if index+1 == len(encoder_weights):
            res_ae = res_ae@w+b 
        else:
            res_ae = np.maximum(0, res_ae@w+b)
    return res_ae
            
res_ae = encoder(encoder_weights, encoder_biases, X_test)
print("_________RES_AE___________")
print(res_ae.shape)

res_de =  autoencoder.predict(X_test)
print("_________RES_DE___________")
print(res_de.shape)
print(y_test)
unique_labels = np.unique(y_test)

silhouette_score(X_train, y_train)

silhouette_score(res_pca, y_test)

silhouette_score(res_ae, y_test)