from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
import tensorflow as tf


def dense_layer(input,nodes,activation='relu'):
    layer=layers.Dense(nodes,activation=activation)(input)
    return layer

def auto_encoder(num_features,enc_dim,out_activation):
    # Defining the Input Layer:    
    input_layer=Input(shape=(num_features,))
    enc_layers=input_layer
    if (type(enc_dim) in ['list']) | (len(enc_dim)>1):
        enc_dim.sort(reverse=True)
        for enc in enc_dim:
            if type(enc) not in ['int64','int32','float32','float64']:
                enc=round(enc)
                enc_layers=dense_layer(enc_layers,enc)
            else:
                print('Wrong Dim Type - This Function Only accepts integers')
                return True
        dec_layers=enc_layers
        for dec in enc_dim[::-1][1:]:
            dec=round(dec)
            dec_layers=dense_layer(dec_layers,dec)
    elif (type(enc_dim) in ['int64','int32','float64','float32']) or (len(enc_dim)==1):
        try:
            enc_dim=enc_dim[0]
        except:
            enc_dim=enc_dim
        enc_layers=dense_layer(enc_layers,enc_dim)
        dec_layers=enc_layers
    
    else:
        print('Wrong Enc Dimension Type - Please use List for Multiple Encodings or Integer numbe rfor single Encoding')
        return True
        
    # Defining Models:
    output_layer=dense_layer(dec_layers,num_features,out_activation)
    main_model=Model(input_layer,output_layer)
    encoder=Model(input_layer,enc_layers)
    
    return main_model,encoder