# Copyright (c) Tulane University, USA. All rights reserved.
# Author: Bikram K. Parida & Abhijit Sen

import numpy as np
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D,Dense
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score,r2_score
from keras.losses import Huber
import keras.backend as K
from keras.layers import Activation
from tcn import TCN, tcn_full_summary



input_size = 512

def swish_activation(x, beta=1.2):
    return x * K.sigmoid(beta * x)


def TCN_autoencoder(input_size=512,f1 = 8,f2 = 6, lf = 6,latent_space_tcn = True,last_2_layer_dense = False):
    # Encoder
    input_data = Input(shape=(input_size, 1))  # Assuming one channel for simplicity
    encoded = TCN(nb_filters=f1, kernel_size=3, dilations=[2, 4,8],kernel_initializer='he_normal',activation= swish_activation, padding='same', return_sequences=True)(input_data)
    encoded = MaxPooling1D(pool_size=2)(encoded)
    encoded = TCN(nb_filters=f2, kernel_size=3, dilations=[2, 4,8],kernel_initializer='he_normal',activation=swish_activation, padding='same', return_sequences=True)(encoded)
    encoded = MaxPooling1D(pool_size=2)(encoded)
    
    #latent
    if latent_space_tcn:
        encoded =TCN(nb_filters=lf, kernel_size=3, dilations=[2, 4,8],kernel_initializer='he_normal',activation=swish_activation, padding='same', return_sequences=True)(encoded)
    else:
        encoded = Dense(lf,activation=swish_activation) (encoded)
        
    
    
    decoded = TCN(nb_filters=f2, kernel_size=3, dilations=[2, 4,8],kernel_initializer='he_normal',activation=swish_activation, padding='same', return_sequences=True)(encoded)
    decoded = UpSampling1D(size=2)(decoded)
    decoded = TCN(nb_filters=f1, kernel_size=3, dilations=[2, 4,8],kernel_initializer='he_normal',activation=swish_activation, padding='same', return_sequences=True)(decoded)
    decoded = UpSampling1D(size=2)(decoded)

    if last_2_layer_dense:
        decoded = Dense(4, activation=swish_activation)(decoded)
        decoded = Dense(1)(decoded)
    else:
        decoded = Dense(1)(decoded)
#     decoded =TCN(nb_filters=1, kernel_size=3, dilations=[ 2, 4,8],kernel_initializer='he_normal',activation=swish_activation, padding='same', return_sequences=True)(decoded)
    
    # Autoencoder model
    autoencoder = Model(input_data, decoded)

    return autoencoder
