# Copyright (c) Tulane University, USA. All rights reserved.
# Author: Bikram K. Parida & Abhijit Sen

import numpy as np
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D,Dense, Flatten, Reshape, Lambda
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



    


def TCN_autoencoder(input_size=512,f1 = 8,f2 = 4,f3=1,  lf = 10,latent_space_tcn = True,last_2_layer_dense = False):
    # Encoder
    input_data = Input(shape=(input_size, 1))  # Assuming one channel for simplicity
    encoded = TCN(nb_filters=f1, kernel_size=3, dilations=[2, 4,8],kernel_initializer='he_normal',activation= swish_activation, padding='same', return_sequences=True)(input_data)
    encoded = MaxPooling1D(pool_size=2)(encoded)
    encoded = TCN(nb_filters=f2, kernel_size=3, dilations=[2, 4,8],kernel_initializer='he_normal',activation=swish_activation, padding='same', return_sequences=True)(encoded)
    encoded = MaxPooling1D(pool_size=2)(encoded)
    encoded = TCN(nb_filters=f3, kernel_size=3, dilations=[2, 4,8],kernel_initializer='he_normal',activation=swish_activation, padding='same', return_sequences=True)(encoded)
    encoded = MaxPooling1D(pool_size=2)(encoded)
    
    #latent
    if latent_space_tcn:
        encoded =TCN(nb_filters=lf, kernel_size=3, dilations=[2, 4,8],kernel_initializer='he_normal',activation=swish_activation, padding='same', return_sequences=True)(encoded)
    else:
        encoded = Dense(lf,activation=swish_activation) (encoded)
        
    
    decoded = TCN(nb_filters=f3, kernel_size=3, dilations=[2, 4,8],kernel_initializer='he_normal',activation=swish_activation, padding='same', return_sequences=True)(encoded)
    decoded = UpSampling1D(size=2)(decoded)
    
    decoded = TCN(nb_filters=f2, kernel_size=3, dilations=[2, 4,8],kernel_initializer='he_normal',activation=swish_activation, padding='same', return_sequences=True)(decoded)
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


# For TCN_VAE_Autoencoder

# Define sampling function
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Variational Autoencoder model with TCN layers
def TCN_VAE_AE(input_size=512, f1=12, f2=12,f3 = 10, latent_dim=10, max_pool = 2,dilation = [2, 4, 8]):
    # Encoder
    input_data = Input(shape=(input_size, 1))  # Assuming one channel for simplicity
    x = TCN(nb_filters=f1, kernel_size=3, dilations=dilation, padding='same', activation=swish_activation, return_sequences=True)(input_data)
    x = MaxPooling1D(pool_size=max_pool)(x)
    x = TCN(nb_filters=f2, kernel_size=3, dilations=dilation, padding='same', activation=swish_activation, return_sequences=True)(x)
    x = MaxPooling1D(pool_size=max_pool)(x)
    x = TCN(nb_filters=f3, kernel_size=3, dilations=dilation, padding='same', activation=swish_activation, return_sequences=True)(x)
    x = MaxPooling1D(pool_size=max_pool)(x)
    
    # Flatten
    x = Flatten()(x)
    
    # Latent space
    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    # Decoder
#     decoder_input = Input(shape=(latent_dim,))
    x = Dense(input_size//8 * f3, activation=swish_activation)(z)
    x = Reshape((input_size//8, f3))(x)
    x = TCN(nb_filters=f3, kernel_size=3, dilations=dilation, padding='same', activation=swish_activation, return_sequences=True)(x)
    x = UpSampling1D(size=max_pool)(x)
    x = TCN(nb_filters=f2, kernel_size=3, dilations=dilation, padding='same', activation=swish_activation, return_sequences=True)(x)
    x = UpSampling1D(size=max_pool)(x)
    x = TCN(nb_filters=f1, kernel_size=3, dilations=dilation, padding='same', activation=swish_activation, return_sequences=True)(x)
    if x.shape[1]==512:
        output_data = Dense(1)(x)
    else:
        x = UpSampling1D(size=max_pool)(x)
        output_data = Dense(1)(x)

    vae = Model(input_data, output_data)
    
    return vae