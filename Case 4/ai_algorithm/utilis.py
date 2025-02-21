# Copyright (c) Tulane University, USA. All rights reserved.
# Author: Bikram K. Parida & Abhijit Sen

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qutip import *
from tcn import TCN, tcn_full_summary
from tqdm import tqdm
import time


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def QuCal():
    
    # N is number of sites for target ising
    N = 10
    Jz = 0.8* np.linspace(0.98, 1, N)
    
    h_alt = np.ones(N)
    
    # Setup operators for individual qubits
    sx_list, sy_list, sz_list = [], [], [] #we need to fill them with required tensor product operators as per the dimentions
    for i in range(N):
        op_list = [qeye(2)] * N # creates a list of N identities
        op_list[i] = 1/2 * sigmax() # replaces the ith list with sigma and in the next step does the tensor product
        sx_list.append(tensor(op_list)) # tensor product of all identities and the sigma above (all of the exists in op_list
        #sx_list has sigmax for each site and the dimentionn of sigmax it contains depends on number of site.
        #print the code separately and see.
        op_list[i] = 1/2 * sigmay()
        sy_list.append(tensor(op_list))
        op_list[i] = 1/2 * sigmaz()
        sz_list.append(tensor(op_list))

    # Hamiltonian - Energy splitting terms
    V_target=0
    V_z = 0
    H_alt = 0
    for i in range(N):
        V_target -=  h_alt[i] * sx_list[i] #the extra term with magnetic field is kept under V_target
        V_z -=  h_alt[i] * sz_list[i]
    # Interaction terms
    for n in range(N - 1):
     
        H_alt +=  -Jz[n] * sz_list[n] * sz_list[n + 1]

    # the periodic boundary condition
    H_alt +=  -Jz[-1] * sz_list[-1] * sz_list[0]

    H0_target = H_alt
    H0_target.eigenenergies()

    E, V = H0_target.eigenstates()

    ψ0  = V[0]


    full_ω = np.linspace(0.55, (E[-1] - E[0])+0.3, 3700)
    t = np.linspace(0, 2 * np.pi / full_ω[0], 512)
    
    plt.plot(np.sin(full_ω[0]*t))
    print('lowest frequency for the data: ',full_ω[-1])
    print('number of frewuencies:', len(full_ω))
    
    return full_ω, t,E


# data pre-processing

def pre_processing(Y , full_ω,t , F0 = 2.5):
            # Create an empty list to store the generated datasets
    F0 = F0
    h1 = -0.8*1.05
    x_dataset = []

    omega_values_x = []

    # Iterate over the omega values and generate the corresponding datasets
    for omega in tqdm(full_ω):
        x = h1 + F0 * np.sin(omega * t)
        x_dataset.append(x)
        omega_values_x.append(omega)

    omega_values_x = np.array(omega_values_x)
    x_dataset = np.array(x_dataset)
    print('omega values for input data: ',omega_values_x.shape)
    print('x data :', x_dataset.shape)
    
    k = 0
    for i in range(50):
        plt.plot(t, Y[k + i, :, 0])
    plt.show()
    
    
        # # Create a StandardScaler object
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    x_dataset_1 = x_dataset.reshape(-1,1)
    Y1 = Y.reshape(-1,1)
    x_dataset_tr = scaler_x.fit_transform(x_dataset_1)
    Y_tr = scaler_y.fit_transform(Y1)
    
    scaled_X1 = x_dataset_tr.reshape(len(x_dataset),len(x_dataset[0]),1)
    scaled_y1 = Y_tr.reshape(len(Y),len(Y[0]),1)
    

    # stacking the omega values along column axis with x_dataset_exp. now the resultant will have the dimension (200,1+128)
    omega_x_data = np.column_stack((omega_values_x,scaled_X1.reshape(len(full_ω),512)))


    omega_Y = np.column_stack((omega_values_x,scaled_y1.reshape(-1,512)))

    X_train, X_test, y_train, y_test = train_test_split(omega_x_data, omega_Y, test_size = 0.2, random_state=42, shuffle = True)

    
    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')
    X_test = X_test.astype('float32')
    y_test = y_test.astype('float32')


    omega_x_train = X_train[:,0]
    scaled_X_train = X_train[:,1:].reshape(len(X_train),len(X_train[0])-1,1)

    omega_x_test = X_test[:,0]
    scaled_X_test = X_test[:,1:].reshape(len(X_test),len(X_test[0])-1,1)

    omega_y_train = y_train[:,0]
    scaled_y_train = y_train[:,1:].reshape(len(y_train), len(y_train[0])-1,1)

    omega_y_test = y_test[:,0]
    scaled_y_test = y_test[:,1:].reshape(len(y_test),len(y_test[0])-1,1)

    print('scaled_X_train: ', scaled_X_train.shape)
    print('scaled_y_train: ', scaled_y_train.shape)
    print('scaled_X_test: ', scaled_X_test.shape)
    print('scaled_y_test: ', scaled_y_test.shape)

    print('scaled_X_train max: ',scaled_X_train.max())
    print('scaled_X_train min: ',scaled_X_train.min())
    print('scaled_y_train max: ', scaled_y_train.max())
    print('scaled_y_train min : ', scaled_y_train.min())
    
    k = 0
    for i in range(50):
        plt.plot(t, scaled_y_train[k + i, :, 0])
    plt.show()
    
    return scaled_X_train, scaled_y_train,scaled_X_test,scaled_y_test, omega_y_test



def plot_losses(loss_sum,val_loss_sum):
    plt.figure(figsize=(8,4))
    plt.plot(loss_sum,color='blue')
    plt.plot(val_loss_sum,color='red')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.savefig('loss_plot.png') 
    plt.show()
