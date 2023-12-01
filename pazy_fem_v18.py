#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 22:40:25 2022

@author: rigm
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import numpy as np
import pandas as pd
import tensorflow as tf
# from tensorflow.keras import regularizers
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler,MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense,Activation, Input
# from tensorflow.keras.optimizers import Adam
import pickle
from scipy.interpolate import interp1d

plot_results = True

if plot_results:
    AoA_exp_onset_up = [3 , 5 , 7]
    AoA_exp_onset_down = [3 , 5]
    AoA_exp_offset_up = [3 , 5 ]
    AoA_exp_offset_down = [3 , 5 , 7]
    q_onset_up_exp = [48.24 , 41.44 , 37.19]
    q_onset_down_exp = [56.37 , 47.23]
    q_offset_up_exp = [40.1 , 36]
    q_offset_down_exp = [58 , 51 , 46]
    # Load flutter boundary
    df=pd.read_csv('q_flutter_onset_offset_AoA.csv')
    q_flutter_onset = df['Q_flutter_onset'].values
    q_flutter_offset = df['Q_flutter_offset'].values
    AoA_input = df['AoA'].values

    # q_flutter_offset = np.nan_to_num(q_flutter_offset)

    # Set up the interpolation function
    interp_onset = interp1d(AoA_input, q_flutter_onset, kind='cubic')
    interp_offset = interp1d(AoA_input, q_flutter_offset, kind='cubic')

    # Create new x and y values for the interpolation
    AoA_input_smooth = np.linspace(min(AoA_input), max(AoA_input), 200)
    q_flutter_onset_smooth = interp_onset(AoA_input_smooth)
    q_flutter_offset_smooth = interp_offset(AoA_input_smooth)

    plt.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(1,1)
    fig.set_size_inches(9.25, 5.25)
    ax.plot(q_flutter_onset_smooth,AoA_input_smooth,'k-',label = 'present',linewidth=2)
    ax.plot(q_flutter_offset_smooth,AoA_input_smooth,'k-',linewidth=2)
    ax.scatter(q_onset_up_exp,AoA_exp_onset_up, s=70, marker='>', c='red',label = 'Exp Onset Up')
    ax.scatter(q_onset_down_exp,AoA_exp_onset_down,s=70, marker='<', c='red',  label = 'Exp Onset Down')
    ax.scatter(q_offset_up_exp,AoA_exp_offset_up, s=70, marker='>', c='grey',label = 'Exp Offset Up')
    ax.scatter(q_offset_down_exp,AoA_exp_offset_down,s=70, marker='<', c='grey',  label = 'Exp Offset Down')
    ax.set_xlabel(r'$v \,\, [m/s]$',fontsize=16)
    ax.set_ylabel(r'$\alpha \,\, [deg]$',fontsize=16)
    ax.legend(loc='upper right',fontsize=16)
    plt.grid()
    # plt.yticks(np.arange(0, 2.5, step=0.5))
    plt.ylim(0, 10)
    plt.xlim(10, 100)
    plt.savefig('Photo\\'+ 'flutter.png', bbox_inches='tight')
    # plt.show()
    plt.close()



# Functions
def beam(EA,EIy,EIz,l,GJ,mass_unit_length,moment_unit_length,ycg,ysc):
   
    M = np.zeros((12,12))
    K = np.zeros((12,12))
   
    K[0,0] = EA/l
    K[6,0] = -EA/l
    K[0,6] = -EA/l
    K[6,6] = EA/l
   
    K[3,3] = GJ/l
    K[9,3] = -GJ/l
    K[3,9] = -GJ/l
    K[9,9] = GJ/l
   
    K[1,1] = 12.*EIz / l**3
    K[1,7] = -12.*EIz / l**3
    K[7,1] = -12.*EIz / l**3
    K[7,7] = 12.*EIz / l**3

    K[1,5]  =  6.*EIz/l**2    
    K[1,11] =  6.*EIz/l**2    
    K[7,5]  = -6.*EIz/l**2    
    K[7,11] = -6.*EIz/l**2    

    K[5,1]  =  6.*EIz/l**2    
    K[11,1] =  6.*EIz/l**2    
    K[5,7]  = -6.*EIz/l**2    
    K[11,7] = -6.*EIz/l**2    

    K[5,5]  =  4.*EIz/l    
    K[5,11] =  2.*EIz/l    
    K[11,5]  = 2.*EIz/l    
    K[11,11] = 4.*EIz/l    

    K[2,2] =  12.*EIy / l**3
    K[2,8] = -12.*EIy / l**3
    K[8,2] = -12.*EIy / l**3
    K[8,8] =  12.*EIy / l**3

    K[2,4]  = -6.*EIy/l**2    
    K[2,10] = -6.*EIy/l**2    
    K[8,4]  =  6.*EIy/l**2    
    K[8,10] =  6.*EIy/l**2    

    K[4,2]  = -6.*EIy/l**2    
    K[4,8]  =  6.*EIy/l**2    
    K[10,2] = -6.*EIy/l**2    
    K[10,8] =  6.*EIy/l**2    

    K[4,4]  =  4.*EIy/l    
    K[4,10] =  2.*EIy/l    
    K[10,4]  = 2.*EIy/l    
    K[10,10] = 4.*EIy/l    
   
    K[2,3] = 12.*EIy*ysc/l**3
    K[3,2] = 12.*EIy*ysc/l**3
    K[8,9] = 12.*EIy*ysc/l**3
    K[9,8] = 12.*EIy*ysc/l**3
    K[2,9] = -12.*EIy*ysc/l**3
    K[9,2] = -12.*EIy*ysc/l**3
    K[8,3] = -12.*EIy*ysc/l**3
    K[3,8] = -12.*EIy*ysc/l**3
   
    K[4,3] =-6.*EIy*ysc/l**2
    K[3,4] =-6.*EIy*ysc/l**2
    K[10,9] = 6.*EIy*ysc/l**2
    K[9,10] = 6.*EIy*ysc/l**2
    K[4,9] =  6.*EIy*ysc/l**2
    K[9,4] = 6.*EIy*ysc/l**2
    K[10,3] =-6.*EIy*ysc/l**2
    K[3,10] =-6.*EIy*ysc/l**2

   
    M[0,0] = l*mass_unit_length/3.
    M[6,0] = l*mass_unit_length/6.
    M[0,6] = l*mass_unit_length/6.
    M[6,6] = l*mass_unit_length/3.
   
    M[3,3] = l*moment_unit_length/3.
    M[9,3] = l*moment_unit_length/6.
    M[3,9] = l*moment_unit_length/6.
    M[9,9] = l*moment_unit_length/3.
   
    M[1,1] = l*mass_unit_length/420*156.
    M[1,7] = l*mass_unit_length/420*54.
    M[7,1] = l*mass_unit_length/420*54.
    M[7,7] = l*mass_unit_length/420*156.

    M[2,2] = l*mass_unit_length/420*156.
    M[2,8] = l*mass_unit_length/420*54.
    M[8,2] = l*mass_unit_length/420*54.
    M[8,8] = l*mass_unit_length/420*156.

    M[1,5]  =  l*l*mass_unit_length/420*22.    
    M[1,11] = -l*l*mass_unit_length/420*13    
    M[7,5]  =  l*l*mass_unit_length/420*13.    
    M[7,11] = -l*l*mass_unit_length/420*22.  

    M[5,1]  =  l*l*mass_unit_length/420*22.    
    M[11,1] = -l*l*mass_unit_length/420*13.  
    M[5,7]  =  l*l*mass_unit_length/420*13.  
    M[11,7] = -l*l*mass_unit_length/420*22.    

    M[5,5]  =   4.*l*l*l*mass_unit_length/420    
    M[5,11] =  -3.*l*l*l*mass_unit_length/420  
    M[11,5]  = -3.*l*l*l*mass_unit_length/420    
    M[11,11] =  4.*l*l*l*mass_unit_length/420    

    M[2,4]  = -l*l*mass_unit_length/420*22.    
    M[2,10] =  l*l*mass_unit_length/420*13    
    M[8,4]  = -l*l*mass_unit_length/420*13.    
    M[8,10] =  l*l*mass_unit_length/420*22.    

    M[4,2]  = -l*l*mass_unit_length/420*22.  
    M[4,8]  = -l*l*mass_unit_length/420*13.    
    M[10,2] =  l*l*mass_unit_length/420*13.
    M[10,8] =  l*l*mass_unit_length/420*22.  

    M[4,4]  =   4.*l*l*l*mass_unit_length/420    
    M[4,10] =  -3.*l*l*l*mass_unit_length/420  
    M[10,4]  = -3.*l*l*l*mass_unit_length/420  
    M[10,10] =  4.*l*l*l*mass_unit_length/420    
   
    M[3,2] = 7./20.*ycg*mass_unit_length*l
    M[2,3] = M[3,2]
    M[3,8] = 3./20.*ycg*mass_unit_length*l
    M[8,3] = M[3,8]
    M[9,2] = 3./20.*ycg*mass_unit_length*l
    M[2,9] = M[9,2]
    M[9,8] = 7./20.*ycg*mass_unit_length*l
    M[8,9] = M[9,8]
   
    M[3,4] = -1/20*ycg*mass_unit_length*l*l
    M[4,3] = M[3,4]
    M[3,10] = 1./30*ycg*mass_unit_length*l*l
    M[10,3] = M[3,10]
    M[9,4] = -1./30*ycg*mass_unit_length*l*l
    M[4,9] = M[9,4]
    M[9,10] = 1./20*ycg*mass_unit_length*l*l
    M[10,9] = M[9,10]

    return M, K

def PAZY_struct(theta_x,theta_y,theta_z): # assemblma matrice di massa e rigidezza per n elementi (n_el) lungo l'ala
   
    Mdummy = np.zeros((n_dof_wing,n_dof_wing))
    Kdummy = np.zeros((n_dof_wing,n_dof_wing))
    #C = np.zeros((n_dof_wing,n_dof_wing))

    # wing

    for i_el in range(0,n_el): # loop per ciascun elemento
        i1 = i_el*n_dof
        i2 = i1 + 2*n_dof
        
        EA_el = 0.50*(EA[i_el] + EA[i_el+1])
        EIy_el = 0.50*(EIy[i_el] + EIy[i_el+1])
        EIz_el = 0.50*(EIz[i_el] + EIz[i_el+1])
        GJ_el = 0.50*(GJ[i_el] + GJ[i_el+1])
        mass_unit_length_el = 0.50*(mass_unit_length[i_el] + mass_unit_length[i_el+1])
        moment_unit_length_el = 0.50*(moment_unit_length[i_el] + moment_unit_length[i_el+1])
        ycg_el = 0.50*(ycg[i_el] + ycg[i_el+1])
        ysc_el = 0.50*(ysc[i_el] + ysc[i_el+1])
       
        M_el, K_el = beam(EA_el, EIy_el, EIz_el, l, GJ_el, mass_unit_length_el, moment_unit_length_el,ycg_el,ysc_el)
        
        # dato che ho grandi deform effettuo rotazione delle matrici 
        # rotation around Y axis
       
        Rot_x  = np.zeros((12,12))
        Rot_y  = np.zeros((12,12))
        Rot_z  = np.zeros((12,12))
       
        # from x_absolute -> x_element
       
        Rx1 =np.array([[1., 0., 0.],[0., np.cos(theta_x[i_el]), np.sin(theta_x[i_el])],[0., -np.sin(theta_x[i_el]), np.cos(theta_x[i_el])]])
        #Ry1 =np.array([[np.cos(theta_y[i_el]), 0., -np.sin(theta_y[i_el])], [0., 1., 0.], [np.sin(theta_y[i_el]), 0., np.cos(theta_y[i_el])]])
        #Rz1 =np.array([[np.cos(theta_z[i_el]), np.sin(theta_z[i_el]), 0.], [-np.sin(theta_z[i_el]), np.cos(theta_z[i_el]), 0.],[0., 0., 1.]])
        Ry1 =np.array([[np.cos(theta_y[i_el]), 0.,-np.sin(theta_y[i_el])], [0., 1., 0.], [np.sin(theta_y[i_el]), 0., np.cos(theta_y[i_el])]])
        Rz1 =np.array([[np.cos(theta_z[i_el]), np.sin(theta_z[i_el]), 0.], [-np.sin(theta_z[i_el]), np.cos(theta_z[i_el]), 0.],[0., 0., 1.]])

        Rx2 =np.array([[1., 0., 0.],[0., np.cos(theta_x[i_el+1]), np.sin(theta_x[i_el+1])],[0., -np.sin(theta_x[i_el+1]), np.cos(theta_x[i_el+1])]])
        Ry2 =np.array([[np.cos(theta_y[i_el+1]), 0., -np.sin(theta_y[i_el+1])], [0., 1., 0.], [np.sin(theta_y[i_el+1]), 0., np.cos(theta_y[i_el+1])]])
        Rz2 =np.array([[np.cos(theta_z[i_el+1]), np.sin(theta_z)[i_el+1], 0.], [-np.sin(theta_z[i_el+1]), np.cos(theta_z[i_el+1]), 0.],[0., 0., 1.]])
        
        tx = (theta_x[i_el] + theta_x[i_el+1])*0.5
        ty = (theta_y[i_el] + theta_y[i_el+1])*0.5
        tz = (theta_z[i_el] + theta_z[i_el+1])*0.5

        cx = np.cos(tx)
        cy = np.cos(ty)
        cz = np.cos(tz)
        sx = np.sin(tx)
        sy = np.sin(ty)
        sz = np.sin(tz)

        Rx1 = np.array([[1., 0., 0.],[0, cx, sx],[0, -sx,cx]])
        Ry1 = np.array([[cy, 0, -sy],[0.,  1., 0.],[sy, 0., cy]])
        Rz1 = np.array([[cz, sz, 0.],[ -sz, cz,0.],[0., 0., 1.]])
        Rx2 = Rx1
        Ry2 = Ry1
        Rz2 = Rz1
        #Ry2 =np.array([[np.cos(theta_y[i_el+1]), 0., -np.sin(theta_y[i_el+1])], [0., 1., 0.], [np.sin(theta_y[i_el+1]), 0., np.cos(theta_y[i_el+1])]])
        #Rz2 =np.array([[np.cos(theta_z[i_el+1]), np.sin(theta_z)[i_el+1], 0.], [-np.sin(theta_z[i_el+1]), np.cos(theta_z[i_el+1]), 0.],[0., 0., 1.]])

        Rot_x[0:3,0:3] = Rx1
        Rot_x[3:6,3:6] = Rx1
        Rot_x[6:9,6:9] = Rx2
        Rot_x[9:12,9:12] = Rx2

        Rot_y[0:3,0:3] = Ry1
        Rot_y[3:6,3:6] = Ry1
        Rot_y[6:9,6:9] = Ry2
        Rot_y[9:12,9:12] = Ry2

        Rot_z[0:3,0:3] = Rz1
        Rot_z[3:6,3:6] = Rz1
        Rot_z[6:9,6:9] = Rz2
        Rot_z[9:12,9:12] = Rz2

        Rot = Rot_z@Rot_y@Rot_x
        #Rot = Rot_y@Rot_x@Rot_z
        #Rot = Rot_y
       
        Mdummy[i1:i2,i1:i2] += Rot.T@M_el@Rot
        Kdummy[i1:i2,i1:i2] += Rot.T@K_el@Rot

        #Mdummy[i1:i2,i1:i2] += Rot@M_el@Rot.T
        #Kdummy[i1:i2,i1:i2] += Rot@K_el@Rot.T
        
        #Mdummy[i1:i2,i1:i2] += Rot_y.T@M_el@Rot_y
        #Kdummy[i1:i2,i1:i2] += Rot_y.T@K_el@Rot_y



    # tolgo i primi 6 DOF per incastrare ala alla radice
    # apply constraint

    M_aa_beam = Mdummy[n_dof:n_dof_wing+1,n_dof:n_dof_wing+1]
    K_aa_beam = Kdummy[n_dof:n_dof_wing+1,n_dof:n_dof_wing+1]
    C_aa_beam = xi*np.sqrt((np.diag(np.diag(M_aa_beam))*np.diag(np.diag(K_aa_beam))))

    return M_aa_beam, C_aa_beam, K_aa_beam

def k_geom(fyA,fyB,fzA,fzB): # correzione matrice di rigidezza nel caso di grandi spostamenti
    K = np.zeros((12,12))
    
    K[3,1] = fzB 
    K[3,2] = -fyB 
    K[9,7] = fzA 
    K[9,8] = -fyA 
    
    return K

def PAZY_followerf(lift,drag,theta_x,theta_y,theta_z): # ruoto matrice di rigidezza per ciascun elemento in modo che sia riferita non in coordinate globali ma locali rispetto a ciascun elemento
    Kdummy = np.zeros((n_dof_wing,n_dof_wing))

    for i_el in range(0,n_el):
        i1 = i_el*n_dof
        i2 = i1 + 2*n_dof
        
        K_geom = k_geom(drag[i_el],drag[i_el+1],lift[i_el],lift[i_el+1])
        
        Rot_x  = np.zeros((12,12))
        Rot_y  = np.zeros((12,12))
        Rot_z  = np.zeros((12,12))
       
        # from x_absolute -> x_element
       
        Rx1 =np.array([[1., 0., 0.],[0., np.cos(theta_x[i_el]), np.sin(theta_x[i_el])],[0., -np.sin(theta_x[i_el]), np.cos(theta_x[i_el])]])
        #Ry1 =np.array([[np.cos(theta_y[i_el]), 0., -np.sin(theta_y[i_el])], [0., 1., 0.], [np.sin(theta_y[i_el]), 0., np.cos(theta_y[i_el])]])
        #Rz1 =np.array([[np.cos(theta_z[i_el]), np.sin(theta_z[i_el]), 0.], [-np.sin(theta_z[i_el]), np.cos(theta_z[i_el]), 0.],[0., 0., 1.]])
        Ry1 =np.array([[np.cos(theta_y[i_el]), 0.,-np.sin(theta_y[i_el])], [0., 1., 0.], [np.sin(theta_y[i_el]), 0., np.cos(theta_y[i_el])]])
        Rz1 =np.array([[np.cos(theta_z[i_el]), np.sin(theta_z[i_el]), 0.], [-np.sin(theta_z[i_el]), np.cos(theta_z[i_el]), 0.],[0., 0., 1.]])

        Rx2 =np.array([[1., 0., 0.],[0., np.cos(theta_x[i_el+1]), np.sin(theta_x[i_el+1])],[0., -np.sin(theta_x[i_el+1]), np.cos(theta_x[i_el+1])]])
        Ry2 =np.array([[np.cos(theta_y[i_el+1]), 0., -np.sin(theta_y[i_el+1])], [0., 1., 0.], [np.sin(theta_y[i_el+1]), 0., np.cos(theta_y[i_el+1])]])
        Rz2 =np.array([[np.cos(theta_z[i_el+1]), np.sin(theta_z)[i_el+1], 0.], [-np.sin(theta_z[i_el+1]), np.cos(theta_z[i_el+1]), 0.],[0., 0., 1.]])
        
        tx = (theta_x[i_el] + theta_x[i_el+1])*0.5
        ty = (theta_y[i_el] + theta_y[i_el+1])*0.5
        tz = (theta_z[i_el] + theta_z[i_el+1])*0.5

        cx = np.cos(tx)
        cy = np.cos(ty)
        cz = np.cos(tz)
        sx = np.sin(tx)
        sy = np.sin(ty)
        sz = np.sin(tz)

        Rx1 = np.array([[1., 0., 0.],[0, cx, sx],[0, -sx,cx]])
        Ry1 = np.array([[cy, 0, -sy],[0.,  1., 0.],[sy, 0., cy]])
        Rz1 = np.array([[cz, sz, 0.],[ -sz, cz,0.],[0., 0., 1.]])
        Rx2 = Rx1
        Ry2 = Ry1
        Rz2 = Rz1
        #Ry2 =np.array([[np.cos(theta_y[i_el+1]), 0., -np.sin(theta_y[i_el+1])], [0., 1., 0.], [np.sin(theta_y[i_el+1]), 0., np.cos(theta_y[i_el+1])]])
        #Rz2 =np.array([[np.cos(theta_z[i_el+1]), np.sin(theta_z)[i_el+1], 0.], [-np.sin(theta_z[i_el+1]), np.cos(theta_z[i_el+1]), 0.],[0., 0., 1.]])

        Rot_x[0:3,0:3] = Rx1
        Rot_x[3:6,3:6] = Rx1
        Rot_x[6:9,6:9] = Rx2
        Rot_x[9:12,9:12] = Rx2

        Rot_y[0:3,0:3] = Ry1
        Rot_y[3:6,3:6] = Ry1
        Rot_y[6:9,6:9] = Ry2
        Rot_y[9:12,9:12] = Ry2

        Rot_z[0:3,0:3] = Rz1
        Rot_z[3:6,3:6] = Rz1
        Rot_z[6:9,6:9] = Rz2
        Rot_z[9:12,9:12] = Rz2

        Rot = Rot_z@Rot_y@Rot_x
        #Rot = Rot_y@Rot_x@Rot_z
        #Rot = Rot_y
       
        Kdummy[i1:i2,i1:i2] += Rot.T@K_geom@Rot
        
    K_g = Kdummy[n_dof:n_dof_wing+1,n_dof:n_dof_wing+1]
    
    return K_g

def neural_network(rey,alpha0,deflection_i):
    version_model = 'FCNN_v2'
    model_dir = 'Model_optimized\\'+version_model+'\\'
    model = tf.keras.models.load_model(model_dir+ 'neural_network_model.h5')
    model.load_weights(model_dir+ 'neural_network_weights.h5')

    # Import scalers for input and output of NN
    with open('Dataset_generator\\scaler_input.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('Dataset_generator\\scaler_output.pkl', 'rb') as f:
        scalery = pickle.load(f)  

    Re_i = np.full((1, 1), rey)
    AoA_i = np.full((1, 1), alpha0)
    deflection = np.full((1, 1), deflection_i)

    input = np.concatenate((Re_i, AoA_i, deflection), axis=1)   

    X_exp = scaler.transform(input)

    Y_exp = model.predict(X_exp, batch_size = 1,verbose=0)

    Y_exp = np.squeeze(scalery.inverse_transform(Y_exp))

    # normalize the lift distribution in order to have 1 at root and 0 at tip
    # lift3d_1 =  (Y_exp - np.min(Y_exp)) / (np.max(Y_exp) - np.min(Y_exp))
    lift3d_1 = Y_exp / Y_exp[0]

    return lift3d_1


# CFD DATA 

aoa_da = np.zeros((18,4))
cl_000 = np.zeros((18,4))
cl_min = np.zeros((18,4))
cl_max = np.zeros((18,4))
cd_000 = np.zeros((18,4))
cd_min = np.zeros((18,4))
cd_max = np.zeros((18,4))

# Hp: centro aerodinamico rimane fisso al 25%
data = np.genfromtxt('Dataset_Marcello\\polar_uq_Re2e5.txt')
aoa_da[:,0] = data[:,0]*np.pi/180. 
cl_000[:,0] = data[:,1]
cl_min[:,0]  = np.min(data[:,2:7],1)
cl_max[:,0]  = np.max(data[:,2:7],1)
cd_000[:,0] = data[:,1+6]
cd_min[:,0]  = np.min(data[:,2+6:7+6],1)
cd_max[:,0]  = np.max(data[:,2+6:7+6],1)

data = np.genfromtxt('Dataset_Marcello\\polar_uq_Re3e5.txt')
aoa_da[:,1] = data[:,0]*np.pi/180. 
cl_000[:,1] = data[:,1]
cl_min[:,1]  = np.min(data[:,2:7],1)
cl_max[:,1]  = np.max(data[:,2:7],1)
cd_000[:,1] = data[:,1+6]
cd_min[:,1]  = np.min(data[:,2+6:7+6],1)
cd_max[:,1]  = np.max(data[:,2+6:7+6],1)

data = np.genfromtxt('Dataset_Marcello\\polar_uq_Re4e5.txt')
aoa_da[:,2] = data[:,0]*np.pi/180. 
cl_000[:,2] = data[:,1]
cl_min[:,2]  = np.min(data[:,2:7],1)
cl_max[:,2]  = np.max(data[:,2:7],1)
cd_000[:,2] = data[:,1+6]
cd_min[:,2]  = np.min(data[:,2+6:7+6],1)
cd_max[:,2]  = np.max(data[:,2+6:7+6],1)

data = np.genfromtxt('Dataset_Marcello\\polar_uq_Re5e5.txt')
aoa_da[:,3] = data[:,0]*np.pi/180. 
cl_000[:,3] = data[:,1]
cl_min[:,3]  = np.min(data[:,2:7],1)
cl_max[:,3]  = np.max(data[:,2:7],1)
cd_000[:,3] = data[:,1+6]
cd_min[:,3]  = np.min(data[:,2+6:7+6],1)
cd_max[:,3]  = np.max(data[:,2+6:7+6],1)

    

span=0.55
n_el=16
l = span/n_el

n_points = n_el + 1

# grid
x_wing = np.linspace(0.,span,n_points)
y_wing = np.zeros(n_points)
z_wing = np.zeros(n_points)

n_dof=6
n_dof_tot = n_dof*n_points
n_dof_wing = n_dof_tot

# FEM DATA
# span coordinates 
x_coarse = np.array([0.,   0.0382,   0.0765,   0.1147,   0.153,   0.1912,   0.2295,   0.2677,   0.306,   0.3442,   0.3825,   0.4207,   0.459,   0.4972,   0.5307, 0.55])

GJ_coarse = np.array([7.2801,   7.2801,   7.2801,   7.2801,   6.4468,   6.4468,   6.4468,   6.4468,   6.4508,   6.4508,   6.4508,   6.4508,   7.0484,   7.0484,   17.1614, 17.1614])


# data from FEM
EIy_coarse = np.array([4.5919, 4.4474,4.4276,4.4243,4.4246,4.4246,4.4247,4.4246,4.4246,4.4246,4.4244,4.4258,4.4387,4.4985,4.7103,4.7183])
EIz_coarse = 0.75*np.array([3318.2,3274.5,3279.6,3283.8,3286.7,3288.4,3289.2,3289.4,3288.8,3287.4,3285.0,3281.3,3276.3,3268.3,3371.1,3371.1])

EIy_coarse = EIy_coarse * (2/2.5)**2
GJ_coarse = GJ_coarse * (2/2.5)**2.5

# Strip theory
cla = 1.2*2*np.pi*np.ones(n_points) #1.4 1.3 1.15 # CL_alpha
x_lift3d = span*np.array([0., 0.052, 0.105, 0.156, 0.208, 0.259, 0.309,  0.358,  0.407, 0.454, 0.5, 0.545, 0.588, 0.629, 0.669, 0.707,   0.743,   0.777,   0.809, 0.839, 0.866, 0.891, 0.914,  0.934,  0.951,  0.966,   0.978,   0.988,   0.995,   0.999,   1.])


chord = 0.10*np.ones(n_points)
e = 0.00*chord #-0.025 -0.05
yac = 0.209*chord #0.209*chord ???????????
mass_unit_length = 1.40*0.620*np.ones(n_points) # 1.20 #0.575 0.600 0.550 0.875 705 695 605 625 555 575 1.1*1.25*575 600 625
moment_unit_length = 1.00*0.0008*np.ones(n_points) # 1.10 #0.00125 0.035 24 0.00175 0.00275 225 195 215 195 185 0.45
# ycg =  -0.00*chord # 0.04-0.10 0.050 -0.075
# tip_mass=0.02 #0.00 0.016 12
# tip_j = 12.e-5 #5e-5 2e-5 12.e-5
# ytip_mass = -0.50*chord[-1] #-0.35 0.2
area_aero = chord*l
area_aero[1] = chord[0]*1.25*l
area_aero[-1] = chord[-1]*l*0.5


EA = 2e5*np.ones(n_points)
EIy = 1.18*np.interp(x_wing,x_coarse,EIy_coarse) # 1.1
EIz = np.interp(x_wing,x_coarse,EIz_coarse)
GJ = 1.10*np.interp(x_wing,x_coarse,GJ_coarse) # 1.1


ycg = 0.0*chord*np.ones(n_points)
ysc = 0.0*chord*np.ones(n_points)

twist = np.zeros(n_points)
out_of_plane = np.zeros(n_points)
in_plane = np.zeros(n_points)

# structural damping

xi =0.000

# constraints applied in PAZY_struct

n_dof_wing_aa = n_dof_wing - n_dof

x_ac=np.zeros(2*n_points-1)
x_ac[0:n_points] = x_wing
x_ac[n_points:2*n_points-1] = -x_wing[1:n_points]



# indici per ciascun punto lungo X, Y, Z per poi studiare displacement lungo i 3 assi
# indices

i_x = list(range(0,n_points*n_dof,n_dof))
i_y = list(range(1,n_points*n_dof+1,n_dof))
i_z = list(range(2,n_points*n_dof+2,n_dof))
i_rx = list(range(3,n_points*n_dof+3,n_dof))
i_ry = list(range(4,n_points*n_dof+4,n_dof))
i_rz = list(range(5,n_points*n_dof+5,n_dof))

i_x_aa = list(range(0,(n_points-1)*n_dof,n_dof))
i_y_aa = list(range(1,(n_points-1)*n_dof+1,n_dof))
i_z_aa = list(range(2,(n_points-1)*n_dof+2,n_dof))
i_rx_aa = list(range(3,(n_points-1)*n_dof+3,n_dof))
i_ry_aa = list(range(4,(n_points-1)*n_dof+4,n_dof))
i_rz_aa = list(range(5,(n_points-1)*n_dof+5,n_dof))


# Flow conditions

# air speed

v_min = 10.
v_max = 60.
n_vel = int(v_max-v_min)
vel = np.linspace(v_min,v_max,n_vel)

V = vel[-1]
rho = 1.225
viscosity = 1.65e-5 #1.79e-5
# q = 0.5*rho*V*V
# alpha0 = 5.0*np.pi/180.
# cla = 5.8
# cma = cla*yac/chord
# cd = 0.01

AoA_input = np.arange(0.0,11.0,1.0)
# Re_i = np.arange(2e5,4e5,(4e5-2e5)/20)

# Define variables to store the flutter onset and offset velocities
q_flutter_onset = []
q_flutter_offset = []

# Loop through each angle of attack 
for alpha0 in AoA_input: 
    # Create empty lists to store the flutter onset and offset velocities for this angle of attack
    q_flutter_onset_aoa = []
    q_flutter_offset_aoa = []
    q_flutter_onset_aoa_temp = []

    print('Input AoA: '+str(alpha0))
    # # aeroloads
    ### Calcolo deformazione statica nonlineare ###

    nloadsteps = 72
    beta = 1/nloadsteps

    # ####
    # disp_aa = np.zeros((n_points-1)*n_dof)
    # disp = np.zeros(n_points*n_dof)

    # loads_aa = np.zeros((n_points-1)*n_dof)

    # loads = np.zeros((n_points)*n_dof)


    # fig, ax = plt.subplots()

    # for istep in range(1,nloadsteps+1): 

    #     rey = chord[0]*V/viscosity*1e-5 # Reynolds number

    #     lift3d_1 = neural_network(rey,alpha0,0.0)

    #     # print(lift3d_1.shape)  
    #     x_lift3d = np.arange(0.55/16,0.55+0.55/16,0.55/16)

    #     lift3d = np.interp(x_wing,x_lift3d,lift3d_1) # distribuzione CL lungo lo span usando funzione empirica, dovrebbe essere sostituito con CFD e neural network

    #     # calcolo carichi aerodinamici
    #     alpha = disp[i_rx] + alpha0*np.pi/180. *np.cos(disp[i_ry])# da aggiungere un coseno della rotazione lungo x per tenere conto della deflessione lungo lo span #### rotazioni ad x di ogni punto + AoA alla radice 
    #     loads[i_x] = q*area_aero*alpha*cla*lift3d*np.sin(disp[i_ry]) #lift3d forza lungo lo span secondo una funzione empirica (da sosoturier con CFD)
    #     loads[i_z] = q*area_aero*alpha*cla*lift3d*np.cos(disp[i_ry])
    #     loads[i_rx] = q*area_aero*alpha*cla*yac #cma
    #     loads_aa = loads[n_dof:]*beta # beta utile a frazionare il carico 
    #     # calcolo matrici strutturali
    #     M_aa, C_aa, K_aa = PAZY_struct(disp[i_rx],disp[i_ry],disp[i_rz])

    #     # calcolo deformate
    #     disp_aa = linalg.inv(K_aa)@loads_aa
    #     disp[n_dof:] += disp_aa # ho il loop per nloadsteps in quanto applico carico per piccoli passi ed aggiorno ogni volta la matrice di rigidezza tenendo in considerazione i carichi vecchi (in questo modo tengo conto della nonlinearita)
    #     ax.plot((x_wing+disp[i_x])/span,disp[i_z]/span, 'b-', linewidth=1,alpha=0.2)

    # ax.plot((x_wing+disp[i_x])/span,disp[i_z]/span, 'k-', linewidth=2)
    # ax.set_xlabel(r'$x/s \,\, [\,]$')
    # ax.set_ylabel(r'$z/s \,\, [\,]$')
    # ax.axis('equal')
    # plt.grid(True)
    # plt.show()


    # eigenvalues_normal, R = linalg.eig(K_aa,M_aa) # usando le matrici dell'ultima iterazione calcolo i modi (in quanto l'utlima iterazione e' quella che tiene conto del carico completo che prima abbiamo applicato per piccoli passi)
    # indices_modes = np.argsort(np.real(eigenvalues_normal))       



    # fig, ax = plt.subplots(3,1)
    # ax[0].plot(x_wing+disp[i_x],disp[i_z], 'ko-', label='Z')
    # ax[0].set_xlabel(r'$x \,\, [m]$')
    # ax[0].set_ylabel(r'$z \,\, [m]$')
    # ax[1].plot(x_wing+disp[i_x],disp[i_y], 'ko-', label='Y')
    # ax[1].set_xlabel(r'$x \,\, [m]$')
    # ax[1].set_ylabel(r'$y \,\, [m]$')
    # ax[2].plot(x_wing+disp[i_x],180/np.pi*disp[i_rx], 'ko-', label='Y')
    # ax[2].set_xlabel(r'$x \,\, [m]$')
    # ax[2].set_ylabel(r'$\theta \,\, [DEG]$')
    # plt.show()

    # # deformata per modo flessionale
    # fig, ax = plt.subplots()
    # ax.plot(x_wing+disp[i_x],disp[i_z], 'k-', label='OUT-OF-PLANE')
    # ax.plot(x_wing+disp[i_x],disp[i_y], 'r-', label='IN-PLANE')
    # ax.set_xlabel(r'$x \,\, [m]$')
    # ax.set_ylabel(r'$z \,\, [m]$')
    # ax.axis('equal')
    # ax.legend()
    # plt.show()


    ######


    # Inizia la parte dinamica
    # # 1st order system

    n_dof_aa = (n_points-1)*n_dof
    n_dof_alags = (n_points-1)*2 # 2 lag state per ciascun punto 
    n_dof_phys = 2*n_dof_aa
    n_dof_tot = 2*n_dof_aa + n_dof_alags # ai DOF strutturali aggiungo 2 lag states aerodinamici per ciascun punto 

    # Jones coeffs
    A1 = 0.335
    A2 = 0.165

    A0 = 1. - A1 - A2
    b1 = 0.30
    b2 = 0.0455

    # distance between elastic axis and 3/4 chord (in chords)
    dist_aoa_eff = 0.209

    ff = np.zeros((n_dof_tot,n_vel))
    gg = np.zeros((n_dof_tot,n_vel))
    axial = np.zeros((n_points,n_vel))
    ooplane = np.zeros((n_points,n_vel)) # flessione
    twist = np.zeros((n_points,n_vel))   # torsione


    for i_vel in range(n_vel):
        V = vel[i_vel]
        q = 0.5*rho*V*V
        
        rey = chord[0]*V/viscosity*1e-5 # Reynolds number

        # Predict lift distribution for a given AoA and velocity with no deflection
        lift3d_1 = neural_network(rey,alpha0,0.0)
        x_lift3d = np.arange(0.55/16,0.55+0.55/16,0.55/16) # coordinates along the span
        lift3d = np.interp(x_wing,x_lift3d,lift3d_1)
        
        # this and the following lines are for interpolating between Reynolds number data 
        i_rey_1 = np.minimum( np.maximum(2,int(rey)), 5)
        i_rey_2 = np.maximum( np.minimum(int(rey)+1,5), 2)
        
        weight_2 = np.maximum(i_rey_2 - rey,0.)
        weight_1 = np.maximum(rey - i_rey_1,0.)
        
        i_w_1 = weight_1/(weight_1 + weight_2)
        i_w_2 = 1. - i_w_1 
        
        alpha_coarse = aoa_da[:,i_rey_1-2]
        
        # b/l
        # cl_coarse = i_w_1*cl_000[:,i_rey_1-2] + i_w_2*cl_000[:,i_rey_2-2]
        # cd_coarse = i_w_1*cd_000[:,i_rey_1-2] + i_w_2*cd_000[:,i_rey_2-2]
        cl_coarse = cl_000[:,3] 
        cd_coarse = cd_000[:,3] 
    

        cla_coarse = np.gradient(cl_coarse,alpha_coarse) # calcolo CL_alpha
            
                
        ### Calcolo deformazione statica nonlineare ### (di nuovo)
        disp_aa = np.zeros((n_points-1)*n_dof)
        disp = np.zeros(n_points*n_dof)
        loads_aa = np.zeros((n_points-1)*n_dof)
        loads = np.zeros((n_points)*n_dof)

        for istep in range(1,nloadsteps+1):
            alpha = disp[i_rx] + alpha0*np.pi/180.*np.cos(disp[i_ry])
            # CFD DATA 
            cl_2d = np.interp(alpha,alpha_coarse,cl_coarse)
            cla_2d = np.interp(alpha,alpha_coarse,cla_coarse)
            cd_2d = np.interp(alpha,alpha_coarse,cd_coarse)
            
            loads[i_x] = q*area_aero*cl_2d*lift3d*np.sin(disp[i_ry])
            loads[i_y] = q*area_aero*cd_2d
            loads[i_z] = q*area_aero*cl_2d*lift3d*np.cos(disp[i_ry])
            loads[i_rx] = q*area_aero*cl_2d*yac*np.cos(disp[i_ry]) #cma
            loads[i_rz] = -q*area_aero*cl_2d*yac*np.sin(disp[i_ry]) #cma
            loads_aa = loads[n_dof:]*beta
            
            M_aa, C_aa, K_aa = PAZY_struct(disp[i_rx],disp[i_ry],disp[i_rz]) # calcolo matrici per ciascun punto lungo lo span
            K_g = PAZY_followerf(loads[i_z], loads[i_y], disp[i_rx],disp[i_ry],disp[i_rz]) # ruoto matrice di rigidezza per passare da SR globale a locale del punto

            disp_aa = linalg.inv(K_aa)@loads_aa
            disp[n_dof:] += disp_aa # applico carico per piccoli passi ed aggiorno ogni volta la matrice di rigidezza (in questo modo tengo conto della nonlinearita)
        
        # assegno spostamenti lungo X, Y, Z
        axial[:,i_vel] = disp[i_x]  
        ooplane[:,i_vel] = disp[i_z]  
        twist[:,i_vel] = disp[i_rx]  
        
        # Update lift distribution with tip deflection
        lift3d_1 = neural_network(rey,alpha0,disp[i_z][-1])

        lift3d = np.interp(x_wing,x_lift3d,lift3d_1) # distribuzione CL lungo lo span usando funzione empirica, dovrebbe essere sostituito con CFD e neural network

        # Introduco matrici per la risposta dinamica

        # no follower forces
        #Aphys = np.vstack([np.hstack([np.zeros((n_dof_aa,n_dof_aa)), np.eye(n_dof_aa)]), np.hstack([-K_aa, -C_aa])])

        # follower forces
        Aphys = np.vstack([np.hstack([np.zeros((n_dof_aa,n_dof_aa)), np.eye(n_dof_aa)]), np.hstack([-(K_aa+K_g), -C_aa])])
        Bphys = np.vstack([np.hstack([np.eye(n_dof_aa), np.zeros((n_dof_aa,n_dof_aa))]), np.hstack([np.zeros((n_dof_aa,n_dof_aa)), M_aa])])

        # matrici A e B contengono anche l'aerodinamica (equazione 17 in paper STATE-SPACE REPRESENTATION OF UNSTEADY AIRFOIL BEHAVIOR)
        A = np.zeros((n_dof_tot,n_dof_tot))
        B = np.zeros((n_dof_tot,n_dof_tot))

        A[0:n_dof_phys,0:n_dof_phys] = Aphys
        B[0:n_dof_phys,0:n_dof_phys] = Bphys

        B[n_dof_phys:n_dof_tot,n_dof_phys:n_dof_tot] = np.eye(n_dof_alags)
        
        for i_point in range(1,n_points): # costruisco le matrici per gli n_el lungo lo span
            i_theta = i_rx_aa[i_point-1]
            i_theta_y = i_ry_aa[i_point-1]
            i_span = i_x_aa[i_point - 1]
            i_zeta = i_z_aa[i_point-1]
            i_zetadot = i_zeta + n_dof_aa
            i_thetadot = i_theta + n_dof_aa
            i_theta_y_dot = i_theta_y + n_dof_aa
            i_spandot = i_span + n_dof_aa
            i_alag1 = n_dof_phys + (i_point - 1)*2
            i_alag2 = i_alag1 +1
            c = chord[i_point]
            area = area_aero[i_point] # correct first and last point
            # CFD DATA 
            cla = cla_2d[i_point]           
            cma = cla *yac[i_point]/c
            cos_di = np.cos(disp[i_ry[i_point]])
            sin_di = np.sin(disp[i_ry[i_point]])
            
            # adjust b_i coeffs empirically
            # introduciamo i lag states
            A[i_alag1,i_alag1] = -b1*2*V/c
            A[i_alag2,i_alag2] = -b2*2*V/c

            A[i_alag1,i_theta] = 1.
            A[i_alag1,i_zetadot] = -1./V*cos_di
            A[i_alag1,i_spandot] = -1./V*sin_di
            A[i_alag1,i_thetadot] = dist_aoa_eff*c/V # aoa 3/4 c

            A[i_alag2,i_theta] = 1.
            A[i_alag2,i_zetadot] = -1./V*cos_di
            A[i_alag2,i_spandot] = -1./V*sin_di
            A[i_alag2,i_thetadot] = dist_aoa_eff*c/V # aoa 3/4 c

            # Theodorsen for calculating unsteady aerodynamic forces on a lifting surface in harmonic motion
            A[i_zetadot,i_alag1] = q*area*cla*2*V/c*A1*b1*cos_di*lift3d[i_point]  
            A[i_zetadot,i_alag2] = q*area*cla*2*V/c*A2*b2*cos_di*lift3d[i_point]
            A[i_zetadot,i_theta] += q*area*cla*A0*cos_di*lift3d[i_point]
            #
            A[i_zetadot,i_thetadot] += rho*V*area*cla*A0*dist_aoa_eff*c*cos_di*lift3d[i_point] 
            A[i_zetadot,i_zetadot] += -rho*V*area*cla*A0*cos_di*lift3d[i_point]

            A[i_thetadot,i_alag1] = q*c*area*cma*2*V/c*A1*b1*cos_di*lift3d[i_point]
            A[i_thetadot,i_alag2] = q*c*area*cma*2*V/c*A2*b2*cos_di*lift3d[i_point]
            A[i_thetadot,i_theta] += q*c*area*cma*A0*cos_di*lift3d[i_point]
            #
            A[i_thetadot,i_thetadot] += rho*V*c*area*cma*A0*dist_aoa_eff*c*cos_di*lift3d[i_point]
            A[i_thetadot,i_zetadot] += -rho*V*c*area*cma*A0*cos_di*lift3d[i_point]

            A[i_theta_y_dot,i_alag1] = -q*c*area*cma*2*V/c*A1*b1*sin_di*lift3d[i_point]
            A[i_theta_y_dot,i_alag2] = -q*c*area*cma*2*V/c*A2*b2*sin_di*lift3d[i_point]
            A[i_theta_y_dot,i_theta] += -q*c*area*cma*A0*sin_di*lift3d[i_point]
            #
            A[i_theta_y_dot,i_thetadot] += -rho*V*c*area*cma*A0*dist_aoa_eff*c*sin_di*lift3d[i_point]
            A[i_theta_y_dot,i_zetadot] += rho*V*c*area*cma*A0*sin_di*lift3d[i_point]

            A[i_spandot,i_alag1] = q*area*cla*2*V/c*A1*b1*sin_di*lift3d[i_point]
            A[i_spandot,i_alag2] = q*area*cla*2*V/c*A2*b2*sin_di*lift3d[i_point]
            A[i_spandot,i_theta] += q*area*cla*A0*sin_di*lift3d[i_point]
            A[i_spandot,i_thetadot] += rho*V*area*cla*A0*dist_aoa_eff*c*sin_di*lift3d[i_point]
            A[i_spandot,i_zetadot] += -rho*V*area*cla*A0*sin_di*lift3d[i_point]
                
        eigenvalues_complex, R_complex = linalg.eig(linalg.inv(B)@A) # CALCOLO AUTOVALORI
        
        ff[:,i_vel] = eigenvalues_complex.imag #/(2*np.pi)
        gg[:,i_vel] = eigenvalues_complex.real  #/np.abs(eigenvalues_complex)
     

        # # Checks if any of the eigenvalues indicate flutter (damping > 0.01) and returns the flutter speed.   
        # for eigenvalue in eigenvalues_complex:
        #     real_part = eigenvalue.real
        #     imag_part = eigenvalue.imag
        #     damping = -real_part / np.sqrt(real_part**2 + imag_part**2)
        #     if damping > 0.01:
        #         flutter_speed = -real_part / imag_part
        #         q_flutter_onset.append(flutter_speed)
        #         print("Flutter detected! Flutter speeds:", flutter_speed)
        #         break

        # Flutter search with eigenvalue analysis
        def flutter_detected(eigenvalues):
            return any((np.real(val)/ np.sqrt(np.real(val)**2 + np.imag(val)**2)) > 0.004 for val in eigenvalues)

        def flutter_criterion(A):
            eigenvalues, _ = linalg.eig(linalg.inv(B)@A)
            return flutter_detected(eigenvalues)

        # Flutter onset:
        if flutter_criterion(A) and not q_flutter_onset_aoa:
            q_flutter_onset_aoa.append(int(V))

        # Flutter offset:
        elif not flutter_criterion(A) and q_flutter_onset_aoa and not q_flutter_offset_aoa:
            q_flutter_offset_aoa.append(int(V))

        # Check for additional flutter onsets and offsets within velocity range
        elif flutter_criterion(A) and q_flutter_offset_aoa and not q_flutter_onset_aoa:
            q_flutter_offset_aoa = []
            q_flutter_onset_aoa.append(int(V))

        elif not flutter_criterion(A) and q_flutter_onset_aoa and q_flutter_offset_aoa:
            q_flutter_onset_aoa_temp = q_flutter_onset_aoa[0]
            q_flutter_onset_aoa = []
            q_flutter_offset_aoa.append(int(V))
            
    # Append the first flutter onset and last flutter offset to the main lists
    if q_flutter_onset_aoa_temp and q_flutter_offset_aoa:
        q_flutter_onset.append(q_flutter_onset_aoa_temp)
        q_flutter_offset.append(q_flutter_offset_aoa[-1])
        print(f"Angle of attack: {alpha0}, Flutter onset: {q_flutter_onset_aoa_temp}, Flutter offset: {q_flutter_offset_aoa[-1]}")
    elif q_flutter_onset_aoa:
        q_flutter_onset.append(q_flutter_onset_aoa[0])
        q_flutter_offset.append(np.nan)
        print(f"Angle of attack: {alpha0}, Flutter onset: {q_flutter_onset_aoa[0]}")
    elif q_flutter_offset_aoa:
        q_flutter_onset.append(np.nan)
        q_flutter_offset.append(q_flutter_offset_aoa[-1])
        print(f"Angle of attack: {alpha0}, Flutter offset: {q_flutter_offset_aoa[-1]}")
    else:
        q_flutter_onset.append(np.nan)
        q_flutter_offset.append(np.nan)
        print(f"Angle of attack: {alpha0}, Flutter not found!")


    indices_nonzero = np.where(np.abs(eigenvalues_complex.imag) > 0.01 )    

    indices_complex = np.argsort(np.abs(np.imag(eigenvalues_complex[indices_nonzero])))



    # plot for damping and frequency at different velocities
    # in un range di Vel il damping diventa maggiore di zero (ala e' in flutter), in questo caso parlo di hump mode (ala diventa instabile ma poi torna stabile)
    fig, ax = plt.subplots(2,1)
    n_modes = 50

    for i_mode in range(n_modes):
        magni = np.sqrt(gg[indices_complex[i_mode]]**2 + ff[indices_complex[i_mode]]**2)
        ax[0].plot(vel,gg[indices_complex[i_mode]]/magni,'k-',linewidth=1)
        ax[1].plot(vel,ff[indices_complex[i_mode]]/(2*np.pi),'k-',linewidth=1)
        
    ax[0].set_xlabel(r'$v \,\, [m/s]$')
    ax[0].set_ylabel(r'$\xi \,\, [\,]$')
    ax[1].set_xlabel(r'$v \,\, [m/s]$')
    ax[1].set_ylabel(r'$f \,\, [Hz]$')
    ax[0].set_ylim([-0.075,0.05])
    #ax[0].set_ylim([-0.025,0.025])
    ax[1].set_ylim([0,110])
    ax[0].grid(visible=True)
    ax[1].grid(visible=True)
    plt.savefig('Photo\\vg_plot_AoA_'+str(alpha0)+'.png')    

    # plt.show()
    plt.close()

    # # root locus 

    # fig, ax = plt.subplots()
    # n_modes = 50

    # for i_mode in range(n_modes):
    #     ax.plot(gg[indices_complex[i_mode]],ff[indices_complex[i_mode]]/(2*np.pi),'k.',markersize=2)
        
    # ax.set_xlabel(r'$Real(\lambda) \,\, [RAD/s]$')
    # ax.set_ylabel(r'$Imag(\lambda) \,\, [Hz]$')
    # ax.set_xlim([-10,5])
    # ax.set_ylim([0,50])
    # ax.grid(visible=True)
    # plt.savefig('Photo\\root_loci_AoA_'+str(alpha0)+'.png')    
    # # plt.show()


    # fig, ax = plt.subplots()
    # ax.plot(vel,ooplane[-1,:]/span,'o-')
    # ax.set_xlabel(r'$V \,\, [m/s]$')
    # ax.set_ylabel(r'$Disp/span \,\, [\,]$')
    # ax.grid(True)
    # plt.show()       

    # fig, ax = plt.subplots()
    # ax.plot(x_wing+disp[i_x],disp[i_z]/span, 'ko-', label='OUT-OF-PLANE')
    # ax.plot(x_wing+disp[i_x],disp[i_y]/span, 'ko-', label='IN-PLANE')
    # ax.set_xlabel(r'$x \,\, [m]$')
    # ax.set_ylabel(r'$z/b \,\, [\,]$')
    # ax.axis('equal')
    # ax.legend(prop={"size":10},frameon=False)
    # ax.set_xlim([0., 0.60])
    # ax.set_ylim([-0.025,0.50])
    # plt.savefig('Photo\\bend.pdf')    
    # plt.show()

    # fig, ax = plt.subplots()
    # ax.plot(x_wing+disp[i_x],180/np.pi*disp[i_rx], 'ko-', label='TWIST')
    # ax.set_xlabel(r'$x \,\, [m]$')
    # ax.set_ylabel(r'$\theta \,\, [DEG]$')
    # #ax.axis('equal')
    # ax.legend(prop={"size":10},frameon=False)
    # ax.set_xlim([0,0.60])
    # #ax.set_ylim([0,7.25])
    # plt.savefig('Photo\\twist_aoa7_55.pdf')    
    # plt.show()

    # # A[2*n_dof:(2*n_dof+2),2*n_dof:(2*n_dof+2)] = 2*V/chord*beta2*np.diag([-b1, -b2])
    # # A[2*n_dof:(2*n_dof+2),0] = 1.
    # # A[2*n_dof:(2*n_dof+2),1] = 0.
    # # A[2*n_dof:(2*n_dof+2),2] = 0.5*chord/V
    # # A[2*n_dof:(2*n_dof+2),3] = -1./V

    # # B_1 = linalg.inv(B)



    # fig, ax = plt.subplots(2,1)
    # for i_vel in range(n_vel):
    #     ax[0].plot(x_wing+axial[:,i_vel],ooplane[:,i_vel]/span,'k-',linewidth=1,alpha=0.5)
    #     ax[1].plot(x_wing+axial[:,i_vel],180/np.pi*twist[:,i_vel],'k-',linewidth=1,alpha=0.5)

    # ax[0].set_xlabel(r'$x \,\, [m]$')
    # ax[0].set_ylabel(r'$z/s \,\, [\,]$')

    # ax[1].set_xlabel(r'$x \,\, [m]$')
    # ax[1].set_ylabel(r'$\theta \,\, [DEG]$')

    # plt.show()

    # fig, ax = plt.subplots(3,1)
    # ax[0].plot(x_wing[1:],R[i_z_aa,indices_modes[0]], 'k-', label='1')
    # ax[0].plot(x_wing[1:],R[i_z_aa,indices_modes[1]], 'r-', label='2')
    # ax[0].plot(x_wing[1:],R[i_z_aa,indices_modes[2]], 'g-', label='3')
    # ax[0].plot(x_wing[1:],R[i_z_aa,indices_modes[3]], 'b-', label='4')
    # ax[0].plot(x_wing[1:],R[i_z_aa,indices_modes[4]], 'k-.', label='5')
    # ax[0].plot(x_wing[1:],R[i_z_aa,indices_modes[5]], 'r-.', label='6')

    # ax[0].set_xlabel(r'$x \,\, [m]$')
    # ax[0].set_ylabel(r'$z \,\, [m]$')

    # ax[0].legend()

    # ax[1].plot(x_wing[1:],R[i_y_aa,indices_modes[0]], 'k-', label='Z')
    # ax[1].plot(x_wing[1:],R[i_y_aa,indices_modes[1]], 'r-', label='Z')
    # ax[1].plot(x_wing[1:],R[i_y_aa,indices_modes[2]], 'g-', label='Z')
    # ax[1].plot(x_wing[1:],R[i_y_aa,indices_modes[3]], 'b-', label='Z')
    # ax[1].plot(x_wing[1:],R[i_y_aa,indices_modes[4]], 'k-.', label='Z')
    # ax[1].plot(x_wing[1:],R[i_y_aa,indices_modes[5]], 'r-.', label='Z')

    # ax[1].set_xlabel(r'$x \,\, [m]$')
    # ax[1].set_ylabel(r'$y \,\, [m]$')

    # ax[2].plot(x_wing[1:],R[i_rx_aa,indices_modes[0]], 'k-', label='Z')
    # ax[2].plot(x_wing[1:],R[i_rx_aa,indices_modes[1]], 'r-', label='Z')
    # ax[2].plot(x_wing[1:],R[i_rx_aa,indices_modes[2]], 'g-', label='Z')
    # ax[2].plot(x_wing[1:],R[i_rx_aa,indices_modes[3]], 'b-', label='Z')
    # ax[2].plot(x_wing[1:],R[i_rx_aa,indices_modes[4]], 'k-.', label='Z')
    # ax[2].plot(x_wing[1:],R[i_rx_aa,indices_modes[5]], 'r-.', label='Z')

    # ax[2].set_xlabel(r'$x \,\, [m]$')
    # ax[2].set_ylabel(r'$\theta \,\, [DEG]$')
    # plt.show()
    

    # # # time int

    # dt = 1e-3
    # ntsteps = 4000

    # beta = 0.5

    # qq = np.zeros((n_dof_tot,ntsteps))
    # g = np.zeros(n_dof_tot)
    # qq[i_z_aa,0] = 0.01*span*(x_wing[1:]/span)**2

    # BB = B - dt*beta*A #(A + 0*Aaero)
    # BB_1 = linalg.inv(BB)
    # AA = B + dt*(1.-beta)*A # - 0*beta*Aaero)

    # for it in range(1,ntsteps):
    #     qq[:,it] = BB_1@AA@qq[:,it-1]


    # fig, ax = plt.subplots(2,1)
    # ax[0].plot(np.array(range(ntsteps))*dt,qq[i_z_aa[-1]],'k-',linewidth=1)
    # ax[1].plot(np.array(range(ntsteps))*dt,qq[i_rx_aa[-1]],'k-',linewidth=1)
        
    # ax[0].set_xlabel(r'$t \,\, [s]$')
    # ax[0].set_ylabel(r'$z \,\, [\,]$')
    # ax[1].set_xlabel(r'$t \,\, [s]$')
    # ax[1].set_ylabel(r'$\theta \,\, [RAD]$')
    # plt.show()


    # fig, ax = plt.subplots(2,1)
    # ax[0].plot(np.array(range(ntsteps))*dt,qq[i_alag1],'k-',linewidth=1)
    # ax[1].plot(np.array(range(ntsteps))*dt,qq[i_alag2],'k-',linewidth=1)
        
    # ax[0].set_xlabel(r'$t \,\, [s]$')
    # ax[0].set_ylabel(r'$\xi_2 \,\, [s]$')
    # ax[1].set_xlabel(r'$t \,\, [s]$')
    # ax[1].set_ylabel(r'$\xi_2 \,\, [s]$')
    # plt.show()


# # Print the lists of flutter onset and offset velocities for each angle of attack
# print("Flutter onset velocities for each angle of attack:", q_flutter_onset)
# print("Flutter offset velocities for each angle of attack:", q_flutter_offset)



# Save data to csv file
d = {
        'AoA' : AoA_input,
        'Q_flutter_onset': np.array(q_flutter_onset),
        'Q_flutter_offset': np.array(q_flutter_offset)
        }
ds = pd.DataFrame(data=d)


ds.to_csv('q_flutter_onset_offset_AoA.csv', sep=',', index=False)

