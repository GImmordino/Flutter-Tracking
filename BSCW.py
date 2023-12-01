import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from scipy.special import genlaguerre
import pandas as pd
import math
import scipy
import scipy.signal
from matplotlib import cm
from scipy.interpolate import interp1d
from scipy.signal import correlate

# Volterra model - Training and prediction
laguerre_type = 'scipy' # scipy # dowell
volterra_model = True
plot_flag_linear_response = False
plot_flag_NL_response = False  
calculate_flutter = True

# Prediction M = 0.80 - 0 < AoA < 5
prediction_M080_AOA5 = False

# Prediction on flow conditions of flutter exp. data
prediction_exp_data = False

# Prediction 0.74 < M < 0.84 - 0 < AoA < 5
prediction_074M084_0AoA5 = False

# Prediction M = 0.70 -  AoA = 5
prediction_M070_AOA5 = False

# Prediction Harmonic signal M = 0.70 - AoA = 5
prediction_harmonic_signal = False

# Prediction step-responses at constant Mach
prediction_step_responses_constant_Mach = False

prediction_harmonic_responses_constant_Mach = False


#%%  PLOT 3D SURFACE FLUTTER BOUNDARY 0.74 < M < 0.84 - 0 < AoA < 5
# df=pd.read_csv('q_flutter_NN_exp_data_074M084_0AoA5.csv')
# df = df[(df['Mach'] >= 0.74) & (df['Mach'] <= 0.84) &
#                (df['AoA'] >= 0) & (df['AoA'] <= 5)]
# fig = plt.figure()
# fig.set_size_inches(9.25, 5.25)
# ax = plt.axes(projection='3d')
# ax.set_xlabel(r'Mach', labelpad=10)
# ax.set_ylabel(r'AoA')
# ax.set_zlabel(r'Q flutter [psf]')
# surf = ax.plot_trisurf(df.Mach, df.AoA, df.Q_flutter_psf, cmap=cm.viridis, edgecolor='black', linewidth=0.2)
# fig.colorbar(surf,location = 'bottom',fraction=0.046, pad=0.04)
# fig.tight_layout()
# ax.view_init(azim=140, elev=30)
# surf.set_alpha(0.8) # Make trisurf plot transparent
# # ax.scatter(df['Mach'].values.reshape(-1,1), df['AoA'].values.reshape(-1,1), df['Q_flutter_psf'].values.reshape(-1,1),color='black', s=20, alpha=0.5 )
# plt.savefig('Photo\\'+ 'flutter_bscw_volterra.png', bbox_inches='tight', dpi=300)
# plt.show()


#%%  Volterra model - Training and prediction
if volterra_model:

    ### Load dataset

    # Prediction kernels 1 deg
    kernels_CL_linear_pred = np.load('Dataset_generator\\kernels_CL_linear_pred.npy') # samples,timesteps,variables
    kernels_CL_linear_pred_GP = np.load('Dataset_generator\\kernels_CL_linear_pred_GP.npy') # samples,timesteps,variables
    kernels_CM_linear_pred = np.load('Dataset_generator\\kernels_CM_linear_pred.npy') # samples,timesteps,variables
    kernels_CM_linear_pred_GP = np.load('Dataset_generator\\kernels_CM_linear_pred_GP.npy') # samples,timesteps,variables

    # Prediction kernels NL
    kernels_CL_NL_pred = np.load('Dataset_generator\\kernels_CL_NL_pred.npy') # samples,timesteps,variables
    kernels_CL_NL_pred_GP = np.load('Dataset_generator\\kernels_CL_NL_pred_GP.npy') # samples,timesteps,variables
    kernels_CM_NL_pred = np.load('Dataset_generator\\kernels_CM_NL_pred.npy') # samples,timesteps,variables
    kernels_CM_NL_pred_GP = np.load('Dataset_generator\\kernels_CM_NL_pred_GP.npy') # samples,timesteps,variables


    # Variance kernels GP
    variance_kernels_CL_linear_pred_GP = np.load('Dataset_generator\\variance_kernels_CL_linear_pred_GP.npy') # samples,timesteps,variables
    variance_kernels_CM_linear_pred_GP = np.load('Dataset_generator\\variance_kernels_CM_linear_pred_GP.npy') # samples,timesteps,variables
    variance_kernels_CL_NL_pred_GP = np.load('Dataset_generator\\variance_kernels_CL_NL_pred_GP.npy') # samples,timesteps,variables
    variance_kernels_CM_NL_pred_GP = np.load('Dataset_generator\\variance_kernels_CM_NL_pred_GP.npy') # samples,timesteps,variables



    # Dataset CFD
    dataset_CL0_CM0_M_AoA = np.load('Dataset_generator\\dataset_CL0_CM0_M_AoA.npy') # samples,timesteps,variables
    dataset_CL_CM_1_deg = np.load('Dataset_generator\\dataset_CL_CM_1_deg.npy') # samples,timesteps,variables
    dataset_CL_CM_2_deg = np.load('Dataset_generator\\dataset_CL_CM_2_deg.npy') # samples,timesteps,variables
    # dataset_CL0_CM0_M_AoA = dataset_CL0_CM0_M_AoA[:,0,:] 
    dataset_CL0_CM0_M_AoA = np.transpose(dataset_CL0_CM0_M_AoA,(0,2,1))
    dataset_CL_CM_1_deg = np.transpose(dataset_CL_CM_1_deg,(0,2,1))
    dataset_CL_CM_2_deg = np.transpose(dataset_CL_CM_2_deg,(0,2,1))


    print( '\n-> Shape of the input matrix: \n')
    print(dataset_CL0_CM0_M_AoA.shape) # M , AoA , CL0 , CM0
    print( '\n-> Shape of the output matrix at 1 deg and 2 deg: \n')
    print(dataset_CL_CM_1_deg.shape)  # CL unsteady , CM unsteady
    print(dataset_CL_CM_2_deg.shape)  # CL unsteady , CM unsteady

    Mach=dataset_CL0_CM0_M_AoA[:,0,0]
    AoA=dataset_CL0_CM0_M_AoA[:,1,0]
    cl0=dataset_CL0_CM0_M_AoA[:,2,:]
    cm0=dataset_CL0_CM0_M_AoA[:,3,:]

    CL_1_deg = dataset_CL_CM_1_deg[:,0,:]
    CM_1_deg = dataset_CL_CM_1_deg[:,1,:]

    CL_2_deg = dataset_CL_CM_2_deg[:,0,:]
    CM_2_deg = dataset_CL_CM_2_deg[:,1,:]


    ### Define parameters for the model 
    chord=0.4064
    area = 2*chord**2
    x_30 = 0.12192
    x_50 = 0.2032
    cm0 = cm0 + (x_50-x_30) * cl0 / chord
    otref = 500. # tempo fisico = 1 / 500 
    ampli_train_pitch = 1.0*np.pi/180.
    dt=5e-5
    factor=2.0
    n_coeff = 17

    ### Setup prediction range
    # validation_range = range(40,55) # validation range
    # validation_range = range(CL_1_deg.shape[0]) # all dataset
    validation_range = [42,47] # only two points for paper plots 

    kernels_CL = []
    kernels_CM = []
    kernels_CL_NL = []
    kernels_CM_NL = []
    cl_NL_FCNN = []
    cm_NL_FCNN = []
    q_flutter = np.empty(CL_1_deg.shape[0])

    # theta_CL_pred = np.load('Dataset_generator\\theta_CL_pred.npy') # samples,timesteps,variables
    # theta_CM_pred = np.load('Dataset_generator\\theta_CM_pred.npy') # samples,timesteps,variables


    for i in validation_range:

        ###### Linear Kernels ######
        # tau: tempo ridotto
        # t: tempo fisico
        V = Mach[i]*np.sqrt(1.116*81.49*304.2128)

        tau_sampling = 0.6 #0.125#0.5
        ntsteps_sampling = 18 # int(tau_sampling/(dt*2*V/chord))
        dtau = ntsteps_sampling*dt*2*V/chord
        
        it_pitch_1_deg = np.arange(0,CL_1_deg.shape[1],1)
        cl_pitch_1_deg = CL_1_deg[i,:]-cl0[i,:]
        cm_pitch_1_deg = CM_1_deg[i,:]-cm0[i,:]

        # apply a 3-pole lowpass filter at 0.1x Nyquist frequency
        b, a = scipy.signal.butter(1, 0.02)
        b1, a1 = scipy.signal.butter(2, 0.007)
        cl_pitch_1_deg = scipy.signal.filtfilt(b, a, cl_pitch_1_deg)
        cm_pitch_1_deg = scipy.signal.filtfilt(b1, a1, cm_pitch_1_deg)

        tau = it_pitch_1_deg*dt*2*V/chord
        time_1_deg = it_pitch_1_deg*dt

        it_train_1_deg = it_pitch_1_deg[0::ntsteps_sampling] # it: numero di iterazioni
        cl_train_1_deg = cl_pitch_1_deg[0::ntsteps_sampling]
        cm_train_1_deg = cm_pitch_1_deg[0::ntsteps_sampling]


        ntsteps = it_train_1_deg.size
        
        tau_train = it_train_1_deg*dt*2*V/chord
        time_train = it_train_1_deg*dt
        tau_ref = 1/(otref*dt*2*V/chord)

        aoa = ampli_train_pitch*(1. - np.exp(-time_1_deg*otref))
        aoa_train = ampli_train_pitch*(1. - np.exp(-(time_train+2*dt*ntsteps_sampling)*otref))

        ### Plot pitch angle input vs tau
        # plt.rcParams.update({'font.size': 16})
        # fig, ax = plt.subplots(1,1)
        # fig.set_size_inches(9.25, 5.25)
        # ax.plot(tau_train,aoa[::ntsteps_sampling]*180./np.pi,'k-', label = '1 deg',linewidth=2)
        # ax.plot(tau_train,aoa[::ntsteps_sampling]*2*180./np.pi,'k--',  label = '2 deg',linewidth=2)
        # ax.set_xlabel(r'$\tau \,\, [\,]$',fontsize=16)
        # ax.set_ylabel(r'$\alpha \,\, [deg]$',fontsize=16)
        # ax.legend(loc='lower right',fontsize=16)
        # plt.grid()
        # plt.yticks(np.arange(0, 2.5, step=0.5))
        # plt.savefig('Photo\\'+ 'step_pitch.png', bbox_inches='tight')
        # plt.show()


        if laguerre_type == 'scipy':
            # Laguerre
            a=1e-20 #0.95 #1.0 #0.5
            alpha = -0.999
            L = np.empty((ntsteps,n_coeff))
            l = np.empty((ntsteps,n_coeff))
            for j in range(n_coeff):
                L[:,j] = genlaguerre(j, alpha)(tau_train)
                l[:,j] = L[:,j]*np.exp(-a*tau_train)
            B = L

        if laguerre_type == 'dowell':
            # Laguerre
            a=0.3 #0.95 #1.0 #0.5
            l = np.empty((ntsteps,n_coeff))         
            for j in range(n_coeff):
                L = np.empty((ntsteps))
                for k in range(j):
                    b= np.sqrt(2*a)* ((-1)**k) *math.factorial(j)* (2**(j-k)) * ((2*a*tau_train)**(j-k)) *np.exp(-a*tau_train) /(math.factorial(k) * (math.factorial(j-k))**2)
                    np.add(L, b, out=L, casting="unsafe")
                l[:,j] = L
            B = np.transpose(l)
            B = B.T
        
        U = np.zeros((ntsteps,ntsteps))

        for k in range(ntsteps):
            U += aoa_train[k]*np.diag(np.ones(ntsteps - k),-k)
        

        uu, s, vh = linalg.svd(U@B,full_matrices=False)    

        theta_cl_temp = vh.T@(np.diag(1/s)@(uu.T@cl_train_1_deg)) # coeff di sviluppi di Volterra lineare
        theta_cm_temp = vh.T@(np.diag(1/s)@(uu.T@cm_train_1_deg))

        kernels_CL_temp = B@theta_cl_temp
        kernels_CM_temp = B@theta_cm_temp

        kernels_CL.append(kernels_CL_temp)
        kernels_CM.append(kernels_CM_temp)
        

        cltest_check = U@B@theta_cl_temp
        cmtest_check = U@B@theta_cm_temp

        cl_pred = U@kernels_CL_linear_pred[i,:]
        cm_pred = U@kernels_CM_linear_pred[i,:]

        cl_pred_GP = U@kernels_CL_linear_pred_GP[i,:]
        cm_pred_GP = U@kernels_CM_linear_pred_GP[i,:]

        confidence_interval = 0.68 # 1 sigma = 68% 
        z = scipy.stats.norm.ppf(1 - (1 - confidence_interval) / 2)

        cl_pred_GP_upper = U@(kernels_CL_linear_pred_GP[i,:] + z * np.sqrt(variance_kernels_CL_linear_pred_GP[i,:]))
        cl_pred_GP_lower = U@(kernels_CL_linear_pred_GP[i,:] - z * np.sqrt(variance_kernels_CL_linear_pred_GP[i,:]))
        cm_pred_GP_upper = U@(kernels_CM_linear_pred_GP[i,:] + z * np.sqrt(variance_kernels_CM_linear_pred_GP[i,:]))
        cm_pred_GP_lower = U@(kernels_CM_linear_pred_GP[i,:] - z * np.sqrt(variance_kernels_CM_linear_pred_GP[i,:]))

        ncoeff = np.arange(kernels_CL_linear_pred.shape[1])

        if plot_flag_linear_response:
            # Prediction
            plt.rcParams.update({'font.size': 16})
            fig, ((ax, ax1), (ax2, ax3)) = plt.subplots(2, 2)
            fig.set_size_inches(18.5, 10.5)

            fig.suptitle(' Prediction - M = '+str(Mach[i])+' - AoA = '+str(AoA[i]))
            ax.bar(ncoeff,kernels_CL_temp,color='none', edgecolor='black',width=1.0, label='Volterra Ref.', linewidth=2)
            ax.bar(ncoeff,kernels_CL_linear_pred[i,:], label='Volterra - FCNN',color='red', edgecolor='red',width=0.5,alpha=0.4)
            ax.bar(ncoeff,kernels_CL_linear_pred_GP[i,:], label='Volterra - GP',color='blue' , edgecolor='blue',width=0.5,alpha=0.4)
            ax.set_xlabel(r'Kernel No.',fontsize = 16)
            ax.set_ylabel(r'Kernel Magnitude',fontsize = 16)
            ax.legend(loc='upper right',fontsize = 16)
            ax.grid()
            ax.set_title('Lift coefficient',fontsize = 16)

            ax1.bar(ncoeff,kernels_CM_temp,color='none', edgecolor='black',width=1.0, label='Volterra Ref.', linewidth=2)
            ax1.bar(ncoeff,kernels_CM_linear_pred[i,:], label='Volterra - FCNN',color='red', edgecolor='red',width=0.5,alpha=0.4)
            ax1.bar(ncoeff,kernels_CM_linear_pred_GP[i,:], label='Volterra - GP',color='blue' , edgecolor='blue',width=0.5,alpha=0.4)
            ax1.set_xlabel(r'Kernel No.',fontsize = 16)
            ax1.set_ylabel(r'Kernel Magnitude',fontsize = 16)
            ax1.legend(loc='upper right',fontsize = 16)
            ax1.grid()
            ax1.set_title('Pitching moment coefficient',fontsize = 16)

            # ax2.plot(tau_train,cl_train_1_deg,'k',linewidth=2,label='CFD')
            ax2.plot(tau_train,cltest_check,'k',linewidth=2,label='Volterra Ref.')
            ax2.scatter(tau_train,cl_pred, marker='o', facecolors='black', edgecolors='black',label='Volterra - FCNN')
            ax2.scatter(tau_train,cl_pred_GP,marker='s', facecolors='white', edgecolors='black',label='Volterra - GP')
            # ax2.fill_between(
            #     tau_train,
            #         cl_pred_GP_lower,
            #         cl_pred_GP_upper,
            #         alpha=0.5,
            #         label=r"GP - $\pm$ 1$\sigma$",
            #     )
            ax2.set_xlabel(r'$\tau$',fontsize = 16)
            ax2.set_ylabel(r'$C_L - C_{L0}$',fontsize = 16)
            ax2.legend(loc='lower right',fontsize = 16)
            ax2.grid()

            # ax3.plot(tau_train,cm_train_1_deg,'k',linewidth=2,label='CFD')
            ax3.plot(tau_train,cmtest_check,'k',linewidth=2,label='Volterra Ref.')
            ax3.scatter(tau_train,cm_pred, marker='o', facecolors='black', edgecolors='black',label='Volterra - FCNN')
            ax3.scatter(tau_train,cm_pred_GP,marker='s', facecolors='white', edgecolors='black',label='Volterra - GP')
            # ax3.fill_between(
            #     tau_train,
            #         cm_pred_GP_lower,
            #         cm_pred_GP_upper,
            #         alpha=0.5,
            #         # label=r"60% confidence interval GP",
            #         label=r"GP - $\pm$ 1$\sigma$",
            #         )
            ax3.set_xlabel(r'$\tau$',fontsize = 16)
            ax3.set_ylabel(r'$C_M - C_{M0}$',fontsize = 16)
            ax3.legend(loc='lower right',fontsize = 16)
            ax3.grid()
            plt.subplots_adjust(wspace=0.4)

            plt.savefig('Photo\\'+ 'volterra_kernel_linear_low_Mach.png', bbox_inches='tight')

            plt.show()
            plt.close()

            # np.save('BSCW_step_plunge\\Dataset_step_pitch\\kernels_CL_linear_step_pitch_M0745_AoA2105.npy', kernels_CL_temp)
            # np.save('BSCW_step_plunge\\Dataset_step_pitch\\kernels_CM_linear_step_pitch_M0745_AoA2105.npy', kernels_CM_temp)
            # np.save('BSCW_step_plunge\\Dataset_step_pitch\\CL_CFD_step_pitch_M0745_AoA2105.npy', cl_train_1_deg)
            # np.save('BSCW_step_plunge\\Dataset_step_pitch\\CL_Volterra_step_pitch_M0745_AoA2105.npy', cltest_check)
            # np.save('BSCW_step_plunge\\Dataset_step_pitch\\CM_CFD_step_pitch_M0745_AoA2105.npy', cm_train_1_deg)
            # np.save('BSCW_step_plunge\\Dataset_step_pitch\\CM_Volterra_step_pitch_M0745_AoA2105.npy', cmtest_check)

        ###### Nonlinear Kernels ######
        ampli_train_pitch_2_deg = 2.0
        it_pitch_2_deg = np.arange(0,CL_2_deg.shape[1],1)
        cl_pitch_2_deg = CL_2_deg[i,:]-cl0[i,:]
        cm_pitch_2_deg = CM_2_deg[i,:]-cm0[i,:]

        # cl_pitch_2_deg = moving_average_output(cl_pitch_2_deg,moving_avarage_skipping_iter = 150 ,averaging_window = 50)
        # cm_pitch_2_deg = moving_average_output(cm_pitch_2_deg,moving_avarage_skipping_iter = 200 ,averaging_window = 200)

        # apply a 3-pole lowpass filter at 0.1x Nyquist frequency
        b, a = scipy.signal.butter(1, 0.02)
        b1, a1 = scipy.signal.butter(2, 0.007)
        cl_pitch_2_deg = scipy.signal.filtfilt(b, a, cl_pitch_2_deg)
        cm_pitch_2_deg = scipy.signal.filtfilt(b1, a1, cm_pitch_2_deg)

        # tau = it_pitch_2_deg*dt*2*V/chord
        time_2_deg = it_pitch_2_deg*dt

        it_train_2_deg = it_pitch_2_deg[0::ntsteps_sampling] # it: numero di iterazioni
        cl_train_2_deg = cl_pitch_2_deg[0::ntsteps_sampling]
        cm_train_2_deg = cm_pitch_2_deg[0::ntsteps_sampling]


        # min length
        n_train_delta = np.minimum(cl_train_1_deg.size,cl_train_2_deg.size)

        tau_train_delta = np.arange(n_train_delta)*dtau
        time_train_delta = np.arange(n_train_delta)*dt

        cl_pitch_train_delta = -cl_train_1_deg[0:n_train_delta]*factor +cl_train_2_deg[0:n_train_delta] 
        cm_pitch_train_delta = -cm_train_1_deg[0:n_train_delta]*factor +cm_train_2_deg[0:n_train_delta] 

        aoa_delta = ampli_train_pitch_2_deg*(1. - np.exp(-time_2_deg*otref))
        aoa_train_delta = ampli_train_pitch_2_deg*(1. - np.exp(-time_train_delta*otref))

        if laguerre_type == 'scipy':
            # Laguerre
            a=1e-20 #0.95 #1.0 #0.5
            alpha = -0.999
            L = np.empty((n_train_delta,n_coeff))
            l = np.empty((n_train_delta,n_coeff))
            for j in range(n_coeff):
                L[:,j] = genlaguerre(j, alpha)(tau_train_delta)
                l[:,j] = L[:,j]*np.exp(-a*tau_train_delta)
            B = L

        if laguerre_type == 'dowell':
        # Laguerre
            a=0.3 #0.95 #1.0 #0.5
            l = np.empty((n_train_delta,n_coeff))          
            for j in range(n_coeff):
                L = np.empty((n_train_delta))
                for k in range(j):
                    b= np.sqrt(2*a)* ((-1)**k) *math.factorial(j)* (2**(j-k)) * ((2*a*tau_train_delta)**(j-k)) *np.exp(-a*tau_train_delta) /(math.factorial(k) * (math.factorial(j-k))**2)
                    np.add(L, b, out=L, casting="unsafe")
                l[:,j] = L
            B = np.transpose(l)
            B = B.T
        
        U2 = np.zeros((n_train_delta,n_train_delta))

        for k in range(n_train_delta):
            U2 += aoa_train_delta[k]*np.diag(np.ones(n_train_delta - k),-k)
        

        uu, s, vh = linalg.svd(U2@B,full_matrices=False)    

        theta_cl_temp_NL = vh.T@(np.diag(1/s)@(uu.T@cl_pitch_train_delta)) # coeff di sviluppi di Volterra lineare
        theta_cm_temp_NL = vh.T@(np.diag(1/s)@(uu.T@cm_pitch_train_delta))

        kernels_CL_NL_temp = B@theta_cl_temp_NL
        kernels_CM_NL_temp = B@theta_cm_temp_NL


        # zeroing higher kernels 
        zeroing = np.tanh(1.-0.02*np.linspace(0,B.shape[0],B.shape[0]))
        kernels_CL_NL_temp = kernels_CL_NL_temp*zeroing
        kernels_CM_NL_temp = kernels_CM_NL_temp*zeroing

        kernels_CL_NL.append(kernels_CL_NL_temp)
        kernels_CM_NL.append(kernels_CM_NL_temp)


        cltest_check_NL = U2@B@theta_cl_temp_NL
        cmtest_check_NL = U2@B@theta_cm_temp_NL


        cl_NL_pred = U2@kernels_CL_NL_pred[i,:]
        cm_NL_pred = U2@kernels_CM_NL_pred[i,:]

        cl_NL_pred_GP = U2@kernels_CL_NL_pred_GP[i,:]
        cm_NL_pred_GP = U2@kernels_CM_NL_pred_GP[i,:]

        confidence_interval = 0.68 # 1 sigma = 68% 
        z = scipy.stats.norm.ppf(1 - (1 - confidence_interval) / 2)

        cl_NL_pred_GP_upper = U2@(kernels_CL_NL_pred_GP[i,:] + z * np.sqrt(variance_kernels_CL_NL_pred_GP[i,:]))
        cl_NL_pred_GP_lower = U2@(kernels_CL_NL_pred_GP[i,:] - z * np.sqrt(variance_kernels_CL_NL_pred_GP[i,:]))
        cm_NL_pred_GP_upper = U2@(kernels_CM_NL_pred_GP[i,:] + z * np.sqrt(variance_kernels_CM_NL_pred_GP[i,:]))
        cm_NL_pred_GP_lower = U2@(kernels_CM_NL_pred_GP[i,:] - z * np.sqrt(variance_kernels_CM_NL_pred_GP[i,:]))

        cl_NL_FCNN.append(cl_NL_pred)
        cm_NL_FCNN.append(cm_NL_pred)

        if plot_flag_NL_response:
            plt.rcParams.update({'font.size': 16})

            fig, ((ax, ax1), (ax2, ax3)) = plt.subplots(2, 2)
            fig.set_size_inches(18.5, 10.5)

            fig.suptitle(' Prediction - M = '+str(Mach[i])+' - AoA = '+str(AoA[i]))
            ax.bar(ncoeff,kernels_CL_NL_temp, color='none', edgecolor='black',width=1.0, label='Volterra Ref.', linewidth=2)
            ax.bar(ncoeff,kernels_CL_NL_pred[i,:], label='Volterra - FCNN',color='red', edgecolor='red',width=0.5,alpha=0.4)
            ax.bar(ncoeff,kernels_CL_NL_pred_GP[i,:], label='Volterra - GP',color='blue' , edgecolor='blue',width=0.5,alpha=0.4)
            ax.set_xlabel(r'Kernel No.',fontsize = 16)
            ax.set_ylabel(r'Kernel Magnitude',fontsize = 16)
            ax.legend(loc='lower right',fontsize = 16)
            ax.grid()
            ax.set_title(r'$\Delta C_L$',fontsize = 16)
            
            ax1.bar(ncoeff,kernels_CM_NL_temp, color='none', edgecolor='black',width=1.0, label='Volterra Ref.', linewidth=2)
            ax1.bar(ncoeff,kernels_CM_NL_pred[i,:], label='Volterra - FCNN',color='red', edgecolor='red',width=0.5,alpha=0.4)
            ax1.bar(ncoeff,kernels_CM_NL_pred_GP[i,:], label='Volterra - GP',color='blue' , edgecolor='blue',width=0.5,alpha=0.4)
            ax1.set_xlabel(r'Kernel No.',fontsize = 16)
            ax1.set_ylabel(r'Kernel Magnitude',fontsize = 16)
            ax1.legend(loc='lower right',fontsize = 16)
            ax1.grid()
            ax1.set_title(r'$\Delta C_M$',fontsize = 16)


            # ax2.plot(it_train_2_deg*dt*2*V/chord,cl_train_2_deg,'k',linewidth=2,label='CFD')
            ax2.plot(tau_train_delta,cltest_check*factor + cltest_check_NL,'k',linewidth=2,label='Volterra Ref.')
            ax2.scatter(tau_train_delta,cl_pred*factor + cl_NL_pred,marker='o', facecolors='black', edgecolors='black',label='Volterra - FCNN')
            ax2.scatter(tau_train_delta,cl_pred_GP*factor + cl_NL_pred_GP,marker='s', facecolors='white', edgecolors='black',label='Volterra - GP')
            # ax2.fill_between(
            #     tau_train_delta,
            #         cl_pred_GP*factor + cl_NL_pred_GP_lower,
            #         cl_pred_GP*factor + cl_NL_pred_GP_upper,
            #         alpha=0.5,
            #         # label=r"60% confidence interval GP",
            #         label=r"GP - $\pm$ 1$\sigma$",
            #         )
            ax2.set_xlabel(r'$\tau$',fontsize = 16)
            ax2.set_ylabel(r'$C_L - C_{L0}$',fontsize = 16)
            ax2.legend(loc='lower right',fontsize = 16)
            ax2.grid()

            # ax3.plot(it_train_2_deg*dt*2*V/chord,cm_train_2_deg,'k',linewidth=2,label='CFD')
            ax3.plot(tau_train_delta,cmtest_check*factor + cmtest_check_NL,'k',linewidth=2,label='Volterra Ref.')
            ax3.scatter(tau_train_delta,cm_pred*factor + cm_NL_pred,marker='o', facecolors='black', edgecolors='black',label='Volterra - FCNN')
            ax3.scatter(tau_train_delta,cm_pred_GP*factor + cm_NL_pred_GP,marker='s', facecolors='white', edgecolors='black',label='Volterra - GP')
            # ax3.fill_between(
            #     tau_train_delta,
            #         cm_pred_GP*factor + cm_NL_pred_GP_lower,
            #         cm_pred_GP*factor + cm_NL_pred_GP_upper,
            #         alpha=0.5,
            #         label=r"GP - $\pm$ 1$\sigma$",
            #     )
            ax3.set_xlabel(r'$\tau$',fontsize = 16)
            ax3.set_ylabel(r'$C_M - C_{M0}$',fontsize = 16)
            ax3.legend(loc='lower right',fontsize = 16)
            ax3.grid()
            plt.subplots_adjust(wspace=0.4)

            plt.savefig('Photo\\'+ 'volterra_kernel_NL_low_Mach.png', bbox_inches='tight')
            plt.show()
            plt.close()

            # np.save('BSCW_step_plunge\\Dataset_step_pitch\\kernels_CL_NL_step_pitch_M0745_AoA2105.npy', kernels_CL_NL_temp)
            # np.save('BSCW_step_plunge\\Dataset_step_pitch\\kernels_CM_NL_step_pitch_M0745_AoA2105.npy', kernels_CM_NL_temp)
            # np.save('BSCW_step_plunge\\Dataset_step_pitch\\CL_CFD_step_pitch_2deg_M0745_AoA2105.npy', cl_train_2_deg)
            # np.save('BSCW_step_plunge\\Dataset_step_pitch\\CL_Volterra_step_pitch_2deg_M0745_AoA2105.npy', (cltest_check*factor + cltest_check_NL))
            # np.save('BSCW_step_plunge\\Dataset_step_pitch\\CM_CFD_step_pitch_2deg_M0745_AoA2105.npy', cm_train_2_deg)
            # np.save('BSCW_step_plunge\\Dataset_step_pitch\\CM_Volterra_step_pitch_2deg_M0745_AoA2105.npy', (cmtest_check*factor + cmtest_check_NL))




        # ## Plot nonlinear contribute to CL
        # plt.rcParams.update({'font.size': 16})
        # fig, (ax, ax1) = plt.subplots(1,2)
        # fig.set_size_inches(18.5, 5.25)
        # ax.plot(it_train_2_deg[::2]*dt*2*V/chord,cl_train_2_deg[::2]/factor,linestyle = 'None',marker='s', markersize=8, markerfacecolor='gray', markeredgecolor='black',label='CFD') 
        # ax.plot(tau_train_delta,(cltest_check*factor)/factor,'k-',linewidth=2,label='Volterra Linear')
        # ax.plot(tau_train_delta,(cltest_check*factor + cltest_check_NL)/factor,'k--',linewidth=2,label='Volterra NL')
        # ax.set_xlabel(r'$\tau \,\, [\,]$',fontsize = 16)
        # ax.set_ylabel(r'$(C_L - C_{L0})\, / \, \alpha_0}$',fontsize = 16)
        # ax.legend(loc='lower right',fontsize = 16)
        # ax.grid()

        # plt.subplots_adjust(wspace=0.3)
        # ax1.plot(it_train_2_deg[::2]*dt*2*V/chord,cm_train_2_deg[::2]/factor,linestyle = 'None',marker='s', markersize=8, markerfacecolor='gray', markeredgecolor='black',label='CFD') 
        # ax1.plot(tau_train_delta,(cmtest_check*factor)/factor,'k-',linewidth=2,label='Volterra Linear')
        # ax1.plot(tau_train_delta,(cmtest_check*factor + cmtest_check_NL)/factor,'k--',linewidth=2,label='Volterra NL')
        # ax1.set_xlabel(r'$\tau \,\, [\,]$',fontsize = 16)
        # ax1.set_ylabel(r'$(C_M - C_{M0})\, / \, \alpha_0}$',fontsize = 16)
        # ax1.legend(loc='lower right',fontsize = 16)
        # ax1.grid()
        # ax1.set_ylim(-0.013,0.022)
        # # plt.yticks(np.arange(0, 2.5, step=0.5))
        # plt.savefig('Photo\\'+ 'nonlinear_contribute_CL.png', bbox_inches='tight')
        # plt.show()
        # plt.close()

        def FreqAnalysManual(t, x):
            '''

            :param t: time
            :param x: signal
            :return: poly curve fitting of the peaks
            '''

            N = x.shape[0]   # sample size

            peaks = []
            n = 5 # look for the 5 points before and after
            for ii in range(n,x.shape[0]-n):
                if max(x[ii-n:ii+n]) == x[ii]:
                    peaks.append(ii)

            #discarding the first few peaks:
            peaks = peaks[4::3]

            # linear fit:  y = a x + b
            a, b = np.polyfit(t[peaks], np.log(x[peaks]), 1)
            # print(a, b)

            return peaks, a, b

        def Func11Fig2Plots(x1, y1, x2, y2, axis_labels, plot_labels, fn):

            '''

            Creates 1 figure plot from 2 datas.

            :param x1: x for plot 1
            :param y1: y for plot 1
            :param x2: x for plot 2
            :param y2: y for plot 2
            :param axis_labels: labels for x and y axes in a 2-element list.
            :param plot_labels: plot legend in a 2-element list
            :param fn: filname of the figure to save without the extension
            :return: makes the plot
            '''

            # wdir = os.getcwd()

            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "sans-serif",
                "font.sans-serif": ["Helvetica"]})

            # Plot sizes:
            plt.rc('font', family='serif')
            SMALL_SIZE = 44  # 26
            MEDIUM_SIZE = 46  # 28
            BIGGER_SIZE = 48  # 30
            plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
            plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
            plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
            plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
            plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
            plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
            plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
            # matplotlib.rcParams[
                # "legend.loc"] = 'best' #'upper right'  # 'lower right'  # 'lower left'   # legend position. 3: lower left




            fig = plt.figure(figsize=(15,15))
            fig.suptitle("Flutter Analysis", fontsize=BIGGER_SIZE)

            ax = fig.add_subplot(111)
            ax.plot(x1, y1, c='b', linewidth=2, marker='+', markersize=20, label=plot_labels[0])
            ax.plot(x2, y2, c='r', linewidth=2, marker='+', markersize=20, label=plot_labels[1])
            ax.set_xlabel(axis_labels[0])
            ax.set_ylabel(axis_labels[1])
            ax.legend()
            # plt.legend(loc='upper right', bbox_to_anchor=(1.02, 0.5))  #0.33)) #bbox_transform=plt.gcf().transFigure)
            ax.set_xlim((min(x1), max(x1)))  # 0.5/max(dt_0,dt2_0)))
            # ax.set_xticklabels([])
            # at = AnchoredText(lab, prop=dict(size=15), frameon=True, loc='upper right',)
            # at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            # ax.add_artist(at)
            ax.grid()

            plt.show()
            # plt.plot()

            # plt.savefig(wdir + '/' + fn + '.png', bbox_inches='tight')

            return
        
        # Calculate flutter using the predictions from Neural Network
        if calculate_flutter:
            ###  FLUTTER ANALYSIS 
            # 2DOF SS system 

            ndof = ntsteps + 4 
            ndof_aero = ntsteps
            
            AA = np.zeros((ndof,ndof))
            BB = np.zeros((ndof,ndof))

            mass= 87.91; # kg
            Iy= 3.765;   # kg m2
            kz=mass*(3.33*2*np.pi)**2;   # N m−1
            ktheta=Iy*(5.20*2*np.pi)**2; # Nm rad−1
            
            for rho_value in np.arange(1.3,2.0,0.1):
                # print(rho_value)
                rho = 1.1751*rho_value #1.15*0.95 #0.87
                q = 0.5*rho*V*V
                # print(q/47.88)

                xi = 0.000 # damping ratio

                n_dof = 2 # physical DOFs

                K_aa = np.array([[ktheta, 0],[0, kz]])
                M_aa = np.array([[Iy, 0],[0,mass]])
                C_aa = 2*xi* np.diag([np.sqrt(Iy*ktheta), np.sqrt(mass*kz)])

                Aphys = np.vstack([np.hstack([np.zeros((n_dof,n_dof)), np.eye(n_dof)]), np.hstack([-K_aa, -C_aa])])
                Bphys = np.vstack([np.hstack([np.eye(n_dof), np.zeros((n_dof,n_dof))]), np.hstack([np.zeros((n_dof,n_dof)), M_aa])])

                DT = dtau*chord/(2*V) 
                nt = 12500
                beta = 0.5 

                # ti_max = DT*nt
                ti = np.linspace(0,nt*DT,nt)
                
                # aerodynamic forces contribution by (n+1) aoa_eff

                Aphys[n_dof,0] += q*area*chord*(kernels_CM_linear_pred[i,:])[0]*beta
                Aphys[n_dof+1,0] += q*area*(kernels_CL_linear_pred[i,:])[0]*beta
                
                Aphys[n_dof,n_dof+1] += -q*area*chord*(kernels_CM_linear_pred[i,:])[0]*beta/V
                Aphys[n_dof+1,n_dof+1] += -q*area*(kernels_CL_linear_pred[i,:])[0]*beta/V

                Aphys[n_dof,n_dof] += 0. #q*area*chord*(B@theta_cm)[0]*beta*chord/(2*V)
                Aphys[n_dof+1,n_dof] += 0. #q*area*(B@theta_cl)[0]*beta*chord/(2*V)

                # discrete time 
                Aphys_D = Bphys + DT*(1.-beta)*Aphys
                Bphys_D = Bphys - DT*beta*Aphys #

                # top left partition of states materix
                AA_D = linalg.inv(Bphys_D)@Aphys_D

                # top right partition of states matrix 
                BZ = np.zeros((2*n_dof,ndof_aero))
                BZ[n_dof,:] = q*area*chord* (beta*(kernels_CM_linear_pred[i,:]) + (1-beta)*np.hstack([kernels_CM_linear_pred[i,:],0.])[1:])
                BZ[n_dof+1,:] = q*area* (    beta*(kernels_CL_linear_pred[i,:])  + (1-beta)*np.hstack([kernels_CL_linear_pred[i,:],0.])[1:])

                BZ_D = linalg.inv(Bphys_D)@BZ

                # assembly of state matrix 
                AA[0:2*n_dof,0:2*n_dof] = AA_D 
                AA[0:2*n_dof,2*n_dof:] = DT*BZ_D

                # aoa_eff value update 
                AA[2*n_dof,0] = 1. 
                AA[2*n_dof,n_dof] = 0. #chord/(2*V) 
                AA[2*n_dof,n_dof+1] = -1./V 
                AA[2*n_dof:,2*n_dof:] = np.diag(np.ones(ndof_aero-1),-1)


                # Flutter search with eigenvalue analysis
                eigenvalue = False
                if eigenvalue: 
                    def flutter_detected(eigenvalues):
                        return any(np.real(val) > 1 for val in eigenvalues)

                    def flutter_criterion(AA):
                        eigenvalues, _ = np.linalg.eig(AA)
                        return flutter_detected(eigenvalues)

                    # Flutter point:
                    if flutter_criterion(AA):
                        print('\n Prediction - M = '+str(Mach[i])+' - AoA = '+str(AoA[i])+ ' Flutter point found at q_f: %0.3f [psf]' %(q/47.88))
            
                        q_flutter[i]=q
                        break

                    else:
                        continue
                
                # LCO determination with time integration
                time_integration = True
                if time_integration: 
                    # time integration 

                    qq_lin = np.empty((ndof,nt))
                    qq_lin[2,0] = 1.0 # PING
                    qq_lin[3,0] = 1.0 # PING
                    qq_NL = np.empty((ndof,nt))
                    qq_NL[2,0] = 1.0 # PING
                    qq_NL[3,0] = 1.0 # PING

                    fNL = np.zeros(ndof)


                    for it in range(1,nt):
                        # Non linear
                        fNL[2] = q*area*chord*(B@theta_cm_temp_NL)@(qq_NL[4:,it-1]*np.abs(qq_NL[4:,it-1])).T
                        fNL[3] = q*area*(B@theta_cl_temp_NL)@(qq_NL[4:,it-1]*np.abs(qq_NL[4:,it-1])).T
                        qq_NL[:,it] = AA@(qq_NL[:,it-1] + DT*fNL) 
                        # Linear
                        qq_lin[:,it] = AA@qq_lin[:,it-1] 


                    plt.rcParams.update({'font.size': 26})
                    fig, ax = plt.subplots(1,1)
                    fig.set_size_inches(18.5, 10.5)

                    fig.suptitle(' Flutter Prediction - M = '+str(Mach[i])+' - AoA = '+str(AoA[i]))

                    ax.plot(qq_NL[1,2000:],qq_NL[0,2000:],'r-',linewidth=2,label='NL')
                    ax.plot(qq_lin[1,2000:],qq_lin[0,2000:],'k-',linewidth=2,label='Linear')
                    ax.set_xlabel(r'$z \,\, [m]$')
                    ax.set_ylabel(r'$\theta \,\, [RAD]$')
                    ax.legend(loc='upper right')
                    # ax[0].set_xlim(0, ti[-1])
                    ax.grid()

                    plt.show()
                    plt.close()

                    plt.rcParams.update({'font.size': 16})
                    fig, ax = plt.subplots(2,1)
                    fig.set_size_inches(18.5, 10.5)

                    fig.suptitle(' Flutter Prediction - M = '+str(Mach[i])+' - AoA = '+str(AoA[i]))

                    ax[0].plot(ti,qq_NL[0,:],'r-',linewidth=2,label='NL')
                    ax[0].plot(ti,qq_lin[0,:],'k-',linewidth=2,label='Linear')
                    ax[0].set_xlabel(r'$\tau \,\, [\,]$',fontsize=16)
                    ax[0].set_ylabel(r'$\theta \,\, [RAD]$',fontsize=16)
                    ax[0].legend(loc='lower right',fontsize=16)
                    ax[0].set_xlim(0, ti[-1])
                    ax[0].grid()


                    ax[1].plot(ti,qq_NL[1,:],'r-',linewidth=2,label='NL')
                    ax[1].plot(ti,qq_lin[1,:],'k-',linewidth=2,label='Linear')
                    ax[1].set_xlabel(r'$\tau \,\, [\,]$',fontsize=16)
                    ax[1].set_ylabel(r'$z \,\, [m]$',fontsize=16)
                    ax[1].legend(loc='lower right',fontsize=16)
                    ax[1].set_xlim(0, ti[-1])
                    ax[1].grid()

                    plt.show()
                    plt.close()

                    ### Calculate flutter with time integration using linear or NL kernels
                    # Getting natural freq and damping factor for each DOF:
                    try:
                        ## Linear
                        peaks, a, b = FreqAnalysManual(ti, qq_lin[0,:])
                        ## NL
                        # peaks, a, b = FreqAnalysManual(ti, qq_NL[0,:])

                        ## Plots
                        # Func11Fig2Plots(ti, qq_NL[i,:], ti[peaks], qq_NL[i,peaks], ['$t$', '$\\alpha$'], ['signal', 'peaks'], 'signal_peaks')
                        # leg2 = '$y = %0.2fx + %0.2f$' % (a, b)
                        # Func11Fig2Plots(ti[peaks], np.log(qq_NL[i,peaks]), ti[peaks], a * ti[peaks] + b, ['$t$', '$\\alpha$'], ['log of peaks', leg2], 'signal_peaks_fit_lin')
                        # leg2 = '$y = %0.2fe^{%0.2fx}$' % (np.exp(b), a)
                        # Func11Fig2Plots(ti[peaks], qq_NL[i,peaks], ti[peaks], np.exp(b) * np.exp(a * ti[peaks]), ['$t$', '$\\alpha$'], ['peaks', leg2], 'signal_peaks_fit_exp')

                        damp = -a

                        if damp <0:
                            # Flutter point:
                            print('\n Prediction - M = '+str(Mach[i])+' - AoA = '+str(AoA[i])+ ' Flutter point found at q_f: %0.3f [psf]' %(q/47.88))
                                    

                            q_flutter[i]=q
                            break

                    except:
                        pass

                    if q_flutter[i]==True:
                        break


    kernels_CL = np.array(kernels_CL)
    kernels_CM = np.array(kernels_CM)
    kernels_CL_NL = np.array(kernels_CL_NL)
    kernels_CM_NL = np.array(kernels_CM_NL)

    # Stack linear kernels for CL and CM
    kernels = np.stack((kernels_CL,kernels_CM), axis=2)
    # Stack NL kernels for CL and CM
    kernels_NL = np.stack((kernels_CL_NL,kernels_CM_NL), axis=2)
    print( '\n-> Shape of the CL and CM matrix for linear kernels: \n')
    print(kernels.shape) 
    print( '\n-> Shape of the CL and CM matrix for NL kernels: \n')
    print(kernels_NL.shape) 

    # save array
    # np.save('Dataset_generator\\kernels_CL_CM_linear.npy', kernels)
    # np.save('Dataset_generator\\kernels_CL_CM_NL.npy', kernels_NL)





# %% ### PLot linear CL and dynamic stall CL for paper plots
# fig , ax = plt.subplots(1, 1)
# fig.set_size_inches(9.25, 5.25)

# x = [47,42]
# for i in x:
#     cl_pitch_1_deg = CL_1_deg[i,:]-cl0[i,:]
#     b, a = scipy.signal.butter(1, 0.01)
#     cl_pitch_1_deg = scipy.signal.filtfilt(b, a, cl_pitch_1_deg)
#     time_1_deg = it_pitch_1_deg*dt
#     cl_train_1_deg = cl_pitch_1_deg[0::ntsteps_sampling]
#     tau_train = it_train_1_deg*dt*2*V/chord


#     # fig.suptitle(' Prediction - M = '+str(Mach[i])+' - AoA = '+str(AoA[i]))

#     ax.plot(tau_train,cl_train_1_deg,linewidth=2,label='M = '+str(Mach[i])+r' - $\alpha_0$ = '+str(AoA[i]))
#     ax.set_xlabel(r'$\tau$')
#     ax.set_ylabel(r'$C_L - C_{L0}$')
#     ax.legend(loc='lower right')

# plt.grid()
# plt.show()

#%% Prediction M = 0.80 - 0 < AoA < 5
if prediction_M080_AOA5:
    # import kernels linear and NL M=0.80 AoA=5
    kernels_CL_linear_pred_M080_AoA5 = np.load('Dataset_generator\\kernels_CL_linear_pred_M080_AoA5.npy') # samples,timesteps,variables
    kernels_CM_linear_pred_M080_AoA5 = np.load('Dataset_generator\\kernels_CM_linear_pred_M080_AoA5.npy') # samples,timesteps,variables
    kernels_CL_NL_pred_M080_AoA5 = np.load('Dataset_generator\\kernels_CL_NL_pred_M080_AoA5.npy') # samples,timesteps,variables
    kernels_CM_NL_pred_M080_AoA5 = np.load('Dataset_generator\\kernels_CM_NL_pred_M080_AoA5.npy') # samples,timesteps,variables
    
    Mach_i = 0.80
    AoA = np.arange(0,5.25,0.25)

    cl_pred__M080_AoA5 = []
    cm_pred__M080_AoA5 = []
    q_flutter_M080_AOA5 = np.empty(kernels_CL_linear_pred_M080_AoA5.shape[0])

    for i in range(kernels_CL_linear_pred_M080_AoA5.shape[0]):

        # tau: tempo ridotto
        # t: tempo fisico
        V = Mach_i*np.sqrt(1.116*81.49*304.2128)

        ### Linear kernels
        tau_sampling = 0.6 #0.125#0.5
        ntsteps_sampling = 18 # int(tau_sampling/(dt*2*V/chord))
        dtau = ntsteps_sampling*dt*2*V/chord
        
        it_pitch_1_deg = np.arange(0,CL_1_deg.shape[1],1)

        tau = it_pitch_1_deg*dt*2*V/chord
        time_1_deg = it_pitch_1_deg*dt

        it_train_1_deg = it_pitch_1_deg[0::ntsteps_sampling] # it: numero di iterazioni

        ntsteps = it_train_1_deg.size
        
        tau_train = it_train_1_deg*dt*2*V/chord
        time_train = it_train_1_deg*dt

        aoa = ampli_train_pitch*(1. - np.exp(-time_1_deg*otref))
        aoa_train = ampli_train_pitch*(1. - np.exp(-(time_train+2*dt*ntsteps_sampling)*otref))

        
        U = np.zeros((ntsteps,ntsteps))

        for k in range(ntsteps):
            U += aoa_train[k]*np.diag(np.ones(ntsteps - k),-k)
        

        cl_pred = U@kernels_CL_linear_pred_M080_AoA5[i,:]
        cm_pred = U@kernels_CM_linear_pred_M080_AoA5[i,:]


        ncoeff = np.arange(kernels_CL_linear_pred.shape[1])


        ### NL kernels
        ampli_train_pitch_2_deg = 2.0
        it_pitch_2_deg = np.arange(0,CL_2_deg.shape[1],1)

        # tau = it_pitch_2_deg*dt*2*V/chord
        time_2_deg = it_pitch_2_deg*dt

        it_train_2_deg = it_pitch_2_deg[0::ntsteps_sampling] # it: numero di iterazioni


        # min length
        n_train_delta = np.minimum(cl_train_1_deg.size,cl_train_2_deg.size)

        tau_train_delta = np.arange(n_train_delta)*dtau
        time_train_delta = np.arange(n_train_delta)*dt


        aoa_delta = ampli_train_pitch_2_deg*(1. - np.exp(-time_2_deg*otref))
        aoa_train_delta = ampli_train_pitch_2_deg*(1. - np.exp(-time_train_delta*otref))

        U2 = np.zeros((n_train_delta,n_train_delta))

        for k in range(n_train_delta):
            U2 += aoa_train_delta[k]*np.diag(np.ones(n_train_delta - k),-k)
        

        cl_NL_pred_M080_AoA5 = U2@kernels_CL_NL_pred_M080_AoA5[i,:]
        cm_NL_pred_M080_AoA5 = U2@kernels_CM_NL_pred_M080_AoA5[i,:]

        cl_pred__M080_AoA5_temp = cl_pred*factor + cl_NL_pred_M080_AoA5
        cl_pred__M080_AoA5.append(cl_pred__M080_AoA5_temp)

        cm_pred__M080_AoA5_temp = cm_pred*factor + cm_NL_pred_M080_AoA5
        cm_pred__M080_AoA5.append(cm_pred__M080_AoA5_temp)


        ###  FLUTTER ANALYSIS
        # 2DOF SS system 

        ndof = ntsteps + 4 
        ndof_aero = ntsteps
        
        AA = np.zeros((ndof,ndof))
        BB = np.zeros((ndof,ndof))

        mass= 87.91; # kg
        Iy= 3.765;   # kg m2
        kz=mass*(3.33*2*np.pi)**2;   # N m−1
        ktheta=Iy*(5.20*2*np.pi)**2; # Nm rad−1
        

        for rho_value in np.arange(0.20,2.0,0.01):
            rho = 1.1751*rho_value #1.15*0.95 #0.87
            q = 0.5*rho*V*V

            xi = 0.000

            n_dof = 2 # physical DOFs

            K_aa = np.array([[ktheta, 0],[0, kz]])
            M_aa = np.array([[Iy, 0],[0,mass]])
            C_aa = 2*xi* np.diag([np.sqrt(Iy*ktheta), np.sqrt(mass*kz)])

            Aphys = np.vstack([np.hstack([np.zeros((n_dof,n_dof)), np.eye(n_dof)]), np.hstack([-K_aa, -C_aa])])
            Bphys = np.vstack([np.hstack([np.eye(n_dof), np.zeros((n_dof,n_dof))]), np.hstack([np.zeros((n_dof,n_dof)), M_aa])])

            DT = dtau*chord/(2*V) 
            nt = 12500
            beta = 0.5 
            # ti_max = DT*nt
            ti = np.linspace(0,nt*DT,nt)
            
            # aerodynamic forces contribution by (n+1) aoa_eff       
            kernels_CL_linear_pred = kernels_CL_linear_pred_M080_AoA5
            kernels_CM_linear_pred = kernels_CM_linear_pred_M080_AoA5
            kernels_CL_NL_pred = kernels_CL_NL_pred_M080_AoA5
            kernels_CM_NL_pred = kernels_CM_NL_pred_M080_AoA5

            Aphys[n_dof,0] += q*area*chord*(kernels_CM_linear_pred[i,:])[0]*beta
            Aphys[n_dof+1,0] += q*area*(kernels_CL_linear_pred[i,:])[0]*beta
            
            Aphys[n_dof,n_dof+1] += -q*area*chord*(kernels_CM_linear_pred[i,:])[0]*beta/V
            Aphys[n_dof+1,n_dof+1] += -q*area*(kernels_CL_linear_pred[i,:])[0]*beta/V

            Aphys[n_dof,n_dof] += 0. #q*area*chord*(B@theta_cm)[0]*beta*chord/(2*V)
            Aphys[n_dof+1,n_dof] += 0. #q*area*(B@theta_cl)[0]*beta*chord/(2*V)

            # discrete time 
            Aphys_D = Bphys + DT*(1.-beta)*Aphys
            Bphys_D = Bphys - DT*beta*Aphys #

            # top left partition of states materix
            AA_D = linalg.inv(Bphys_D)@Aphys_D

            # top right partition of states matrix 
            BZ = np.zeros((2*n_dof,ndof_aero))
            BZ[n_dof,:] = q*area*chord* (beta*(kernels_CM_linear_pred[i,:]) + (1-beta)*np.hstack([kernels_CM_linear_pred[i,:],0.])[1:])
            BZ[n_dof+1,:] = q*area* (    beta*(kernels_CL_linear_pred[i,:])  + (1-beta)*np.hstack([kernels_CL_linear_pred[i,:],0.])[1:])

            BZ_D = linalg.inv(Bphys_D)@BZ

            # assembly of state matrix 
            AA[0:2*n_dof,0:2*n_dof] = AA_D 
            AA[0:2*n_dof,2*n_dof:] = DT*BZ_D

            # aoa_eff value update 
            AA[2*n_dof,0] = 1. 
            AA[2*n_dof,n_dof] = 0. #chord/(2*V) 
            AA[2*n_dof,n_dof+1] = -1./V 
            AA[2*n_dof:,2*n_dof:] = np.diag(np.ones(ndof_aero-1),-1)


            # Flutter search with eigenvalue analysis

            def flutter_detected(eigenvalues):
                return any(np.real(val) > 1 for val in eigenvalues)

            def flutter_criterion(AA):
                eigenvalues, _ = np.linalg.eig(AA)
                return flutter_detected(eigenvalues)

            # Flutter point:
            if flutter_criterion(AA):
                # print('\n Prediction - M = 0.80 - AoA = '+str(AoA[i])+ ' Flutter point found at q_f: %0.3f [psf]' %(q/47.88))
                q_flutter_M080_AOA5[i]=q
                break

            else:
                continue
            


    cl_pred__M080_AoA5 = np.array(cl_pred__M080_AoA5)
    cm_pred__M080_AoA5 = np.array(cm_pred__M080_AoA5)


    # Plot q flutter for M = 0.80 - 0 < AoA < 5 
    fig, (ax1) = plt.subplots(1,1)
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle('Q flutter [psf] - M = 0.80')
    ax1.plot(AoA,q_flutter_M080_AOA5/47.88,'k',linewidth=2)#,label='M = 0.80')
    ax1.set_xlabel(r'AoA [deg]')
    ax1.set_ylabel(r'Q flutter [psf]')
    ax1.legend(loc='lower left')
    ax1.grid()
    plt.show()



#%% Prediction on flow conditions of flutter exp. data
if prediction_exp_data:
    # import kernels linear and NL M=0.80 AoA=5
    kernels_CL_linear_pred_exp_data = np.load('Dataset_generator\\kernels_CL_linear_pred_exp_data.npy') # samples,timesteps,variables
    kernels_CM_linear_pred_exp_data = np.load('Dataset_generator\\kernels_CM_linear_pred_exp_data.npy') # samples,timesteps,variables
    kernels_CL_NL_pred_exp_data = np.load('Dataset_generator\\kernels_CL_NL_pred_exp_data.npy') # samples,timesteps,variables
    kernels_CM_NL_pred_exp_data = np.load('Dataset_generator\\kernels_CM_NL_pred_exp_data.npy') # samples,timesteps,variables

    # Load experimental data
    experimental_data = pd.read_csv("Varie\\q_flutter_psf_exp.csv", sep=',', engine='python')
    AoA = np.round(experimental_data["AoA"].values,decimals=3) 
    Mach = np.round(experimental_data["Mach"].values,decimals=3) 

    cl_pred__exp_data = []
    cm_pred__exp_data = []
    q_flutter_exp_data = np.empty(AoA.shape[0])

    for i in range(kernels_CL_linear_pred_exp_data.shape[0]):

        # tau: tempo ridotto
        # t: tempo fisico
        V = Mach[i]*np.sqrt(1.116*81.49*304.2128)

        ### Linear kernels
        tau_sampling = 0.6 #0.125#0.5
        ntsteps_sampling = 18 # int(tau_sampling/(dt*2*V/chord))
        dtau = ntsteps_sampling*dt*2*V/chord
        
        it_pitch_1_deg = np.arange(0,CL_1_deg.shape[1],1)

        tau = it_pitch_1_deg*dt*2*V/chord
        time_1_deg = it_pitch_1_deg*dt

        it_train_1_deg = it_pitch_1_deg[0::ntsteps_sampling] # it: numero di iterazioni

        ntsteps = it_train_1_deg.size
        
        tau_train = it_train_1_deg*dt*2*V/chord
        time_train = it_train_1_deg*dt

        aoa = ampli_train_pitch*(1. - np.exp(-time_1_deg*otref))
        aoa_train = ampli_train_pitch*(1. - np.exp(-(time_train+2*dt*ntsteps_sampling)*otref))


        U = np.zeros((ntsteps,ntsteps))

        for k in range(ntsteps):
            U += aoa_train[k]*np.diag(np.ones(ntsteps - k),-k)
        

        cl_pred = U@kernels_CL_linear_pred_exp_data[i,:]
        cm_pred = U@kernels_CM_linear_pred_exp_data[i,:]


        ncoeff = np.arange(kernels_CL_linear_pred.shape[1])


        ### NL kernels
        ampli_train_pitch_2_deg = 2.0
        it_pitch_2_deg = np.arange(0,CL_2_deg.shape[1],1)

        # tau = it_pitch_2_deg*dt*2*V/chord
        time_2_deg = it_pitch_2_deg*dt

        it_train_2_deg = it_pitch_2_deg[0::ntsteps_sampling] # it: numero di iterazioni


        # min length
        n_train_delta = np.minimum(cl_train_1_deg.size,cl_train_2_deg.size)

        tau_train_delta = np.arange(n_train_delta)*dtau
        time_train_delta = np.arange(n_train_delta)*dt


        aoa_delta = ampli_train_pitch_2_deg*(1. - np.exp(-time_2_deg*otref))
        aoa_train_delta = ampli_train_pitch_2_deg*(1. - np.exp(-time_train_delta*otref))


        U2 = np.zeros((n_train_delta,n_train_delta))

        for k in range(n_train_delta):
            U2 += aoa_train_delta[k]*np.diag(np.ones(n_train_delta - k),-k)
        

        cl_NL_pred_exp_data = U2@kernels_CL_NL_pred_exp_data[i,:]
        cm_NL_pred_exp_data = U2@kernels_CM_NL_pred_exp_data[i,:]

        cl_pred__exp_data_temp = cl_pred*factor + cl_NL_pred_exp_data
        cl_pred__exp_data.append(cl_pred__exp_data_temp)

        cm_pred__exp_data_temp = cm_pred*factor + cm_NL_pred_exp_data
        cm_pred__exp_data.append(cm_pred__exp_data_temp)


        ###  FLUTTER ANALYSIS
        # 2DOF SS system 

        ndof = ntsteps + 4 
        ndof_aero = ntsteps
        
        AA = np.zeros((ndof,ndof))
        BB = np.zeros((ndof,ndof))

        mass= 87.91; # kg
        Iy= 3.765;   # kg m2
        kz=mass*(3.33*2*np.pi)**2;   # N m−1
        ktheta=Iy*(5.20*2*np.pi)**2; # Nm rad−1
        

        for rho_value in np.arange(0.20,2.0,0.01):
            rho = 1.1751*rho_value #1.15*0.95 #0.87
            q = 0.5*rho*V*V

            xi = 0.000

            n_dof = 2 # physical DOFs

            K_aa = np.array([[ktheta, 0],[0, kz]])
            M_aa = np.array([[Iy, 0],[0,mass]])
            C_aa = 2*xi* np.diag([np.sqrt(Iy*ktheta), np.sqrt(mass*kz)])

            Aphys = np.vstack([np.hstack([np.zeros((n_dof,n_dof)), np.eye(n_dof)]), np.hstack([-K_aa, -C_aa])])
            Bphys = np.vstack([np.hstack([np.eye(n_dof), np.zeros((n_dof,n_dof))]), np.hstack([np.zeros((n_dof,n_dof)), M_aa])])

            DT = dtau*chord/(2*V) 
            nt = 12500
            beta = 0.5 
            # ti_max = DT*nt
            ti = np.linspace(0,nt*DT,nt)
            
            # aerodynamic forces contribution by (n+1) aoa_eff        
            kernels_CL_linear_pred = kernels_CL_linear_pred_exp_data
            kernels_CM_linear_pred = kernels_CM_linear_pred_exp_data
            kernels_CL_NL_pred = kernels_CL_NL_pred_exp_data
            kernels_CM_NL_pred = kernels_CM_NL_pred_exp_data



            Aphys[n_dof,0] += q*area*chord*(kernels_CM_linear_pred[i,:])[0]*beta
            Aphys[n_dof+1,0] += q*area*(kernels_CL_linear_pred[i,:])[0]*beta
            
            Aphys[n_dof,n_dof+1] += -q*area*chord*(kernels_CM_linear_pred[i,:])[0]*beta/V
            Aphys[n_dof+1,n_dof+1] += -q*area*(kernels_CL_linear_pred[i,:])[0]*beta/V

            Aphys[n_dof,n_dof] += 0. #q*area*chord*(B@theta_cm)[0]*beta*chord/(2*V)
            Aphys[n_dof+1,n_dof] += 0. #q*area*(B@theta_cl)[0]*beta*chord/(2*V)

            # discrete time 
            Aphys_D = Bphys + DT*(1.-beta)*Aphys
            Bphys_D = Bphys - DT*beta*Aphys #

            # top left partition of states materix
            AA_D = linalg.inv(Bphys_D)@Aphys_D

            # top right partition of states matrix 
            BZ = np.zeros((2*n_dof,ndof_aero))
            BZ[n_dof,:] = q*area*chord* (beta*(kernels_CM_linear_pred[i,:]) + (1-beta)*np.hstack([kernels_CM_linear_pred[i,:],0.])[1:])
            BZ[n_dof+1,:] = q*area* (    beta*(kernels_CL_linear_pred[i,:])  + (1-beta)*np.hstack([kernels_CL_linear_pred[i,:],0.])[1:])

            BZ_D = linalg.inv(Bphys_D)@BZ

            # assembly of state matrix 
            AA[0:2*n_dof,0:2*n_dof] = AA_D 
            AA[0:2*n_dof,2*n_dof:] = DT*BZ_D

            # aoa_eff value update 
            AA[2*n_dof,0] = 1. 
            AA[2*n_dof,n_dof] = 0. #chord/(2*V) 
            AA[2*n_dof,n_dof+1] = -1./V 
            AA[2*n_dof:,2*n_dof:] = np.diag(np.ones(ndof_aero-1),-1)


            # Flutter search with eigenvalue analysis
            def flutter_detected(eigenvalues):
                return any(np.real(val) > 1 for val in eigenvalues)

            def flutter_criterion(AA):
                eigenvalues, _ = np.linalg.eig(AA)
                return flutter_detected(eigenvalues)

            # Flutter point:
            if flutter_criterion(AA):   
                q_flutter_exp_data[i]=q
                break

            else:
                continue
            

    cl_pred__exp_data = np.array(cl_pred__exp_data)
    cm_pred__exp_data = np.array(cm_pred__exp_data)

    q_flutter_exp_data = np.array(q_flutter_exp_data)

    d = {
            'Mach': Mach,
            'AoA' : AoA,
            'Q_flutter_psf': np.round(q_flutter_exp_data/47.88,decimals=3) 
            }
    ds = pd.DataFrame(data=d)


    ds.to_csv('q_flutter_NN_exp_data.csv', sep=',', index=False)


#%% Prediction 0.74 < M < 0.84 - 0 < AoA < 5
if prediction_074M084_0AoA5:
    def extract_elements(original_array):
        # Initialize an empty list to store the extracted elements
        new_vector = []

        # Iterate over the curves in steps of 60
        for i in range(0, 360, 60):
            # Extract the 6 elements from each curve and append to the new vector
            new_vector.extend(original_array[i:i+6, :])

        # Convert the new vector to a NumPy array
        new_vector = np.array(new_vector)

        return new_vector

 # Import linear and NL kernels 
    kernels_CL_linear_pred_074M084_0AoA5 = np.load('Dataset_generator\\kernels_CL_linear_pred_074M084_0AoA5.npy') # samples,timesteps,variables
    kernels_CM_linear_pred_074M084_0AoA5 = np.load('Dataset_generator\\kernels_CM_linear_pred_074M084_0AoA5.npy') # samples,timesteps,variables
    kernels_CL_NL_pred_074M084_0AoA5 = np.load('Dataset_generator\\kernels_CL_NL_pred_074M084_0AoA5.npy') # samples,timesteps,variables
    kernels_CM_NL_pred_074M084_0AoA5 = np.load('Dataset_generator\\kernels_CM_NL_pred_074M084_0AoA5.npy') # samples,timesteps,variables

    kernels_CL_linear_pred_074M084_0AoA5 = kernels_CL_linear_pred_074M084_0AoA5[::10,:]
    kernels_CM_linear_pred_074M084_0AoA5 = kernels_CM_linear_pred_074M084_0AoA5[::10,:]
    kernels_CL_NL_pred_074M084_0AoA5 = kernels_CL_NL_pred_074M084_0AoA5[::10,:]
    kernels_CM_NL_pred_074M084_0AoA5 = kernels_CM_NL_pred_074M084_0AoA5[::10,:]

    kernels_CL_linear_pred_074M084_0AoA5 = extract_elements(kernels_CL_linear_pred_074M084_0AoA5)
    kernels_CM_linear_pred_074M084_0AoA5 = extract_elements(kernels_CM_linear_pred_074M084_0AoA5)
    kernels_CL_NL_pred_074M084_0AoA5 = extract_elements(kernels_CL_NL_pred_074M084_0AoA5)
    kernels_CM_NL_pred_074M084_0AoA5 = extract_elements(kernels_CM_NL_pred_074M084_0AoA5)

    Mach = np.round(np.arange(0.74,0.86,0.02),decimals=2)
    AoA = np.round(np.arange(0,6,1),decimals=2)

    # create a 2D grid of Mach and AoA values
    Mach_grid, AoA_grid = np.meshgrid(Mach, AoA)

    # stack the Mach and AoA grids horizontally
    Mach_AoA_array = np.hstack((Mach_grid.reshape(-1, 1), AoA_grid.reshape(-1, 1)))

    Mach = Mach_AoA_array[:,0]
    AoA = Mach_AoA_array[:,1]

    cl_pred__074M084_0AoA5 = []
    cm_pred__074M084_0AoA5 = []
    q_flutter_074M084_0AoA5 = np.empty(kernels_CL_linear_pred_074M084_0AoA5.shape[0])
    flutter_search = False

    if flutter_search:
        for i in range(kernels_CL_linear_pred_074M084_0AoA5.shape[0]):

            # tau: tempo ridotto
            # t: tempo fisico
            V = Mach[i]*np.sqrt(1.116*81.49*304.2128)

            ### Linear kernels
            tau_sampling = 0.6 #0.125#0.5
            ntsteps_sampling = 18 # int(tau_sampling/(dt*2*V/chord))
            dtau = ntsteps_sampling*dt*2*V/chord
            
            it_pitch_1_deg = np.arange(0,CL_1_deg.shape[1],1)

            tau = it_pitch_1_deg*dt*2*V/chord
            time_1_deg = it_pitch_1_deg*dt

            it_train_1_deg = it_pitch_1_deg[0::ntsteps_sampling] # it: numero di iterazioni

            ntsteps = it_train_1_deg.size
            
            tau_train = it_train_1_deg*dt*2*V/chord
            time_train = it_train_1_deg*dt

            aoa = ampli_train_pitch*(1. - np.exp(-time_1_deg*otref))
            aoa_train = ampli_train_pitch*(1. - np.exp(-(time_train+2*dt*ntsteps_sampling)*otref))


            U = np.zeros((ntsteps,ntsteps))

            for k in range(ntsteps):
                U += aoa_train[k]*np.diag(np.ones(ntsteps - k),-k)
            

            cl_pred = U@kernels_CL_linear_pred_074M084_0AoA5[i,:]
            cm_pred = U@kernels_CM_linear_pred_074M084_0AoA5[i,:]


            ncoeff = np.arange(kernels_CL_linear_pred_074M084_0AoA5.shape[1])

            ### NL kernels
            ampli_train_pitch_2_deg = 2.0
            it_pitch_2_deg = np.arange(0,CL_2_deg.shape[1],1)

            # tau = it_pitch_2_deg*dt*2*V/chord
            time_2_deg = it_pitch_2_deg*dt

            it_train_2_deg = it_pitch_2_deg[0::ntsteps_sampling] # it: numero di iterazioni


            # min length
            n_train_delta = np.minimum(cl_train_1_deg.size,cl_train_2_deg.size)

            tau_train_delta = np.arange(n_train_delta)*dtau
            time_train_delta = np.arange(n_train_delta)*dt


            aoa_delta = ampli_train_pitch_2_deg*(1. - np.exp(-time_2_deg*otref))
            aoa_train_delta = ampli_train_pitch_2_deg*(1. - np.exp(-time_train_delta*otref))


            U2 = np.zeros((n_train_delta,n_train_delta))

            for k in range(n_train_delta):
                U2 += aoa_train_delta[k]*np.diag(np.ones(n_train_delta - k),-k)
            

            cl_NL_pred_074M084_0AoA5 = U2@kernels_CL_NL_pred_074M084_0AoA5[i,:]
            cm_NL_pred_074M084_0AoA5 = U2@kernels_CM_NL_pred_074M084_0AoA5[i,:]

            cl_pred__074M084_0AoA5_temp = cl_pred*factor + cl_NL_pred_074M084_0AoA5
            cl_pred__074M084_0AoA5.append(cl_pred__074M084_0AoA5_temp)

            cm_pred__074M084_0AoA5_temp = cm_pred*factor + cm_NL_pred_074M084_0AoA5
            cm_pred__074M084_0AoA5.append(cm_pred__074M084_0AoA5_temp)


            ###  FLUTTER ANALYSIS
            # 2DOF SS system 

            ndof = ntsteps + 4 
            ndof_aero = ntsteps
            
            AA = np.zeros((ndof,ndof))
            BB = np.zeros((ndof,ndof))

            mass= 87.91; # kg
            Iy= 3.765;   # kg m2
            kz=mass*(3.33*2*np.pi)**2;   # N m−1
            ktheta=Iy*(5.20*2*np.pi)**2; # Nm rad−1
            

            for rho_value in np.arange(0.0,2.5,0.01):
                rho = 1.1751*rho_value #1.15*0.95 #0.87
                q = 0.5*rho*V*V

                xi = 0.005

                n_dof = 2 # physical DOFs

                K_aa = np.array([[ktheta, 0],[0, kz]])
                M_aa = np.array([[Iy, 0],[0,mass]])
                C_aa = 2*xi* np.diag([np.sqrt(Iy*ktheta), np.sqrt(mass*kz)])

                Aphys = np.vstack([np.hstack([np.zeros((n_dof,n_dof)), np.eye(n_dof)]), np.hstack([-K_aa, -C_aa])])
                Bphys = np.vstack([np.hstack([np.eye(n_dof), np.zeros((n_dof,n_dof))]), np.hstack([np.zeros((n_dof,n_dof)), M_aa])])

                DT = dtau*chord/(2*V) 
                nt = 12500
                # ti_max = DT*nt
                ti = np.linspace(0,nt*DT,nt)
                
                # aerodynamic forces contribution by (n+1) aoa_eff

                beta = 0.5 
                

                kernels_CL_linear_pred = kernels_CL_linear_pred_074M084_0AoA5
                kernels_CM_linear_pred = kernels_CM_linear_pred_074M084_0AoA5
                kernels_CL_NL_pred = kernels_CL_NL_pred_074M084_0AoA5
                kernels_CM_NL_pred = kernels_CM_NL_pred_074M084_0AoA5



                Aphys[n_dof,0] += q*area*chord*(kernels_CM_linear_pred[i,:])[0]*beta
                Aphys[n_dof+1,0] += q*area*(kernels_CL_linear_pred[i,:])[0]*beta
                
                Aphys[n_dof,n_dof+1] += -q*area*chord*(kernels_CM_linear_pred[i,:])[0]*beta/V
                Aphys[n_dof+1,n_dof+1] += -q*area*(kernels_CL_linear_pred[i,:])[0]*beta/V

                Aphys[n_dof,n_dof] += 0. #q*area*chord*(B@theta_cm)[0]*beta*chord/(2*V)
                Aphys[n_dof+1,n_dof] += 0. #q*area*(B@theta_cl)[0]*beta*chord/(2*V)

                # discrete time 
                Aphys_D = Bphys + DT*(1.-beta)*Aphys
                Bphys_D = Bphys - DT*beta*Aphys #

                # top left partition of states materix
                AA_D = linalg.inv(Bphys_D)@Aphys_D

                # top right partition of states matrix 
                BZ = np.zeros((2*n_dof,ndof_aero))
                BZ[n_dof,:] = q*area*chord* (beta*(kernels_CM_linear_pred[i,:]) + (1-beta)*np.hstack([kernels_CM_linear_pred[i,:],0.])[1:])
                BZ[n_dof+1,:] = q*area* (    beta*(kernels_CL_linear_pred[i,:])  + (1-beta)*np.hstack([kernels_CL_linear_pred[i,:],0.])[1:])

                BZ_D = linalg.inv(Bphys_D)@BZ

                # assembly of state matrix 
                AA[0:2*n_dof,0:2*n_dof] = AA_D 
                AA[0:2*n_dof,2*n_dof:] = DT*BZ_D

                # aoa_eff value update 
                AA[2*n_dof,0] = 1. 
                AA[2*n_dof,n_dof] = 0. #chord/(2*V) 
                AA[2*n_dof,n_dof+1] = -1./V 
                AA[2*n_dof:,2*n_dof:] = np.diag(np.ones(ndof_aero-1),-1)


                # Flutter search with eigenvalue analysis
                def flutter_detected(eigenvalues):
                    return any(np.real(val) > 1 for val in eigenvalues)

                def flutter_criterion(AA):
                    eigenvalues, _ = np.linalg.eig(AA)
                    return flutter_detected(eigenvalues)

                # Flutter point:
                if flutter_criterion(AA):
                    # print('\n Prediction - M = '+str(Mach[i])+' - AoA = '+str(AoA[i])+ ' Flutter point found at q_f: %0.3f [psf]' %(q/47.88))
                    q_flutter_074M084_0AoA5[i]=q
                    break

                else:
                    continue
                

        cl_pred__074M084_0AoA5 = np.array(cl_pred__074M084_0AoA5)
        cm_pred__074M084_0AoA5 = np.array(cm_pred__074M084_0AoA5)

        # Save data to csv file
        d = {
                'Mach': Mach,
                'AoA' : AoA,
                'Q_flutter_psf': np.round(q_flutter_074M084_0AoA5/47.88,decimals=3) 
                }
        ds = pd.DataFrame(data=d)


        ds.to_csv('q_flutter_NN_exp_data_074M084_0AoA5_xi005.csv', sep=',', index=False)

    # Load flutter boundary
    df=pd.read_csv('q_flutter_NN_exp_data_074M084_0AoA5_xi005.csv')

    q_flutter_074M084_0AoA5_xi005 = df['Q_flutter_psf'].values.reshape(6,6)

    df1=pd.read_csv('q_flutter_NN_exp_data_074M084_0AoA5.csv')

    q_flutter_074M084_0AoA5 = df1['Q_flutter_psf'].values.reshape(60,60)
    q_flutter_074M084_0AoA5 = q_flutter_074M084_0AoA5[::10,::10]

    Mach = np.round(np.arange(0.74,0.86,0.02),decimals=2)
    AoA = np.round(np.arange(0,6,1),decimals=2)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Plot q flutter for 0.74 < M < 0.84 - 0 < AoA < 5 
    plt.rcParams.update({'font.size': 16})
    fig, (ax1,ax2) = plt.subplots(1,2)
    fig.set_size_inches(18.5, 10.5)
    ax1.set_title('Flutter speed at constant Mach')
    for i in range(6):
        ax1.plot(AoA,q_flutter_074M084_0AoA5[:,i],'.-',color=colors[i],linewidth=2,markersize = 20,label= 'Mach = '+str(Mach[i]))
        ax1.plot(AoA,q_flutter_074M084_0AoA5_xi005[:,i],'.--',color=colors[i],linewidth=2)

    ax1.set_xlabel(r'AoA [deg]',fontsize =16)
    ax1.set_ylabel(r'Q flutter [psf]',fontsize =16)
    ax1.legend(loc='lower left',fontsize =16)
    ax1.grid()

    ax2.set_title('Flutter speed at constant AoA')
    for i in range(6):
        ax2.plot(Mach,q_flutter_074M084_0AoA5[i,:],'.-',color=colors[i],linewidth=2,markersize = 20,label= 'AoA = '+str(AoA[i]))
        ax2.plot(Mach,q_flutter_074M084_0AoA5_xi005[i,:],'.--',color=colors[i],linewidth=2)
    ax2.set_xlabel(r'Mach',fontsize =16)
    ax2.set_ylabel(r'Q flutter [psf]',fontsize =16)
    ax2.legend(loc='lower left',fontsize =16)
    ax2.grid()

    plt.savefig('Photo\\'+ 'flutter_speed_074M084_0AoA5_xi005.png', bbox_inches='tight')
    
    plt.show()


#%% Prediction M = 0.70 -  AoA = 5
if prediction_M070_AOA5:
    # import kernels linear and NL M=0.70 AoA=5
    kernels_CL_linear_pred_M070_AoA5 = np.load('Dataset_generator\\kernels_CL_linear_pred_M070_AoA5.npy') # samples,timesteps,variables
    kernels_CM_linear_pred_M070_AoA5 = np.load('Dataset_generator\\kernels_CM_linear_pred_M070_AoA5.npy') # samples,timesteps,variables
    kernels_CL_NL_pred_M070_AoA5 = np.load('Dataset_generator\\kernels_CL_NL_pred_M070_AoA5.npy') # samples,timesteps,variables
    kernels_CM_NL_pred_M070_AoA5 = np.load('Dataset_generator\\kernels_CM_NL_pred_M070_AoA5.npy') # samples,timesteps,variables

    Mach = 0.70
    AoA = 5.0

    ## Prediction M = 0.70 - alpha_mean = 5 deg - alpha_ampl = 1.03 deg - f = 10 Hz - q = 170 psf 

    dataset_CL0_CM0_M070_AoA5 = np.load('Dataset_generator\\kernels_CL0_CM0_pred_M070_AoA5.npy') # samples,timesteps,variables

    # tau: tempo ridotto
    # t: tempo fisico
    V = Mach*np.sqrt(1.116*81.49*304.2128)

    ### Linear kernels
    tau_sampling = 0.6 #0.125#0.5
    ntsteps_sampling = 18 # int(tau_sampling/(dt*2*V/chord))
    dtau = ntsteps_sampling*dt*2*V/chord
    it_pitch_1_deg = np.arange(0,CL_1_deg.shape[1],1)
    tau = it_pitch_1_deg*dt*2*V/chord
    time_1_deg = it_pitch_1_deg*dt
    it_train_1_deg = it_pitch_1_deg[0::ntsteps_sampling] # it: numero di iterazioni
    ntsteps = it_train_1_deg.size
    tau_train = it_train_1_deg*dt*2*V/chord
    time_train = it_train_1_deg*dt
    aoa_train = ampli_train_pitch*(1. - np.exp(-(time_train+2*dt*ntsteps_sampling)*otref))


    U = np.zeros((ntsteps,ntsteps))

    for k in range(ntsteps):
        U += aoa_train[k]*np.diag(np.ones(ntsteps - k),-k)
    

    cl_pred = U@kernels_CL_linear_pred_M070_AoA5[0,:]
    cm_pred = U@kernels_CM_linear_pred_M070_AoA5[0,:]


    ### NL kernels
    ampli_train_pitch_2_deg = 2.0
    it_pitch_2_deg = np.arange(0,CL_2_deg.shape[1],1)
    # tau = it_pitch_2_deg*dt*2*V/chord
    time_2_deg = it_pitch_2_deg*dt
    it_train_2_deg = it_pitch_2_deg[0::ntsteps_sampling] # it: numero di iterazioni


    # min length
    n_train_delta = np.minimum(cl_train_1_deg.size,cl_train_2_deg.size)

    tau_train_delta = np.arange(n_train_delta)*dtau
    time_train_delta = np.arange(n_train_delta)*dt


    aoa_delta = ampli_train_pitch_2_deg*(1. - np.exp(-time_2_deg*otref))
    aoa_train_delta = ampli_train_pitch_2_deg*(1. - np.exp(-time_train_delta*otref))


    U2 = np.zeros((n_train_delta,n_train_delta))

    for k in range(n_train_delta):
        U2 += aoa_train_delta[k]*np.diag(np.ones(n_train_delta - k),-k)
    

    cl_NL_pred_M070_AoA5 = U2@kernels_CL_NL_pred_M070_AoA5[0,:]
    cm_NL_pred_M070_AoA5 = U2@kernels_CM_NL_pred_M070_AoA5[0,:]

    cl_pred__M070_AoA5 = cl_pred*factor + cl_NL_pred_M070_AoA5

    cm_pred__M070_AoA5 = cm_pred*factor + cm_NL_pred_M070_AoA5

    # Load CFD data
    ds = pd.read_csv("BSCW_step_2_deg\\M070_AoA5_2_deg\\history_00002.dat", sep=',', engine='python',skiprows=1)
    ds.columns = ds.columns.str.strip()
    ds.columns = ds.columns.str.replace('"', '')
    dataset_aero = ds.drop_duplicates('Time_Iter',keep='last')

    data = {'CL': dataset_aero.CL[:900], 'CMy': dataset_aero.CMy[:900]}

    df = pd.DataFrame(data=data)

    # 2 deg
    ampli_train_pitch_2_deg = 2.0
    it_pitch_2_deg = np.arange(0,CL_2_deg.shape[1],1)
    cl_pitch_2_deg = df.CL -  0.59603458972 
    cm_pitch_2_deg = df.CMy -0.07652068245

    # apply a 3-pole lowpass filter at 0.1x Nyquist frequency
    b, a = scipy.signal.butter(1, 0.02)
    b1, a1 = scipy.signal.butter(2, 0.007)
    cl_pitch_2_deg = scipy.signal.filtfilt(b, a, cl_pitch_2_deg)
    cm_pitch_2_deg = scipy.signal.filtfilt(b1, a1, cm_pitch_2_deg)

    # tau = it_pitch_2_deg*dt*2*V/chord
    time_2_deg = it_pitch_2_deg*dt

    it_train_2_deg = it_pitch_2_deg[0::ntsteps_sampling] # it: numero di iterazioni
    cl_train_2_deg = cl_pitch_2_deg[0::ntsteps_sampling]
    cm_train_2_deg = cm_pitch_2_deg[0::ntsteps_sampling]


    plt.rcParams.update({'font.size': 16})
    fig, ((ax, ax1)) = plt.subplots(1, 2)
    fig.set_size_inches(18.5, 5.25)
    fig.suptitle(' Prediction - M = 0.70 - AoA = 5 ')

    ax.plot(tau_train[::2],cl_train_2_deg[::2],linestyle = 'None',marker='s', markersize=8, markerfacecolor='gray', markeredgecolor='black',label='CFD') 
    ax.plot(tau_train_delta,cl_pred__M070_AoA5,'k',linewidth=2,label='Volterra - FCNN')
    ax.set_xlabel(r'$\tau$',fontsize=16)
    ax.set_ylabel(r'$C_L - C_{L0}$',fontsize=16)
    ax.legend(loc='lower right',fontsize=16)
    ax.grid()

    plt.subplots_adjust(wspace=0.3)

    ax1.plot(tau_train[::2],cm_train_2_deg[::2],linestyle = 'None',marker='s', markersize=8, markerfacecolor='gray', markeredgecolor='black',label='CFD')
    ax1.plot(tau_train_delta,cm_pred__M070_AoA5,'k',linewidth=2,label='Volterra - FCNN')
    ax1.set_xlabel(r'$\tau$',fontsize=16)
    ax1.set_ylabel(r'$C_M - C_{M0}$',fontsize=16)
    ax1.legend(loc='lower right',fontsize=16)
    ax1.grid()
    plt.savefig('Photo\\'+ 'step_M070_AoA5.png', bbox_inches='tight')

    plt.show()
    plt.close()


#%% Prediction Harmonic signal M = 0.70 - AoA = 5

if prediction_harmonic_signal:

    # Import linear and NL kernels NN
    kernels_CL_linear_pred_M070_AoA5 = np.load('Dataset_generator\\kernels_CL_linear_pred_M070_AoA5.npy') # samples,timesteps,variables
    kernels_CM_linear_pred_M070_AoA5 = np.load('Dataset_generator\\kernels_CM_linear_pred_M070_AoA5.npy') # samples,timesteps,variables
    kernels_CL_NL_pred_M070_AoA5 = np.load('Dataset_generator\\kernels_CL_NL_pred_M070_AoA5.npy') # samples,timesteps,variables
    kernels_CM_NL_pred_M070_AoA5 = np.load('Dataset_generator\\kernels_CM_NL_pred_M070_AoA5.npy') # samples,timesteps,variables

    # Import linear and NL kernels GP
    kernels_CL_linear_pred_GP_M070_AoA5 = np.load('Dataset_generator\\kernels_CL_linear_pred_GP_M070_AoA5.npy') # samples,timesteps,variables
    kernels_CM_linear_pred_GP_M070_AoA5 = np.load('Dataset_generator\\kernels_CM_linear_pred_GP_M070_AoA5.npy') # samples,timesteps,variables
    kernels_CL_NL_pred_GP_M070_AoA5 = np.load('Dataset_generator\\kernels_CL_NL_pred_GP_M070_AoA5.npy') # samples,timesteps,variables
    kernels_CM_NL_pred_GP_M070_AoA5 = np.load('Dataset_generator\\kernels_CM_NL_pred_GP_M070_AoA5.npy') # samples,timesteps,variables
    # import variance kernels GP
    variance_kernels_CL_linear_pred_GP_M070_AoA5 = np.load('Dataset_generator\\variance_kernels_CL_linear_pred_GP_M070_AoA5.npy') # samples,timesteps,variables
    variance_kernels_CM_linear_pred_GP_M070_AoA5 = np.load('Dataset_generator\\variance_kernels_CM_linear_pred_GP_M070_AoA5.npy') # samples,timesteps,variables
    variance_kernels_CL_NL_pred_GP_M070_AoA5 = np.load('Dataset_generator\\variance_kernels_CL_NL_pred_GP_M070_AoA5.npy') # samples,timesteps,variables
    variance_kernels_CM_NL_pred_GP_M070_AoA5 = np.load('Dataset_generator\\variance_kernels_CM_NL_pred_GP_M070_AoA5.npy') # samples,timesteps,variables

    Mach = 0.70
    AoA = 5.0

    ### Prediction M = 0.70 - alpha_mean = 5 deg - alpha_ampl = 1.03 deg - f = 10 Hz - q = 170 psf 
    dataset_CL0_CM0_M070_AoA5 = np.load('Dataset_generator\\kernels_CL0_CM0_pred_M070_AoA5.npy') # samples,timesteps,variables

    i = 0 # 10 Hz
    ds = pd.read_csv("BSCW_armonic\\M070_10Hz\\history.dat", sep=',', engine='python',skiprows=1)
    ds.columns = ds.columns.str.strip()
    ds.columns = ds.columns.str.replace('"', '')
    dataset_aero = ds.drop_duplicates('Time_Iter',keep='last')

    ds1 = pd.read_csv("BSCW_armonic\\M070_10Hz\\history_00400.dat", sep=',', engine='python',skiprows=1)
    ds1.columns = ds1.columns.str.strip()
    ds1.columns = ds1.columns.str.replace('"', '')
    dataset_aero1 = ds1.drop_duplicates('Time_Iter',keep='last')
    CL = pd.concat([dataset_aero['CL'], dataset_aero1['CL']])
    CM = pd.concat([dataset_aero['CMy'], dataset_aero1['CMy']])
    data = {'CL': CL, 'CMy': CM}
    df = pd.DataFrame(data=data)

    # cmo_new = dataset_CL0_CM0_M070_AoA5[1,1] - (chord*0.5-chord*0.7)*dataset_CL0_CM0_M070_AoA5[0,0]
    CL_CFD = df.CL - dataset_CL0_CM0_M070_AoA5[0,0]
    CM_CFD = df.CMy - (chord*0.5-chord*0.359)* df.CL + dataset_CL0_CM0_M070_AoA5[1,1]

    # CL_CFD_t = CL_CFD[251:350]
    # CM_CFD_t = CM_CFD[251:350]
    # CL_CFD = np.concatenate((CL_CFD[:251],CL_CFD_t,CL_CFD_t,CL_CFD_t,CL_CFD_t))
    # CM_CFD = np.concatenate((CM_CFD[:251],CM_CFD_t,CM_CFD_t,CM_CFD_t,CM_CFD_t))
    ntsteps = (CL_CFD).size
    V = Mach*np.sqrt(1.116*81.49*304.2128)
    ampli_pitch = 1.03*np.pi/180.
    freq = 10.1 # Hz

    # Timestep CFD
    dt = 1e-3 
    dt = dt*V/chord
    # time = np.arange(0,ntsteps*dt,dt)
    time = np.arange(1,ntsteps*dt+1,dt)

    # Timestep Volterra
    tau_sampling = 0.6 
    dtau = tau_sampling #tau_sampling*chord / (2*V)
    ti_max = dtau*ntsteps
    ti = np.linspace(0,tau_sampling*ntsteps,ntsteps)

    k = 2* np.pi * freq * chord / V
    aoa_input = -ampli_pitch*np.sin( k*ti )#- 0.09549296585513721 / np.pi * 180 ) 
    aoa_input_cfd = -ampli_pitch*np.sin( k*time )

    # NN
    number_states = kernels_CL_linear_pred_M070_AoA5.shape[1]
    ndof = number_states 
    q = np.zeros((ndof,ntsteps))
    cl_pred =  np.zeros(ntsteps)
    cm_pred =  np.zeros(ntsteps)
    AA = np.diag(np.ones(ndof-1),-1)
    BB = np.zeros((ndof))
    BB[0] = 1.
    CC_CL = kernels_CL_linear_pred_M070_AoA5[i,:] + kernels_CL_NL_pred_M070_AoA5[i,:]
    CC_CM = kernels_CM_linear_pred_M070_AoA5[i,:] + kernels_CM_NL_pred_M070_AoA5[i,:]
    for it in range(1,ntsteps):
        q[:,it] = AA@q[:,it-1] + BB*aoa_input[it] 
        cl_pred[it] = CC_CL@q[:,it]
        cm_pred[it] = CC_CM@q[:,it]


    # Interpolate the signal
    f_cl = interp1d(ti, cl_pred, kind='cubic', fill_value="extrapolate")
    cl_pred_interp = f_cl(time)

    f_cm = interp1d(ti, cm_pred, kind='cubic', fill_value="extrapolate")
    cm_pred_interp = f_cm(time)


    # GP
    confidence_interval = 0.68 # 1 sigma = 68% 
    z = scipy.stats.norm.ppf(1 - (1 - confidence_interval) / 2)


    number_states_GP = kernels_CL_linear_pred_M070_AoA5.shape[1]
    ndof_GP = number_states_GP
    q_GP = np.zeros((ndof_GP,ntsteps))
    cl_pred_GP =  np.zeros(ntsteps)
    cm_pred_GP =  np.zeros(ntsteps)
    cl_pred_GP_upper =  np.zeros(ntsteps)
    cl_pred_GP_lower =  np.zeros(ntsteps)
    cm_pred_GP_upper =  np.zeros(ntsteps)
    cm_pred_GP_lower =  np.zeros(ntsteps)
    
    AA_GP = np.diag(np.ones(ndof_GP-1),-1)
    BB_GP = np.zeros((ndof_GP))
    BB_GP[0] = 1.
    CC_CL_GP = kernels_CL_linear_pred_GP_M070_AoA5[i,:] + kernels_CL_NL_pred_GP_M070_AoA5[i,:]
    CC_CM_GP = kernels_CM_linear_pred_GP_M070_AoA5[i,:] + kernels_CM_NL_pred_GP_M070_AoA5[i,:]
    CC_CL_GP_upper = CC_CL_GP + z * np.sqrt(variance_kernels_CL_linear_pred_GP_M070_AoA5[i,:] + variance_kernels_CL_linear_pred_GP_M070_AoA5[i,:])
    CC_CL_GP_lower = CC_CL_GP - z * np.sqrt(variance_kernels_CL_linear_pred_GP_M070_AoA5[i,:] + variance_kernels_CL_NL_pred_GP_M070_AoA5[i,:])
    CC_CM_GP_upper = CC_CM_GP + z * np.sqrt(variance_kernels_CM_linear_pred_GP_M070_AoA5[i,:] + variance_kernels_CM_linear_pred_GP_M070_AoA5[i,:])
    CC_CM_GP_lower = CC_CM_GP - z * np.sqrt(variance_kernels_CM_linear_pred_GP_M070_AoA5[i,:] + variance_kernels_CM_NL_pred_GP_M070_AoA5[i,:])


    for it in range(1,ntsteps):
        q_GP[:,it] = AA_GP@q_GP[:,it-1] + BB_GP*aoa_input[it] 
        cl_pred_GP[it] = CC_CL_GP@q_GP[:,it]
        cm_pred_GP[it] = CC_CM_GP@q_GP[:,it]
        cl_pred_GP_upper[it] =  CC_CL_GP_upper@q_GP[:,it]
        cl_pred_GP_lower[it] =  CC_CL_GP_lower@q_GP[:,it]
        cm_pred_GP_upper[it] =  CC_CM_GP_upper@q_GP[:,it]
        cm_pred_GP_lower[it] =  CC_CM_GP_lower@q_GP[:,it]


    # Interpolate the signal
    f_cl_GP = interp1d(ti, cl_pred_GP, kind='cubic', fill_value="extrapolate")
    cl_pred_interp_GP = f_cl_GP(time)

    f_cm_GP = interp1d(ti, cm_pred_GP, kind='cubic', fill_value="extrapolate")
    cm_pred_interp_GP = f_cm_GP(time)

    f_cl_GP_upper = interp1d(ti, cl_pred_GP_upper, kind='cubic', fill_value="extrapolate")
    cl_pred_GP_upper = f_cl_GP_upper(time)

    f_cl_GP_lower = interp1d(ti, cl_pred_GP_lower, kind='cubic', fill_value="extrapolate")
    cl_pred_GP_lower = f_cl_GP_lower(time)

    f_cm_GP_upper = interp1d(ti, cm_pred_GP_upper, kind='cubic', fill_value="extrapolate")
    cm_pred_GP_upper = f_cm_GP_upper(time)

    f_cm_GP_lower = interp1d(ti, cm_pred_GP_lower, kind='cubic', fill_value="extrapolate")
    cm_pred_GP_lower = f_cm_GP_lower(time)


    # # Calculate the complex representation of the signals
    # z1 = CL_CFD + 1j*np.zeros(len(CL_CFD))
    # z2 = cl_pred_interp + 1j*np.zeros(len(cl_pred_interp))

    # # Compute the phase difference between the two signals
    # phase_diff = np.angle(z2 / z1)
    # print(phase_diff[0])

        
    # # Calculate the cross-correlation function between the two signals
    # corr = correlate(CM_CFD, cm_pred_interp, mode='same')

    # # Find the index of the maximum value in the cross-correlation function
    # max_idx = np.argmax(corr)

    # # Calculate the phase difference between the two signals
    # phase_diff = (max_idx - len(corr)/2) / (2*np.pi*5)
    # # Print the result in degrees
    # print('Phase difference: {:.2f} degrees'.format(np.rad2deg(phase_diff.item())))

    # # Prediction
    # plt.rcParams.update({'font.size': 16})
    # fig, ((ax2, ax3)) = plt.subplots(2, 1)
    # fig.set_size_inches(18.5, 10.5)
    # fig.suptitle(' Prediction - M = 0.70 - AoA_0 = 5 - Pitch_ampl = 1.03 - f = 10 Hz ')

    # ax2.plot(ti,CL_CFD,'k',linewidth=2,label='CFD')
    # # ax2.plot(tau_train,cltest_check,'r',linewidth=2,label='Volterra')
    # # ax2.plot(ti,cl_pred,'g',linewidth=2,label='FCNN')
    # ax2.scatter(ti[::8],cl_pred_interp[::8], marker='o', facecolors='black', edgecolors='black',label='Volterra-FCNN')
    # ax2.scatter(ti[::8],cl_pred_interp_GP[::8], marker='o', facecolors='red', edgecolors='black',label='Volterra-GP - Mean')
    # ax2.fill_between(
    #     ti[::8],
    #         cl_pred_GP_lower[::8],
    #         cl_pred_GP_upper[::8],
    #         alpha=0.5,
    #         # label=r"60% confidence interval GP",
    #         label=r"GP - $\pm$ 1$\sigma$",
    #     )
    # ax2.set_xlabel(r'$\tau$',fontsize=16)
    # ax2.set_ylabel(r'$C_L - C_{L0}$',fontsize=16)
    # ax2.legend(loc='lower right',fontsize=16)
    # ax2.grid()

    # ax3.plot(ti,CM_CFD,'k',linewidth=2,label='CFD')
    # # ax3.plot(tau_train,cmtest_check,'r',linewidth=2,label='Volterra')
    # # ax3.plot(ti,cm_pred,'g',linewidth=2,label='Volterra-FCNN')
    # ax3.scatter(ti[::8],cm_pred_interp[::8], marker='o', facecolors='black', edgecolors='black',label='Volterra-FCNN')
    # ax3.scatter(ti[::8],cm_pred_interp_GP[::8], marker='o', facecolors='red', edgecolors='black',label='Volterra-GP - Mean')
    # ax3.fill_between(
    #     ti[::8],
    #         cm_pred_GP_lower[::8],
    #         cm_pred_GP_upper[::8],
    #         alpha=0.5,
    #         # label=r"60% confidence interval GP",
    #         label=r"GP - $\pm$ 1$\sigma$",
    #         )
    # ax3.set_xlabel(r'$\tau$',fontsize=16)
    # ax3.set_ylabel(r'$C_M - C_{M0}$',fontsize=16)
    # ax3.legend(loc='lower right',fontsize=16)
    # ax3.grid()
    # # plt.savefig('Photo\\'+ 'harmonic_signal_10hz.png', bbox_inches='tight')

    # plt.show()
    # plt.close()

    # Prediction
    plt.rcParams.update({'font.size': 16})
    fig, ((ax2, ax3)) = plt.subplots(2, 1)
    fig.set_size_inches(9.25, 10.5)

    fig.suptitle(' Prediction - M = 0.70 - AoA_0 = 5 - Pitch_ampl = 1.03 - f = 10 Hz ')

    ax2.plot(aoa_input_cfd[351:450:2]*180/np.pi,CL_CFD[351:450:2],linestyle = 'None',marker='s', markersize=8, markerfacecolor='gray', markeredgecolor='black',label='CFD')
    # ax2.plot(tau_train,cltest_check,'r',linewidth=2,label='Volterra')
    # ax2.plot(ti,cl_pred,'g',linewidth=2,label='FCNN')
    ax2.plot(aoa_input[251:]*180/np.pi,cl_pred[251:],'k',linewidth=2,label='Volterra - FCNN')
    ax2.set_xlabel(r'$\alpha [deg]$')
    ax2.set_ylabel(r'$C_L - C_{L0}$')
    ax2.legend(loc='lower right')
    ax2.grid()

    ax3.plot(aoa_input_cfd[351:450:2]*180/np.pi,CM_CFD[351:450:2],linestyle = 'None',marker='s', markersize=8, markerfacecolor='gray', markeredgecolor='black',label='CFD')
    # ax3.plot(tau_train,cmtest_check,'r',linewidth=2,label='Volterra')
    # ax3.plot(ti,cm_pred,'g',linewidth=2,label='FCNN')
    ax3.plot(aoa_input[251:]*180/np.pi,cm_pred[251:],'k',linewidth=2,label='Volterra - FCNN')
    ax3.set_xlabel(r'$\alpha [deg]$')
    ax3.set_ylabel(r'$C_M - C_{M0}$')
    ax3.legend(loc='lower right')
    ax3.grid()
    plt.savefig('Photo\\'+ 'harmonic_signal_hysteresis_10hz.png', bbox_inches='tight')

    plt.show()
    plt.close()


    i = 1 # 20 Hz
    ds = pd.read_csv("BSCW_armonic\\M070_20Hz\\history.dat", sep=',', engine='python',skiprows=1)
    ds.columns = ds.columns.str.strip()
    ds.columns = ds.columns.str.replace('"', '')
    dataset_aero = ds.drop_duplicates('Time_Iter',keep='last')

    data = {'CL': dataset_aero.CL, 'CMy': dataset_aero.CMy}
    df = pd.DataFrame(data=data)

    CL_CFD = df.CL - dataset_CL0_CM0_M070_AoA5[0,0]
    CM_CFD = df.CMy - (chord*0.5-chord*0.359)* df.CL+ dataset_CL0_CM0_M070_AoA5[1,1]

    # CL_CFD = np.concatenate((CL_CFD,CL_CFD,CL_CFD))
    # CM_CFD = np.concatenate((CM_CFD,CM_CFD,CM_CFD))
    ntsteps = (CL_CFD).size
    V = Mach*np.sqrt(1.116*81.49*304.2128)
    ampli_pitch = 0.50*np.pi/180.
    freq = 20.0 # Hz

    # Timestep CFD
    dt = 5e-4
    dt = dt*V/chord
    # time = np.arange(0,ntsteps*dt,dt)
    time = np.arange(1,ntsteps*dt+1,dt)

    # Timestep Volterra
    tau_sampling = 0.6 
    dtau = tau_sampling #tau_sampling*chord / (2*V)
    ti_max = dtau*ntsteps
    ti = np.linspace(0,tau_sampling*ntsteps,ntsteps)

    k = 2* np.pi * freq * chord / V
    aoa_input = - ampli_pitch*np.sin( k*ti ) 
    aoa_input_cfd = -ampli_pitch*np.sin( k*time )

    number_states = kernels_CL_linear_pred_M070_AoA5.shape[1]
    ndof = number_states 
    q = np.zeros((ndof,ntsteps))
    cl_pred =  np.zeros(ntsteps)
    cm_pred =  np.zeros(ntsteps)
    AA = np.diag(np.ones(ndof-1),-1)
    BB = np.zeros((ndof))
    BB[0] = 1.
    CC_CL = kernels_CL_linear_pred_M070_AoA5[i,:] + kernels_CL_NL_pred_M070_AoA5[i,:]
    CC_CM = kernels_CM_linear_pred_M070_AoA5[i,:] + kernels_CM_NL_pred_M070_AoA5[i,:]
    for it in range(1,ntsteps):
        q[:,it] = AA@q[:,it-1] + BB*aoa_input[it] 
        cl_pred[it] = CC_CL@q[:,it]
        cm_pred[it] = CC_CM@q[:,it]


    # Interpolate the signal
    f_cl = interp1d(ti, cl_pred, kind='cubic')
    cl_pred_interp = f_cl(time)

    f_cm = interp1d(ti, cm_pred, kind='cubic')
    cm_pred_interp = f_cm(time)



    # GP
    confidence_interval = 0.68 # 1 sigma = 68% 
    z = scipy.stats.norm.ppf(1 - (1 - confidence_interval) / 2)


    number_states_GP = kernels_CL_linear_pred_M070_AoA5.shape[1]
    ndof_GP = number_states_GP
    q_GP = np.zeros((ndof_GP,ntsteps))
    cl_pred_GP =  np.zeros(ntsteps)
    cm_pred_GP =  np.zeros(ntsteps)
    cl_pred_GP_upper =  np.zeros(ntsteps)
    cl_pred_GP_lower =  np.zeros(ntsteps)
    cm_pred_GP_upper =  np.zeros(ntsteps)
    cm_pred_GP_lower =  np.zeros(ntsteps)
    
    AA_GP = np.diag(np.ones(ndof_GP-1),-1)
    BB_GP = np.zeros((ndof_GP))
    BB_GP[0] = 1.
    CC_CL_GP = kernels_CL_linear_pred_GP_M070_AoA5[i,:] + kernels_CL_NL_pred_GP_M070_AoA5[i,:]
    CC_CM_GP = kernels_CM_linear_pred_GP_M070_AoA5[i,:] + kernels_CM_NL_pred_GP_M070_AoA5[i,:]
    CC_CL_GP_upper = CC_CL_GP + z * np.sqrt(variance_kernels_CL_linear_pred_GP_M070_AoA5[i,:] + variance_kernels_CL_linear_pred_GP_M070_AoA5[i,:])
    CC_CL_GP_lower = CC_CL_GP - z * np.sqrt(variance_kernels_CL_linear_pred_GP_M070_AoA5[i,:] + variance_kernels_CL_NL_pred_GP_M070_AoA5[i,:])
    CC_CM_GP_upper = CC_CM_GP + z * np.sqrt(variance_kernels_CM_linear_pred_GP_M070_AoA5[i,:] + variance_kernels_CM_linear_pred_GP_M070_AoA5[i,:])
    CC_CM_GP_lower = CC_CM_GP - z * np.sqrt(variance_kernels_CM_linear_pred_GP_M070_AoA5[i,:] + variance_kernels_CM_NL_pred_GP_M070_AoA5[i,:])


    for it in range(1,ntsteps):
        q_GP[:,it] = AA_GP@q_GP[:,it-1] + BB_GP*aoa_input[it] 
        cl_pred_GP[it] = CC_CL_GP@q_GP[:,it]
        cm_pred_GP[it] = CC_CM_GP@q_GP[:,it]
        cl_pred_GP_upper[it] =  CC_CL_GP_upper@q_GP[:,it]
        cl_pred_GP_lower[it] =  CC_CL_GP_lower@q_GP[:,it]
        cm_pred_GP_upper[it] =  CC_CM_GP_upper@q_GP[:,it]
        cm_pred_GP_lower[it] =  CC_CM_GP_lower@q_GP[:,it]


    # Interpolate the signal
    f_cl_GP = interp1d(ti, cl_pred_GP, kind='cubic')
    cl_pred_interp_GP = f_cl_GP(time)

    f_cm_GP = interp1d(ti, cm_pred_GP, kind='cubic')
    cm_pred_interp_GP = f_cm_GP(time)

    f_cl_GP_upper = interp1d(ti, cl_pred_GP_upper, kind='cubic')
    cl_pred_GP_upper = f_cl_GP_upper(time)

    f_cl_GP_lower = interp1d(ti, cl_pred_GP_lower, kind='cubic')
    cl_pred_GP_lower = f_cl_GP_lower(time)

    f_cm_GP_upper = interp1d(ti, cm_pred_GP_upper, kind='cubic')
    cm_pred_GP_upper = f_cm_GP_upper(time)

    f_cm_GP_lower = interp1d(ti, cm_pred_GP_lower, kind='cubic')
    cm_pred_GP_lower = f_cm_GP_lower(time)


    # # Calculate the complex representation of the signals
    # z1 = CL_CFD + 1j*np.zeros(len(CL_CFD))
    # z2 = cl_pred_interp + 1j*np.zeros(len(cl_pred_interp))

    # # Compute the phase difference between the two signals
    # phase_diff = np.angle(z2 / z1)
    # print(phase_diff[0])

        
    # # Calculate the cross-correlation function between the two signals
    # corr = correlate(CM_CFD, cm_pred_interp, mode='same')

    # # Find the index of the maximum value in the cross-correlation function
    # max_idx = np.argmax(corr)

    # # Calculate the phase difference between the two signals
    # phase_diff = (max_idx - len(corr)/2) / (2*np.pi*5)
    # # Print the result in degrees
    # print('Phase difference: {:.2f} degrees'.format(np.rad2deg(phase_diff.item())))



    # # Prediction
    # plt.rcParams.update({'font.size': 16})
    # fig, ((ax2, ax3)) = plt.subplots(2, 1)
    # fig.set_size_inches(18.5, 10.5)

    # fig.suptitle(' Prediction - M = 0.70 - AoA_0 = 5 - Pitch_ampl = 0.50 - f = 20 Hz ')

    # ax2.plot(ti,CL_CFD,'k',linewidth=2,label='CFD')
    # # ax2.plot(tau_train,cltest_check,'r',linewidth=2,label='Volterra')
    # # ax2.plot(ti,cl_pred,'g',linewidth=2,label='FCNN')
    # ax2.scatter(ti[::8],cl_pred_interp[::8], marker='o', facecolors='black', edgecolors='black',label='Volterra-FCNN')
    # ax2.scatter(ti[::8],cl_pred_interp_GP[::8], marker='o', facecolors='red', edgecolors='black',label='Volterra-GP - Mean')
    # ax2.fill_between(
    #     ti[::8],
    #         cl_pred_GP_lower[::8],
    #         cl_pred_GP_upper[::8],
    #         alpha=0.5,
    #         # label=r"60% confidence interval GP",
    #         label=r"GP - $\pm$ 1$\sigma$",
    #     )
    # ax2.set_xlabel(r'$\tau$',fontsize=16)
    # ax2.set_ylabel(r'$C_L - C_{L0}$',fontsize=16)
    # ax2.legend(loc='lower right',fontsize=16)
    # ax2.grid()

    # ax3.plot(ti,CM_CFD,'k',linewidth=2,label='CFD')
    # # ax3.plot(tau_train,cmtest_check,'r',linewidth=2,label='Volterra')
    # # ax3.plot(ti,cm_pred,'g',linewidth=2,label='FCNN')
    # ax3.scatter(ti[::8],cm_pred_interp[::8], marker='o', facecolors='black', edgecolors='black',label='Volterra-FCNN')
    # ax3.scatter(ti[::8],cm_pred_interp_GP[::8], marker='o', facecolors='red', edgecolors='black',label='Volterra-GP - Mean')
    # ax3.fill_between(
    #     ti[::8],
    #         cm_pred_GP_lower[::8],
    #         cm_pred_GP_upper[::8],
    #         alpha=0.5,
    #         # label=r"60% confidence interval GP",
    #         label=r"GP - $\pm$ 1$\sigma$",
    #         )  
    # ax3.set_xlabel(r'$\tau$',fontsize=16)
    # ax3.set_ylabel(r'$C_M - C_{M0}$',fontsize=16)
    # ax3.legend(loc='lower right',fontsize=16)
    # ax3.grid()
    # # plt.savefig('Photo\\'+ 'harmonic_signal_20hz.png', bbox_inches='tight')

    # plt.show()
    # plt.close()


    # Prediction
    fig, ((ax2, ax3)) = plt.subplots(2, 1)
    fig.set_size_inches(9.25, 10.5)

    fig.suptitle(' Prediction - M = 0.70 - AoA_0 = 5 - Pitch_ampl = 0.50 - f = 20 Hz ')

    ax2.plot(aoa_input_cfd[351:450:2]*180/np.pi,CL_CFD[351:450:2],linestyle = 'None',marker='s', markersize=8, markerfacecolor='gray', markeredgecolor='black',label='CFD')
    # ax2.plot(tau_train,cltest_check,'r',linewidth=2,label='Volterra')
    # ax2.plot(ti,cl_pred,'g',linewidth=2,label='FCNN')
    ax2.plot(aoa_input[251:]*180/np.pi,cl_pred[251:],'k',linewidth=2,label='Volterra - FCNN')
    ax2.set_xlabel(r'$\alpha [deg]$')
    ax2.set_ylabel(r'$C_L - C_{L0}$')
    ax2.legend(loc='lower right')
    ax2.grid()

    ax3.plot(aoa_input_cfd[351:450:2]*180/np.pi,CM_CFD[351:450:2],linestyle = 'None',marker='s', markersize=8, markerfacecolor='gray', markeredgecolor='black',label='CFD')
    # ax3.plot(tau_train,cmtest_check,'r',linewidth=2,label='Volterra')
    # ax3.plot(ti,cm_pred,'g',linewidth=2,label='FCNN')
    ax3.plot(aoa_input[251:]*180/np.pi,cm_pred[251:],'k',linewidth=2,label='Volterra - FCNN')
    ax3.set_xlabel(r'$\alpha [deg]$')
    ax3.set_ylabel(r'$C_M - C_{M0}$')
    ax3.legend(loc='lower right')
    ax3.grid()
    ax3.set_ylim(-0.023,0.023)
    plt.savefig('Photo\\'+ 'harmonic_signal_hysteresis_20hz.png', bbox_inches='tight')

    plt.show()
    plt.close()

#%% Prediction prediction step_responses constant Mach 

if prediction_step_responses_constant_Mach:

    def extract_elements(original_array):
        # Initialize an empty list to store the extracted elements
        new_vector = []

        # Iterate over the curves in steps of 60
        for i in range(0, 360, 60):
            # Extract the 6 elements from each curve and append to the new vector
            new_vector.extend(original_array[i:i+6, :])

        # Convert the new vector to a NumPy array
        new_vector = np.array(new_vector)

        return new_vector

 # Import linear and NL kernels 
    kernels_CL_linear_pred_074M084_0AoA5 = np.load('Dataset_generator\\kernels_CL_linear_pred_074M084_0AoA5.npy') # samples,timesteps,variables
    kernels_CM_linear_pred_074M084_0AoA5 = np.load('Dataset_generator\\kernels_CM_linear_pred_074M084_0AoA5.npy') # samples,timesteps,variables
    kernels_CL_NL_pred_074M084_0AoA5 = np.load('Dataset_generator\\kernels_CL_NL_pred_074M084_0AoA5.npy') # samples,timesteps,variables
    kernels_CM_NL_pred_074M084_0AoA5 = np.load('Dataset_generator\\kernels_CM_NL_pred_074M084_0AoA5.npy') # samples,timesteps,variables

    kernels_CL_linear_pred_074M084_0AoA5 = kernels_CL_linear_pred_074M084_0AoA5[::10,:]
    kernels_CM_linear_pred_074M084_0AoA5 = kernels_CM_linear_pred_074M084_0AoA5[::10,:]
    kernels_CL_NL_pred_074M084_0AoA5 = kernels_CL_NL_pred_074M084_0AoA5[::10,:]
    kernels_CM_NL_pred_074M084_0AoA5 = kernels_CM_NL_pred_074M084_0AoA5[::10,:]

    kernels_CL_linear_pred_074M084_0AoA5 = extract_elements(kernels_CL_linear_pred_074M084_0AoA5)
    kernels_CM_linear_pred_074M084_0AoA5 = extract_elements(kernels_CM_linear_pred_074M084_0AoA5)
    kernels_CL_NL_pred_074M084_0AoA5 = extract_elements(kernels_CL_NL_pred_074M084_0AoA5)
    kernels_CM_NL_pred_074M084_0AoA5 = extract_elements(kernels_CM_NL_pred_074M084_0AoA5)


    Mach = np.round(np.arange(0.74,0.86,0.002),decimals=2)
    AoA = np.round(np.arange(0,6,0.1),decimals=2)

    # create a 2D grid of Mach and AoA values
    Mach_grid, AoA_grid = np.meshgrid(Mach, AoA)

    # stack the Mach and AoA grids horizontally
    Mach_AoA_array = np.hstack((Mach_grid.reshape(-1, 1), AoA_grid.reshape(-1, 1)))
    Mach_AoA_array = Mach_AoA_array[::10,:]

    Mach_AoA_array = extract_elements(Mach_AoA_array)


    Mach = Mach_AoA_array[:,0]
    AoA = Mach_AoA_array[:,1]
   
   
    cl_pred__074M084_0AoA5 = []
    cm_pred__074M084_0AoA5 = []

    for i in range(kernels_CL_linear_pred_074M084_0AoA5.shape[0]):

        # tau: tempo ridotto
        # t: tempo fisico
        V = Mach[i]*np.sqrt(1.116*81.49*304.2128)

        ### Linear kernels
        tau_sampling = 0.6 #0.125#0.5
        ntsteps_sampling = 18 # int(tau_sampling/(dt*2*V/chord))
        dtau = ntsteps_sampling*dt*2*V/chord
        
        it_pitch_1_deg = np.arange(0,CL_1_deg.shape[1],1)

        tau = it_pitch_1_deg*dt*2*V/chord
        time_1_deg = it_pitch_1_deg*dt

        it_train_1_deg = it_pitch_1_deg[0::ntsteps_sampling] # it: numero di iterazioni

        ntsteps = it_train_1_deg.size
        
        tau_train = it_train_1_deg*dt*2*V/chord
        time_train = it_train_1_deg*dt

        aoa = ampli_train_pitch*(1. - np.exp(-time_1_deg*otref))
        aoa_train = ampli_train_pitch*(1. - np.exp(-(time_train+2*dt*ntsteps_sampling)*otref))


        U = np.zeros((ntsteps,ntsteps))

        for k in range(ntsteps):
            U += aoa_train[k]*np.diag(np.ones(ntsteps - k),-k)
        

        cl_pred = U@kernels_CL_linear_pred_074M084_0AoA5[i,:]
        cm_pred = U@kernels_CM_linear_pred_074M084_0AoA5[i,:]


        ncoeff = np.arange(kernels_CL_linear_pred_074M084_0AoA5.shape[1])

        ### NL kernels
        ampli_train_pitch_2_deg = 2.0
        it_pitch_2_deg = np.arange(0,CL_2_deg.shape[1],1)

        # tau = it_pitch_2_deg*dt*2*V/chord
        time_2_deg = it_pitch_2_deg*dt

        it_train_2_deg = it_pitch_2_deg[0::ntsteps_sampling] # it: numero di iterazioni


        # min length
        n_train_delta = np.minimum(cl_train_1_deg.size,cl_train_2_deg.size)

        tau_train_delta = np.arange(n_train_delta)*dtau
        time_train_delta = np.arange(n_train_delta)*dt


        aoa_delta = ampli_train_pitch_2_deg*(1. - np.exp(-time_2_deg*otref))
        aoa_train_delta = ampli_train_pitch_2_deg*(1. - np.exp(-time_train_delta*otref))


        U2 = np.zeros((n_train_delta,n_train_delta))

        for k in range(n_train_delta):
            U2 += aoa_train_delta[k]*np.diag(np.ones(n_train_delta - k),-k)
        

        cl_NL_pred_074M084_0AoA5 = U2@kernels_CL_NL_pred_074M084_0AoA5[i,:]
        cm_NL_pred_074M084_0AoA5 = U2@kernels_CM_NL_pred_074M084_0AoA5[i,:]

        cl_pred__074M084_0AoA5_temp = cl_pred*factor + cl_NL_pred_074M084_0AoA5
        cl_pred__074M084_0AoA5.append(cl_pred__074M084_0AoA5_temp)

        cm_pred__074M084_0AoA5_temp = cm_pred*factor + cm_NL_pred_074M084_0AoA5
        cm_pred__074M084_0AoA5.append(cm_pred__074M084_0AoA5_temp)

    num_curves = np.array(cl_pred__074M084_0AoA5).shape[0]  # Total number of curves
    num_plots = num_curves // 6 # Number of plots with 6 curves each

    Mach = np.round(np.arange(0.74,0.86,0.02),decimals=2)
    AoA = np.round(np.arange(0,6,1.0),decimals=2)

    ##### CONSTANT AOA ##########

    # plt.rcParams.update({'font.size': 16})
    # for i in range(num_plots):
    #     start_index = i * 6
    #     end_index = start_index + 6
    #     fig, (ax) = plt.subplots(1,1)
    #     fig.set_size_inches(18.5, 10.5)

    #     k = 0
    #     for j in range(start_index, end_index):
    #         curve = cl_pred__074M084_0AoA5[j]
    #         ax.plot(curve, label=f'M = {Mach[k]}')
    #         k = k+1

    #     ax.set_xlabel(r'$\tau$ []')
    #     ax.set_ylabel(r'$C_L - C_{L0}$')
    #     ax.legend(loc='lower right')
    #     ax.grid()
    #     fig.suptitle(f'AoA = {AoA[i]}')
    #     ax.legend(loc='lower right')
    #     plt.show()

    ##### CONSTANT MACH ##########
    # Reshape the original array to (6, 6, 50) to separate Mach and AOA
    cl_pred__074M084_0AoA5=np.array(cl_pred__074M084_0AoA5)
    reshaped_array = cl_pred__074M084_0AoA5.reshape(6, 6, cl_pred__074M084_0AoA5.shape[1])

    # Transpose the reshaped array to group AOA values for each specific Mach
    rearranged_array = reshaped_array.transpose(1, 0, 2)

    # Reshape the rearranged array back to (36, 50)
    cl_pred__074M084_0AoA5 = rearranged_array.reshape(36, cl_pred__074M084_0AoA5.shape[1])

    colors = ['#1f77b4', '#2ca02c', '#ff7f0e',  '#d62728' , '#8c564b' , '#9467bd']

    # colors = ['blue', 'green', 'orange', 'red', 'brown', 'purple']

    plt.rcParams.update({'font.size': 16})
    for i in range(num_plots):
        start_index = i * 6
        end_index = start_index + 6
        fig, (ax) = plt.subplots(1,1)
        fig.set_size_inches(9.25, 5.25)

        k = 0
        for j in range(start_index, end_index):
            curve = cl_pred__074M084_0AoA5[j]
            ax.plot(curve, linewidth=2,c=colors[k], label=r'$\alpha_0$ = '+f'{AoA[k]}')
            ax.set_ylim(-0.04,0.22)
            k = k+1

        ax.set_xlabel(r'$\tau$ []')
        ax.set_ylabel(r'$C_L - C_{L0}$')
        # ax.legend(loc='lower right',bbox_to_anchor=(0.4, 0))
        ax.grid()
        fig.suptitle(f'$Mach = ${Mach[i]}')
        plt.show()
        plt.close()


# # #  Mach number and AoA used for the identification process of pitch step-responses
# mach_numbers = [0.78, 0.80, 0.82, 0.84]
# aoa_values = [0, 1, 2, 3, 4, 5]

# # Define colors for AoA values
# # colors = ['blue', 'green', 'orange', 'red', 'brown', 'purple']
# colors = ['#1f77b4', '#2ca02c', '#ff7f0e',  '#d62728' , '#8c564b' , '#9467bd']

# plt.rcParams.update({'font.size': 16})
# fig, ax = plt.subplots()
# fig.set_size_inches(9.25, 5.25)
# for i, mach in enumerate(mach_numbers):
#     ax.axvline(mach, color='black', linestyle='--', zorder=1)

# for aoa in aoa_values:
#     aoa_color = colors[aoa % len(colors)]
#     ax.scatter(mach_numbers, [aoa] * len(mach_numbers), s=90, marker='o', color=aoa_color, alpha=1.0, zorder=2)

# # Set axis labels and title
# ax.set_xlabel(r'$Mach$')
# # ax.set_ylabel(r'$\alpha_0 \, \, [deg]$')
# ax.set_ylabel(r'$\alpha_0$[deg]')
# # ax.set_title(r'Mach Number vs $\alpha_0$')

# # Set x-axis and y-axis limits
# ax.set_xlim(0.765, 0.855)
# ax.set_ylim(-0.5, 5.5)
# ax.set_xticks(mach_numbers)


# # # Add a legend for the vertical lines
# # legend_labels = [f'Mach {mach}' for mach in mach_numbers]
# # ax.legend(legend_labels, loc='upper left')

# plt.show()


if prediction_harmonic_responses_constant_Mach:

    def extract_elements(original_array):
        # Initialize an empty list to store the extracted elements
        new_vector = []

        # Iterate over the curves in steps of 60
        for i in range(0, 360, 60):
            # Extract the 6 elements from each curve and append to the new vector
            new_vector.extend(original_array[i:i+6, :])

        # Convert the new vector to a NumPy array
        new_vector = np.array(new_vector)

        return new_vector

 # Import linear and NL kernels 
    kernels_CL_linear_pred_074M084_0AoA5 = np.load('Dataset_generator\\kernels_CL_linear_pred_074M084_0AoA5.npy') # samples,timesteps,variables
    kernels_CM_linear_pred_074M084_0AoA5 = np.load('Dataset_generator\\kernels_CM_linear_pred_074M084_0AoA5.npy') # samples,timesteps,variables
    kernels_CL_NL_pred_074M084_0AoA5 = np.load('Dataset_generator\\kernels_CL_NL_pred_074M084_0AoA5.npy') # samples,timesteps,variables
    kernels_CM_NL_pred_074M084_0AoA5 = np.load('Dataset_generator\\kernels_CM_NL_pred_074M084_0AoA5.npy') # samples,timesteps,variables

    kernels_CL_linear_pred_074M084_0AoA5 = kernels_CL_linear_pred_074M084_0AoA5[::10,:]
    kernels_CM_linear_pred_074M084_0AoA5 = kernels_CM_linear_pred_074M084_0AoA5[::10,:]
    kernels_CL_NL_pred_074M084_0AoA5 = kernels_CL_NL_pred_074M084_0AoA5[::10,:]
    kernels_CM_NL_pred_074M084_0AoA5 = kernels_CM_NL_pred_074M084_0AoA5[::10,:]

    kernels_CL_linear_pred_074M084_0AoA5 = extract_elements(kernels_CL_linear_pred_074M084_0AoA5)
    kernels_CM_linear_pred_074M084_0AoA5 = extract_elements(kernels_CM_linear_pred_074M084_0AoA5)
    kernels_CL_NL_pred_074M084_0AoA5 = extract_elements(kernels_CL_NL_pred_074M084_0AoA5)
    kernels_CM_NL_pred_074M084_0AoA5 = extract_elements(kernels_CM_NL_pred_074M084_0AoA5)


    Mach = np.round(np.arange(0.74,0.86,0.002),decimals=2)
    AoA = np.round(np.arange(0,6,0.1),decimals=2)

    # create a 2D grid of Mach and AoA values
    Mach_grid, AoA_grid = np.meshgrid(Mach, AoA)

    # stack the Mach and AoA grids horizontally
    Mach_AoA_array = np.hstack((Mach_grid.reshape(-1, 1), AoA_grid.reshape(-1, 1)))
    Mach_AoA_array = Mach_AoA_array[::10,:]

    Mach_AoA_array = extract_elements(Mach_AoA_array)


    Mach = Mach_AoA_array[:,0]
    AoA = Mach_AoA_array[:,1]
   
   
    cl_pred__074M084_0AoA5 = []
    cm_pred__074M084_0AoA5 = []
    aoa_input_array = []

    ntsteps = 600

    for i in range(kernels_CL_linear_pred_074M084_0AoA5.shape[0]):

        # tau: tempo ridotto
        # t: tempo fisico
        V = Mach[i]*np.sqrt(1.116*81.49*304.2128)


        ### Linear kernels
        tau_sampling = 0.6
        ntsteps_sampling = 18 
        
        ampli_pitch = 0.5*np.pi/180.
        freq = 20.0 # Hz

        # Timestep Volterra
        tau_sampling = 0.6 
        ti = np.linspace(0,tau_sampling*ntsteps,ntsteps)

        k = 2* np.pi * freq * chord / V
        aoa_input = - ampli_pitch*np.sin( k*ti ) 
        aoa_input_array.append(aoa_input)
        number_states = kernels_CL_linear_pred_074M084_0AoA5.shape[1]
        ndof = number_states 
        q = np.zeros((ndof,ntsteps))
        cl_pred =  np.zeros(ntsteps)
        cm_pred =  np.zeros(ntsteps)
        AA = np.diag(np.ones(ndof-1),-1)
        BB = np.zeros((ndof))
        BB[0] = 1.
        CC_CL = kernels_CL_linear_pred_074M084_0AoA5[i,:] + kernels_CL_NL_pred_074M084_0AoA5[i,:]
        CC_CM = kernels_CM_linear_pred_074M084_0AoA5[i,:] + kernels_CM_NL_pred_074M084_0AoA5[i,:]
        for it in range(1,ntsteps):
            q[:,it] = AA@q[:,it-1] + BB*aoa_input[it] 
            cl_pred[it] = CC_CL@q[:,it]
            cm_pred[it] = CC_CM@q[:,it]



        cl_pred__074M084_0AoA5.append(cl_pred)
        cm_pred__074M084_0AoA5.append(cm_pred)


    aoa_input_array = np.array(aoa_input_array)
    reshaped_array = aoa_input_array.reshape(6, 6, aoa_input_array.shape[1])

    # Transpose the reshaped array to group AOA values for each specific Mach
    rearranged_array = reshaped_array.transpose(1, 0, 2)

    # Reshape the rearranged array back to (36, 50)
    aoa_input_array = rearranged_array.reshape(36, aoa_input_array.shape[1])

    num_curves = np.array(cl_pred__074M084_0AoA5).shape[0]  # Total number of curves
    num_plots = num_curves // 6 # Number of plots with 6 curves each

    Mach = np.round(np.arange(0.74,0.86,0.02),decimals=2)
    AoA = np.round(np.arange(0,6,1.0),decimals=2)

    ##### CONSTANT AOA ##########

    # plt.rcParams.update({'font.size': 16})
    # for i in range(num_plots):
    #     start_index = i * 6
    #     end_index = start_index + 6
    #     fig, (ax) = plt.subplots(1,1)
    #     fig.set_size_inches(18.5, 10.5)

    #     k = 0
    #     for j in range(start_index, end_index):
    #         curve = cl_pred__074M084_0AoA5[j]
    #         ax.plot(curve, label=f'M = {Mach[k]}')
    #         k = k+1

    #     ax.set_xlabel(r'$\tau$ []')
    #     ax.set_ylabel(r'$C_L - C_{L0}$')
    #     ax.legend(loc='lower right')
    #     ax.grid()
    #     fig.suptitle(f'AoA = {AoA[i]}')
    #     ax.legend(loc='lower right')
    #     plt.show()

    ##### CONSTANT MACH ##########
    # Reshape the original array to (6, 6, 50) to separate Mach and AOA
    cl_pred__074M084_0AoA5=np.array(cl_pred__074M084_0AoA5)
    reshaped_array = cl_pred__074M084_0AoA5.reshape(6, 6, cl_pred__074M084_0AoA5.shape[1])

    # Transpose the reshaped array to group AOA values for each specific Mach
    rearranged_array = reshaped_array.transpose(1, 0, 2)

    # Reshape the rearranged array back to (36, 50)
    cl_pred__074M084_0AoA5 = rearranged_array.reshape(36, cl_pred__074M084_0AoA5.shape[1])

    plt.rcParams.update({'font.size': 16})
    for i in range(num_plots):
        start_index = i * 6
        end_index = start_index + 6
        fig, (ax) = plt.subplots(1,1)
        fig.set_size_inches(9.25, 6.25)

        k = 0
        # for j in range(start_index, end_index):
        #     curve = cl_pred__074M084_0AoA5[j] 
        #     # Compute the FFT
        #     fft = np.fft.fft(curve)
        #     frequencies = np.fft.fftfreq(len(curve), ti[1] - ti[0])
        #     main_freq = np.abs(frequencies[np.argmax(np.abs(fft))])
        #     aoa_input = - ampli_pitch*np.sin( 2* np.pi * main_freq*ti ) 
        #     ax.plot(aoa_input[150:]*180/np.pi,curve[150:], linewidth=2, label=f'AoA = {AoA[k]}')
        #     # ax.set_ylim(-0.04,0.22)
        #     k = k+1
        colors = ['#1f77b4', '#2ca02c', '#ff7f0e',  '#d62728' , '#8c564b' , '#9467bd']

        for j in range(start_index, end_index):
            curve = cl_pred__074M084_0AoA5[j] 
            ax.plot(aoa_input_array[j,-100:]*180/np.pi,curve[-100:], c=colors[k], linewidth=2, label=r'$\alpha_0$ = '+f'{AoA[k]}')
            ax.set_ylim(-0.045,0.05)
            k = k+1

        ax.set_xlabel(r'$\alpha [deg]$')
        ax.set_ylabel(r'$C_L - C_{L0}$')
        # ax.legend(loc='upper left')
        ax.grid()
        fig.suptitle(f'Mach = {Mach[i]}')
        plt.show()

