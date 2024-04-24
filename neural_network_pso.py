# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 00:16:03 2021

@author: Piston
"""

import random
import numpy
import matplotlib.pyplot as plt
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
from sklearn.preprocessing import StandardScaler, MinMaxScaler  
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
import sys
from sklearn.neighbors import KNeighborsClassifier

# TensorFlow y tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
# Librerias de ayuda
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import warnings
with tf.device('/cpu:0'):
    data = pd.read_csv('taiwan.csv.csv')
    target = data['Bankrupt?']
    del data['Bankrupt?']
    dato = data[:1]
    SMOTE_oversample = SMOTE()
    data,target = SMOTE_oversample.fit_resample(data,target)
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.20, random_state=42)
    std_slc =  StandardScaler()
    std_slc.fit(X_train)
    X_train_std = std_slc.transform(X_train)
    X_test_std = std_slc.transform(X_test)
    pca = PCA(n_components=30)# adjust yourself
    pca.fit(X_train_std)
    X_t_train_std = pca.transform(X_train_std)
    X_t_test_std = pca.transform(X_test_std)
    def funcion_activacion(funcion):
        if funcion ==0:
            return "linear"
        elif funcion == 1:
            return "sigmoid"
        elif funcion == 2:
            return "relu"
        elif funcion == 3:
            return "tanh"
        elif funcion == 4:
            return "PReLU"
        elif funcion == 5:
            return "selu"
        else:
            return "elu" 
    
            
    def funcion_optimizador(optimizador,learning):
        if learning == 0:
            learning = 0.01
        elif learning == 1:
            learning = 0.001
        elif learning == 2:
            learning = 0.0001
        else:
            learning = 0.00001
        
        if optimizador == 0:
            return keras.optimizers.Adam(learning_rate=learning)
        elif optimizador == 1:
            return keras.optimizers.RMSprop(learning_rate=learning)
        elif optimizador == 2:
            return keras.optimizers.SGD(learning_rate=learning)
        elif optimizador == 3:
            return keras.optimizers.Adadelta(learning_rate=learning)
        elif optimizador == 4:
            return keras.optimizers.Adagrad(learning_rate=learning)
        elif optimizador == 5:
            return keras.optimizers.Adamax(learning_rate=learning)
        elif optimizador == 6:
            return keras.optimizers.Nadam(learning_rate=learning)
        else:
            return keras.optimizers.Ftrl(learning_rate=learning)
    def pso(n_particles, iterations, dimensions, inertia):
    
        # Range of SVR's hyperparameters (Particles' search space)
        # C, Epsilon and Gamma
        max_cap = 3
        min_cap = 1
        max_neu_1 = 100
        min_neu_1 = 1
        max_neu_2 = 100
        min_neu_2 = 1
        max_neu_3 = 100
        min_neu_3 = 1
        max_act_1 = 6
        min_act_1 = 0
        max_act_2 = 6
        min_act_2 = 0
        max_act_3 = 6
        min_act_3 = 0
        max_opt= 7
        min_opt= 0
        max_rate= 3
        min_rate = 0
        max_salida = 1
        min_salida = 0
        
        # Initializing particles' positions randomly, inside
        # the search space
        x = np.random.rand(n_particles, 1)*(max_cap - min_cap) + min_cap
        y = np.random.rand(n_particles, 1)*(max_neu_1 - min_neu_1) + min_neu_1
        z = np.random.rand(n_particles, 1)*(max_neu_2 - min_neu_2) + min_neu_2
        h = np.random.rand(n_particles, 1)*(max_neu_3 - min_neu_3) + min_neu_3
        i = np.random.rand(n_particles, 1)*(max_act_1 - min_act_1) + min_act_1
        j = np.random.rand(n_particles, 1)*(max_act_2 - min_act_2) + min_act_2
        k = np.random.rand(n_particles, 1)*(max_act_3 - min_act_3) + min_act_3
        l = np.random.rand(n_particles, 1)*(max_opt - min_opt) + min_opt
        m = np.random.rand(n_particles, 1)*(max_rate - min_rate) + min_rate
        n = np.random.rand(n_particles, 1)*(max_salida - min_salida) + min_salida     
        #print(x)
        x = np.round(x)
        x = x.astype(int)
        y = np.round(y)
        y = y.astype(int)
        z = np.round(z)
        z = z.astype(int)
        h = np.round(h)
        h = h.astype(int)
        i = np.round(i)
        i = i.astype(int)
        j = np.round(j)
        j = j.astype(int)
        k = np.round(k)
        k = k.astype(int)
        l = np.round(l)
        l = l.astype(int)
        m = np.round(m)
        m = m.astype(int)
        n = np.round(n)
        n = n.astype(int)
        
        
        #c = np.concatenate((x,y,z), axis=1)
        c = np.concatenate((x,y,z,h,i,j,k,l,m,n), axis=1)
        # Initializing particles' parameters
        v = np.zeros((n_particles, dimensions))
        c1 = 2
        c2 = 2
        p_best = np.zeros((n_particles, dimensions))
        p_best_val = np.zeros(n_particles)  
        g_best = np.zeros(dimensions)
        g_best_val = 0
    
        best_iter = np.zeros(iterations)
    
        # Initializing regression variables
        p_best_RGS = np.empty((n_particles), dtype = object);
        g_best_RGS = 0
    
        
    
        # Displaying tridimensional search space
        #plot(c)
    
        from sklearn.metrics import mean_squared_error, accuracy_score
        
        for i in range(iterations):
    
            for j in range(n_particles):
              # Starting Regression
              #rgs = svm.SVC(C = c[j][0], epsilon = c[j][1], gamma = c[j][2])
              if c[j][0]==2:
                  model = keras.Sequential([
                  layers.Dense(c[j][1] , activation=funcion_activacion(c[j][4]), input_shape=[30]),
                  layers.Dense(c[j][2], activation=funcion_activacion(c[j][5])),
                  layers.Dense(1, activation=funcion_activacion(c[j][9])),
                  ])
              elif c[j][0]==3:
                  model = keras.Sequential([
                  layers.Dense(c[j][1] , activation=funcion_activacion(c[j][4]), input_shape=[30]),
                  layers.Dense(c[j][2], activation=funcion_activacion(c[j][5])),
                  layers.Dense(c[j][3], activation=funcion_activacion(c[j][6])),
                  layers.Dense(1, activation=funcion_activacion(c[j][9])),
                  ])
              else:
                  model = keras.Sequential([
                  layers.Dense(c[j][1] , activation=funcion_activacion(c[j][4]), input_shape=[30]),
                  layers.Dense(1, activation=funcion_activacion(c[j][9])),
                  ])
            
              model.compile(loss='binary_crossentropy', optimizer=funcion_optimizador(c[j][7],c[j][8]), metrics=['accuracy'])
              #print(model.summary())
              rgs = model
              rgs.fit(
              X_t_train_std, 
              y_train, 
              epochs=100,
              verbose=0
              #batch_size=10,  
              )
              y_predict = rgs.predict(X_t_test_std)
              accuracy = model.evaluate(X_t_test_std, y_test)
    
              #accuracy = rgs.evaluate(X_t_test_std, y_test)
              
              
    
              # If mse value for that search point, for that particle,
              # is less than its personal best point,
              # replace personal best
              if(accuracy[1] > p_best_val[j]):   # mse < p_best_val[j]
                  # The value below represents the current least Mean Squared Error
                  p_best_val[j] = accuracy[1]
                  
                  p_best_RGS[j] = rgs
                               
    
                  # The value below represents the current search coordinates for
                  # the particle's current least Mean Squared Error found
                  p_best[j] = c[j].copy()
                  
              # Using auxiliar variable to get the index of the
              # particle that found the configuration with the 
              # minimum MSE value
              aux = np.argmax(p_best_val)        
            
              if(p_best_val[aux] > g_best_val):
                  # Assigning Particle's current best MSE to the Group's best    
                  g_best_val = p_best_val[aux]
    
                  # Assigning Particle's current best configuration to the Group's best
                  g_best = p_best[aux].copy()
    
                  # Group best regressor:
                  # the combination of C, Epsilon and Gamma
                  # that computes the best fitting curve
                  g_best_RGS = p_best_RGS[aux]
    
            
              rand1 = np.random.random()
              rand2 = np.random.random()
    
              # The variable below influences directly the particle's velocity.
              # It can either make it smaller or bigger. 
              w = inertia
    
              # The equation below represents Particle's velocity, which is
              # the rate of change in its position
              v[j] = w*v[j] + c1*(p_best[j] - c[j])*rand1 + c2*(g_best - c[j])*rand2
    
              # Change in the Particle's position 
              c[j] = c[j] + v[j]
    
              # Below is a series of conditions that stop the particles from
              # leaving the search space
             
              
    
              if(c[j][0] < min_cap):
                  c[j][0] = min_cap
              if(c[j][0] > max_cap):
                  c[j][0] = max_cap
              if(c[j][1] < min_neu_1):
                  c[j][1] = min_neu_1
              if(c[j][1] > max_neu_1):
                  c[j][1] = max_neu_1
              if(c[j][2] < min_neu_2):
                  c[j][2] = min_neu_2
              if(c[j][2] > max_neu_2):
                  c[j][2] = max_neu_2
              if(c[j][3] < min_neu_3):
                  c[j][3] = min_neu_3
              if(c[j][3] > max_neu_3):
                  c[j][3] = max_neu_3
              if(c[j][4] < min_act_1):
                  c[j][4] = min_act_1
              if(c[j][4] > max_act_1):
                  c[j][4] = max_act_1
              if(c[j][5] < min_act_2):
                  c[j][5] = min_act_2
              if(c[j][5] > max_act_2):
                  c[j][5] = max_act_2
              if(c[j][6] < min_act_3):
                  c[j][6] = min_act_3
              if(c[j][6] > max_act_3):
                  c[j][6] = max_act_3
              if(c[j][7] < min_opt):
                  c[j][7] = min_opt
              if(c[j][7] > max_opt):
                  c[j][7] = max_opt
              if(c[j][8] < min_rate):
                  c[j][8] = min_rate
              if(c[j][8] > max_rate):
                  c[j][8] = max_rate
              if(c[j][9] < min_salida):
                  c[j][9] = min_salida
              if(c[j][9] > max_salida):
                  c[j][9] = max_salida
          
            
            # The variable below represents the least Mean Squared Error
            # of the current iteration
            best_iter[i] = g_best_val
                    
            print('Best value iteration #',i,' ', g_best_val)
    
        # Coordinates found after all the iterations
        print('Group Best configuration found: ')
        print(g_best)
        print('\n')
        print('Best Regressor:\n')
        print(g_best_RGS)
        print('\n')
        # Displaying the MSE value variation throughout the iterations
        t = range(iterations)
        plt.plot(t, best_iter, label='Fitness Value')
        plt.legend()
        plt.show()
    
        # Displaying Particles' final configuration
        #plot(c)
    
        # Making the prediction with the best configuration of C, Epsilon and
        # Gamma found by the particles
        predict_test = g_best_RGS.predict(X_t_test_std)
    
        # Displaying actual values and predicted values for
        # Group's best configuration found overall
        print(color.BOLD + 'Predictions with the Population Best Value found:\n' + color.END)
        #evaluate(predict_test)  
    class color:
       PURPLE = '\033[95m'
       CYAN = '\033[96m'
       DARKCYAN = '\033[36m'
       BLUE = '\033[94m'
       GREEN = '\033[92m'
       YELLOW = '\033[93m'
       RED = '\033[91m'
       BOLD = '\033[1m'
       UNDERLINE = '\033[4m'
       END = '\033[0m'
    # Function that displays tridimensional plot
    def plot(some_list):
      from mpl_toolkits.mplot3d import axes3d, Axes3D
      ax = Axes3D(plt.figure())
      ax.scatter3D(some_list[:,0], some_list[:,1], some_list[:,2], color = 'r')
      ax.set_xlabel('$C$', fontsize = 20)
      ax.set_ylabel('$\epsilon$', fontsize = 25)
      ax.zaxis.set_rotate_label(False) 
      ax.set_zlabel('$\gamma$', fontsize=30, rotation = 0)
      ax.zaxis._axinfo['label']['space_factor'] = 1.0
      plt.show()
    
      print('\n')
      print('\n')
    def evaluate(predictions):
    
        from sklearn.metrics import mean_squared_error, accuracy_score
        import statistics as st
    
        predict_test = predictions
    
        # To un-normalize the data:
        # Multiply the values by
        # data.to_numpy().max()
    
        plt.plot(range(len(y_test)), y_test, label='Real')
        plt.plot(range(len(predict_test)), predict_test, label='Predicted')
        plt.legend()
        plt.show()
        print("y_test ",y_test, " predict ", predict_test)
        mse = accuracy_score(y_test, predict_test)
        print('\n')
        print('\n')
        print('Mean Squared Error for the Test Set:\t %f' %mse)
        print('\n')
        print('\n')
        print('Predictions Average:\t %f' %((predict_test.sum()/len(predict_test))))
        print('\n')
        print('\n')
        print('Predictions Median:\t %f' %(st.median(predict_test)))
        print('\n')
        print('\n')
    pso(2, 2, 10, 1)

