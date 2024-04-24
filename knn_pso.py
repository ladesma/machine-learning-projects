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
pca = PCA(n_components=93)# adjust yourself
pca.fit(X_train_std)
X_t_train_std = pca.transform(X_train_std)
X_t_test_std = pca.transform(X_test_std)
def pso(n_particles, iterations, dimensions, inertia):

    # Range of SVR's hyperparameters (Particles' search space)
    # C, Epsilon and Gamma
    max_c = 10
    min_c = 1
    max_e = 2
    min_e = 1
    max_g = 2
    min_g = 1
    
    # Initializing particles' positions randomly, inside
    # the search space
    x = np.random.rand(n_particles, 1)*(max_c - min_c) + min_c
    y = np.random.rand(n_particles, 1)*(max_e - min_e) + min_e
    z = np.random.rand(n_particles, 1)*(max_g - min_g) + min_g
    #print(x)
    x = np.round(x)
    x = x.astype(int)
    y = np.round(y)
    y = y.astype(int)
    z = np.round(z)
    z = z.astype(int)
    
    
    #c = np.concatenate((x,y,z), axis=1)
    c = np.concatenate((x,y,z), axis=1)
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

          pesos = "uniform"
          if c[j][1] == 2:
              pesos = "distance"
          print(type(c[j][0]))
          rgs = KNeighborsClassifier(n_neighbors = c[j][0], p=c[j][2], weights=pesos)

          # Fitting the curve
          rgs.fit(X_t_train_std, y_train)
          y_predict = rgs.predict(X_t_test_std)

          # Using Mean Squared Error to verify prediction accuracy
          accuracy = accuracy_score(y_test, y_predict) 
          print(accuracy)
          # If mse value for that search point, for that particle,
          # is less than its personal best point,
          # replace personal best
          if(accuracy > p_best_val[j]):   # mse < p_best_val[j]
              # The value below represents the current least Mean Squared Error
              p_best_val[j] = accuracy
              
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
          if(c[j][2] < min_g):
            c[j][2] = min_g
          if(c[j][2] > max_g):
            c[j][2] = max_g
          if(c[j][1] < min_e):
            c[j][1] = min_e
          if(c[j][1] > max_e):
            c[j][1] = max_e
          if(c[j][0] < min_c):
            c[j][0] = min_c
          if(c[j][0] > max_c):
            c[j][0] = max_c
            
        
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
    evaluate(predict_test)  
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
pso(200, 100, 3, 1)