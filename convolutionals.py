# -*- coding: utf-8 -*-
"""
Created on Sun May 16 00:53:29 2021

@author: danrm
"""

from sklearn.preprocessing import StandardScaler, MinMaxScaler  
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping

# TensorFlow y tf.keras
# Librerias de ayuda
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) 
data = pd.read_csv('taiwan.csv.csv')
corr_data = data.corr()
high_corr = ~(corr_data.mask(np.eye(len(corr_data ), dtype=bool)).abs() > 0.5).any()
high_corr

corr_data = corr_data.loc[high_corr,high_corr]
print(corr_data.columns)
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
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from sklearn.model_selection import train_test_split
model = Sequential()
model.add(Conv1D(64, 2, activation="relu", input_shape=(93,1)))
#model.add(Dense(16, activation="relu"))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(2, activation = 'softmax'))
model.compile(loss = 'binary_crossentropy', 
     optimizer = "adam",               
              metrics = ['accuracy'])
model.summary()
X_t_train_std = X_t_train_std.reshape(10558, 93, 1)
X_t_test_std = X_t_test_std.reshape(2640, 93, 1)

model.fit(X_t_train_std , y_train, batch_size=20,epochs=10, verbose=1)
acc = model.evaluate(X_t_train_std, y_train)
print("Loss:", acc[0], " Accuracy:", acc[1])
#print(X_t_test_std.shape)


pred = model.predict(X_t_test_std)
pred_y = pred.argmax(axis=-1)
print(pred_y)
cm = confusion_matrix(y_test, pred_y)
print(cm)