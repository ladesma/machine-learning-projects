# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 21:41:08 2021

@author: Piston
"""
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
#from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
from sklearn.model_selection import StratifiedKFold

# TensorFlow y tf.keras
#import tensorflow as tf
#from tensorflow import keras
from tensorflow.keras import layers, callbacks
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
sns.set_style('white');
sns.set_context(context='notebook',font_scale=1.2)
sns.countplot(x=target);
plt.title('Target variable unbalanced');
SMOTE_oversample = SMOTE()
data,target = SMOTE_oversample.fit_resample(data,target)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.20, random_state=42)
"""
sns.set_style('white');
sns.set_context(context='notebook',font_scale=1.2)
sns.countplot(x=y_train);
plt.title('Target variable balanced');
"""
std_slc =  StandardScaler()
std_slc.fit(X_train)
X_train_std = std_slc.transform(X_train)
X_test_std = std_slc.transform(X_test)
pca = PCA(n_components=50)# adjust yourself
pca.fit(X_train_std)
X_t_train_std = pca.transform(X_train_std)
X_t_test_std = pca.transform(X_test_std)
print(y_test)
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, LSTM, Dropout
from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix
#from numpy import unique

model = Sequential()
model.add(Conv1D(128, 4, activation="tanh", input_shape=(50,1)))
model.add(MaxPooling1D())
#model.add(Conv1D(32, 4, activation="relu"))
#model.add(MaxPooling1D())
#model.add(LSTM(64))
model.add(Flatten())
model.add(Dense(100 , activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.10))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation='relu'))
"""
#model.add(Dense(50, activation="relu"))
model.add(Dense(100, activation="tanh"))
model.add(Dense(4, activation="tanh"))
model.add(Dense(2,activation = 'relu'))
"""
model.compile(loss = 'binary_crossentropy', 
     optimizer = "adam",               
              metrics = ['accuracy'])
model.summary()
y_train = np.asarray(y_train).astype('float32').reshape((-1,1))
y_test = np.asarray(y_test).astype('float32').reshape((-1,1))
X_t_train_std = X_t_train_std.reshape(10558, 50, 1)
early_stop =  EarlyStopping(monitor='loss',mode='max', verbose=1, patience=30,restore_best_weights=True)
"""
model.fit(X_t_train_std , y_train, batch_size=16,epochs=100, verbose=1)#callbacks=[early_stop]
print(X_t_train_std.shape)
acc = model.evaluate(X_t_train_std, y_train)
print("Loss:", acc[0], " Accuracy:", acc[1])
X_t_test_std = X_t_test_std.reshape(2640,50, 1)

pred = model.predict(X_t_test_std)
pred_y = pred.argmax(axis=-1)
print(pred_y)
cm = confusion_matrix(y_test, pred_y)
print(cm)
"""



early_stop =  EarlyStopping(monitor='loss',mode='max', verbose=1, patience=30,restore_best_weights=True)
std_slc =  StandardScaler()
std_slc.fit(data)
data_std = std_slc.transform(data)
pca = PCA(n_components=30)# adjust yourself
pca.fit(data_std)
data_std = pca.transform(data_std)
print(data_std.shape)
data_std = data_std.reshape(13198, 30, 1)


# Wrap Keras model so it can be used by scikit-learn
#neural_network = KerasClassifier(build_fn=create_network, epochs=1000, batch_size=100, verbose=1, callbacks=[early_stop])
# Evaluate neural network using three-fold cross-validation
#cross_val_score(neural_network, data_std, target, cv=10)
seed = 7

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(data_std, target):
  early_stop =  EarlyStopping(monitor='loss',mode='max', verbose=1, patience=30,restore_best_weights=True)
  model = Sequential()
  model.add(Conv1D(16, 8, activation="relu", input_shape=(30,1)))
  model.add(MaxPooling1D())
  #model.add(Conv1D(32, 2, activation="relu"))
  #model.add(MaxPooling1D())
#model.add(LSTM(64))
  model.add(Flatten())
  model.add(Dense(100 , activation='relu'))
  model.add(Dense(100, activation='relu'))
  model.add(Dropout(0.10))
  model.add(Dense(100, activation='relu'))
  model.add(Dense(2, activation='relu'))
  """
  model.add(Dense(100, activation="tanh"))
  model.add(Dense(4, activation="tanh"))
  model.add(Dense(2,activation = 'relu'))
  """
  model.compile(loss = 'binary_crossentropy', 
      optimizer = 'adam',               
                metrics = ['accuracy'])

  model.fit(data_std[train], target[train], epochs=100, batch_size=16, verbose=1)#, callbacks=[early_stop]
	# evaluate the model
  scores = model.evaluate(data_std[test], target[test], verbose=0)
  print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
  cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
