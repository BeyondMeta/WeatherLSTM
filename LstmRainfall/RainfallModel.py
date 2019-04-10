# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 20:24:48 2019

@author: Sophie
"""

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


#We are fixing the random seed so we can reproduce our results
np.random.seed(27)

#Loading the dataset
df = pd.read_csv("C:/Users/Sophie/Documents/PythonScripts/LstmRainfall/data.csv")
#Looking briefly at the data we notice that 
#the data is split between 28 different locations
#While we have over 2000 observations.
#Most of the different locations have about 70 observations
#Splitting the dataset based on location.
df.sort_values(['venue', 'date'], inplace=True)
Venues = {}
for item in df.venue.unique():
    Venues[item] = df[df['venue'] == item] 
dfCentral = df[df['venue'] == 'Capital Center']
# =============================================================================
# for key in Venues:
#     print(key)
#     plt.plot(Venues[key].date, Venues[key].precipitation )
# =============================================================================
    
plt.plot(Venues['Capital Center'].date, Venues['Capital Center'].temperature )    
#plt.plot(Venues['Td Garden'].date, Venues['Td Garden'].temperature )

#define model
n_features=1
n_steps= 4

model = Sequential()
model.add(LSTM(50, activation='relu', return_sequences= True, input_shape =(n_steps, n_features)))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss='mse')
#fit model
X_train = np.array([Venues['Capital Center'].date, Venues['Capital Center'].precipitation])
X_train.shape
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
model.fit(, Venues['Capital Center'].temperature, epochs= 50, verbose =0)

