# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 09:47:48 2020

@author: YuKaiyu
"""

# ============================== Importing the libraries ===================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
# ============================== Importing the libraries END ===================================================


# ============================== Load the Dataset ===================================================
data_address = 'D:\\Documents\\StoreInDLUTCloud\\MachineLearning\\Program\\EEG_Classification\\PythonScripts\\EEG_Classification_Deeplearning\\Epilepesy_Sezuire_dtection\\Data\\Epileptic Seizure Recognition.csv'
ESR = pd.read_csv(data_address)
ESR.head()
# ============================== Load the Dataset END ===================================================


# ============================== Read and Show Dataset ===================================================
'''
The original dataset from the reference consists of 5 different folders, each with 100 files, with each file representing a single subject/person. Each file is a recording of brain activity for 23.6 seconds.

The corresponding time-series is sampled into 4097 data points. Each data point is the value of the EEG recording at a different point in time. So we have total 500 individuals with each has 4097 data points for 23.5 seconds.

We divided and shuffled every 4097 data points into 23 chunks, each chunk contains 178 data points for 1 second, and each data point is the value of the EEG recording at a different point in time.

So now we have 23 x 500 = 11500 pieces of information(row), each information contains 178 data points for 1 second(column), the last column represents the label y {1,2,3,4,5}.

The response variable is y in column 179, the Explanatory variables X1, X2, ..., X178
'''

ESR.head()

cols = ESR.columns
tgt = ESR.y
tgt.unique()
tgt[tgt>1] = 0
ax = sn.countplot(tgt,label="Count")
non_seizure, seizure = tgt.value_counts()
print('The number of trials for the non-seizure class is:', non_seizure)
print('The number of trials for the seizure class is:', seizure)

'''
As we can see, there are 178 EEG features and 5 possible classes. The main goal of the dataset it's to be able to correctly identify epileptic seizures from EEG data, so a binary classification between classes of label 1 and the rest (2,3,4,5). In order to train our model, let's define our independent variables (X) and our dependent variable (y).
'''
# ============================== Read and Show Dataset END ===================================================


# ============================== Data Pre-processing  ===================================================
'''
What is Data Pre-pocessing?
Data preprocessing is a data mining technique that involves transforming raw data into an understandable format. Real-world data is often incomplete, inconsistent, and/or lacking in certain behaviors or trends, and is likely to contain many errors. Data preprocessing is a proven method of resolving such issues. Data preprocessing prepares raw data for further processing.
'''

# -------------------------- 1. Checking Missing Data ----------------------------------------------------
ESR.isnull().sum()

ESR.info()

ESR.describe()

X = ESR.iloc[:,1:179].values
X.shape

plt.subplot(511)
plt.plot(X[1,:])
plt.title('Classes')
plt.ylabel('uV')
plt.subplot(512)
plt.plot(X[7,:])
plt.subplot(513)
plt.plot(X[12,:])
plt.subplot(514)
plt.plot(X[0,:])
plt.subplot(515)
plt.plot(X[2,:])
plt.xlabel('Samples')

y = ESR.iloc[:,179].values
y

y[y>1]=0
y
# -------------------------- 1. Checking Missing Data END ----------------------------------------------------

# -------------------------- 2. Standardize features by removing the mean and scaling to unit variance ----------------------------------------------------
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
x = scaler.transform(X)

x.shape
# -------------------------- 2. Standardize features by removing the mean and scaling to unit variance END ----------------------------------------------------

# ============================== Data Pre-processing END ===================================================


# ============================== Building NN Models ===================================================
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Bidirectional, GRU, Dropout, BatchNormalization, Embedding, ConvLSTM2D
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from timeit import default_timer as timer

# -------------------------------------------------------------------------------------------------------------------
# ------------------------ (1)ANN ------------------------------

# ................ Splitting the Dataset into the Training set and Test set ................ 
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# ................  Splitting the Dataset into the Training set and Test set END ................ 

#................ create model ................
# Initializing the ANN
classifier = Sequential()
# Adding input layer and first hidden layer
classifier.add(Dense(output_dim = 80, init = 'uniform', activation = 'relu', input_dim = 178))
# Adding second hidden layer
classifier.add(Dense(output_dim = 80, init = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier.summary()
plot_model(classifier, to_file='ModelStructure_ANN.png', show_shapes=True)
#................ create model END ................

#................ compile model ................
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#................ compile model END ................

#................ train model ................
start_time = timer()

#Fitting the ANN to the training set
classifier.fit(x_train, y_train, batch_size = 10, epochs = 10)

end_time = timer()
print ("Train Duration in seconds :", end_time-start_time)
#................ train model END ................

#................ evaluate model ................
score, acc = classifier.evaluate(x_test, y_test)
acc
#................ evaluate model END ................
# ------------------------ (1)ANN END ------------------------------
# -------------------------------------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------------------------------
# --------------------------------------------- CNNs and RNNs---------------------------------------------------
# ................ Splitting the Dataset into the Training set and Test set ................ 
y = to_categorical(y)
y.shape

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# ................ Splitting the Dataset into the Training set and Test set END ................ 

# ------------------------ Feature Scaling ------------------------------
x_train = np.reshape(x_train, (x_train.shape[0],1,X.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0],1,X.shape[1]))

x_train.shape
x_test.shape
# ------------------------ Feature Scaling END ------------------------------


# ------------------------ (2)CNN 1d ------------------------------
#........... parameters for model ...........
n_timesteps = x_train.shape[1]
n_features = x_train.shape[2]
n_outputs = y_train.shape[1]

#n_timesteps = 1
#n_features = 178
#n_outputs = 5 # 5 labels

verbose = 2
epochs = 100
batch_size = 32
#........... parameters for model END ...........

#................ create model ................
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=1, activation='relu',input_shape=(n_timesteps, n_features)))
model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=1))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.summary()

plot_model(model, to_file='ModelStructure_conv1d.png', show_shapes=True) 
#................ create model END ................

#................ compile model ................
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5), metrics=['accuracy'])
#................ compile model END ................

#................ train model ................
start_time=timer()

model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, verbose = verbose)

end_time=timer()
print ('Train Duration in seconds :', end_time-start_time)

train_loss, train_accuracy = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0) 
print("train_loss =",train_loss)
print("train_accuracy =",train_accuracy)
#................ train model END ................

#................ evaluate model ................
score, acc = model.evaluate(x_test, y_test)

pred = model.predict(x_test)
predict_classes = np.argmax(pred,axis=1)
expected_classes = np.argmax(y_test,axis=1)

print(expected_classes.shape)
print(predict_classes.shape)

correct = accuracy_score(expected_classes,predict_classes)
print(f"Training Accuracy: {correct}")
#................ evaluate model END ................
# ------------------------ (2)CNN 1d END ------------------------------


# ------------------------ (3)conv1d_lstm, doesn't work ------------------------------
# ........... parameters for model ...........
n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]

verbose, epochs, batch_size = 2, 50, 32

n_steps, n_length = 1, 32 
# ........... parameters for model END ...........

# ........... transform data format ...........
x_train_conv1d_lstm = x_train.reshape((x_train.shape[0], n_steps, n_length, n_features))
x_test_conv1d_lstm = x_test.reshape((x_test.shape[0], n_steps, n_length, n_features))
# ........... transform data format END ...........

#................ create model ................
model = Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(1,1), activation='relu', input_shape=(n_steps, 1, n_length, n_features)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.summary()

plot_model(model, to_file='ModelStructure_convlstm2d.png', show_shapes=True) 
#................ create model END ................
# ------------------------ (3)conv1d_lstm END, doesn't work  ------------------------------


# ------------------------ (4)LSTM ------------------------------
#................ create model ................
model = Sequential()
model.add(LSTM(64, input_shape=(1,178),activation="relu",return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32,activation="sigmoid"))
model.add(Dropout(0.5))

#model.add(LSTM(100,return_sequences=True))
#model.add(Dropout(0.2))
#model.add(LSTM(50))
#model.add(Dropout(0.2))

model.add(Dense(2, activation='sigmoid'))

model.summary()
plot_model(model, to_file='ModelStructure_LSTM.png', show_shapes=True) 
#................ create model END ................

#................ compile model ................
model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics = ['accuracy'])
#................ compile model END ................

#................ train model ................
start_time = timer()

history = model.fit(x_train, y_train, epochs = 50, validation_data= (x_test, y_test))

history=model.fit(x_train, 
                  y_train, 
                  batch_size=16,
                  validation_data=(x_test, y_test),
                  epochs=50)

end_time = timer()
print ("Train Duration in seconds :", end_time-start_time)
#................ train model END ................

#................ evaluate model ................
score, acc = model.evaluate(x_test, y_test)

pred = model.predict(x_test)
predict_classes = np.argmax(pred,axis=1)
expected_classes = np.argmax(y_test,axis=1)

print(expected_classes.shape)
print(predict_classes.shape)

correct = accuracy_score(expected_classes,predict_classes)
print(f"Training Accuracy: {correct}")
#................ evaluate model END ................
# ------------------------ (4)LSTM END ------------------------------

# ------------------------ (5)GRU ------------------------------
#........... parameters for model ...........
n_timesteps = x_train.shape[1]
n_features = x_train.shape[2]
n_outputs = y_train.shape[1]

n_epoch = 50
batch_size = 16
n_hidden = 32*3
#........... parameters for model END ...........

#................ create model ................
model = Sequential()
model.add( GRU(output_dim=n_hidden, activation='tanh', input_shape=(n_timesteps, n_features)) )
model.add(Dense(n_hidden, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.summary()

plot_model(model, to_file='ModelStructure_GRU.png', show_shapes=True) 
#................ create model END ................

#................ compile model ................
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
#................ compile model END ................

#................ train model ................
start_time = timer()

#history = model.fit(x_train, y_train, epochs=n_epoch, validation_data= (x_test, y_test))

history=model.fit(x_train, 
                  y_train, 
                  batch_size=batch_size,
                  validation_data=(x_test, y_test),
                  epochs=n_epoch)

end_time = timer()
print ("Train Duration in seconds :", end_time-start_time)
#................ train model END ................

#................ evaluate model ................
score, acc = model.evaluate(x_test, y_test)

pred = model.predict(x_test)
predict_classes = np.argmax(pred,axis=1)
expected_classes = np.argmax(y_test,axis=1)

print(expected_classes.shape)
print(predict_classes.shape)

correct = accuracy_score(expected_classes,predict_classes)
print(f"Training Accuracy: {correct}")
#................ evaluate model END ................
# ------------------------ (5)GRU END ------------------------------


# ------------------------ (6)BiLSTM ------------------------------
#........... parameters for model ...........
n_timesteps = x_train.shape[1]
n_features = x_train.shape[2]
n_outputs = y_train.shape[1]

n_epoch = 50
batch_size = 16
n_hidden = 32*3
#........... parameters for model END ...........

#................ create model ................
model = Sequential()
model.add( Bidirectional( LSTM(output_dim=n_hidden, activation='tanh', input_shape=(n_timesteps, n_features)) ) )
model.add(Dense(n_hidden, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))

plot_model(model, to_file='ModelStructure_BiLSTM.png', show_shapes=True) 
#................ create model END ................

#................ compile model ................
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
#................ compile model END ................

#................ train model ................
start_time = timer()

model.fit(x_train, 
          y_train, 
          batch_size=batch_size,
          validation_data=(x_test, y_test),
          epochs=n_epoch)
end_time = timer()
print ("Train Duration in seconds :", end_time-start_time)
#................ train model END ................

#................ evaluate model ................
score, acc = model.evaluate(x_test, y_test)

pred = model.predict(x_test)
predict_classes = np.argmax(pred,axis=1)
expected_classes = np.argmax(y_test,axis=1)

print(expected_classes.shape)
print(predict_classes.shape)

correct = accuracy_score(expected_classes,predict_classes)
print(f"Training Accuracy: {correct}")
#................ evaluate model END ................
# ------------------------ (6)BiLSTM END ------------------------------

# --------------------------------------------- CNNs and RNNs ---------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------

# ============================== Building NN Models END ===================================================
