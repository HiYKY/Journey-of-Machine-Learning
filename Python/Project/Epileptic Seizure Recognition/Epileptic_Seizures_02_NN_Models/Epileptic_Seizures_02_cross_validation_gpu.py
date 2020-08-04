# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 11:28:27 2020

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
from timeit import default_timer as timer

# -------------------------------------------------------------------------------------------------------------------
# ------------------------ (1)ANN ------------------------------

# ................ function to creat and compile model ................ 
def build_model():
    # Create model
    model = Sequential()
    model.add( Dense(output_dim = 80, init = 'uniform', activation = 'relu', input_dim = 178) )
    model.add( Dense(output_dim = 80, init = 'uniform', activation = 'relu') )
    model.add( Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid') )
#    model.summary()
#    plot_model( model, to_file='model.png', show_shapes=True )
    
    # Compile model
    model.compile( optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )
    return model
# ................ function to creat and compile model END ................ 
 
# ................ K Fold Validation  ................ 
from sklearn.model_selection import StratifiedKFold
seed = 7
np.random.seed(seed)
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
cvscores = []

for train, test in kfold.split(x, y):   
    model = build_model()    
	# Fit the model	
    history = model.fit(x[train], 
                        y[train], 
                        validation_split=0.33,
                        epochs=15, 
                        batch_size=10, 
                        verbose=0)
	# Evaluate the model	
    scores = model.evaluate(x[test], y[test], verbose=0)	
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))	
    cvscores.append(scores[1] * 100)
    
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
print("The cvscores are:",cvscores)
print("The history is: ")
print(history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# ................ K Fold Validation  END ................ 

# ------------------------ (1)ANN END ------------------------------
# -------------------------------------------------------------------------------------------------------------------

# ============================== Building NN Models END ===================================================





