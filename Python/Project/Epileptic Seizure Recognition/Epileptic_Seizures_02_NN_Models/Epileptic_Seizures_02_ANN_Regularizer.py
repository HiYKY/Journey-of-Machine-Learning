# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 21:21:49 2020

@author: YuKaiyu
"""
# Refered from: https://blog.csdn.net/YouZYC/article/details/97387443?depth_1-

# ============================== Load the Dataset and Data Pre-processing ===================================================
#加载机器学习数据集
data_address = 'D:\\Documents\\StoreInDLUTCloud\\MachineLearning\\Program\\EEG_Classification\\PythonScripts\\EEG_Classification_Deeplearning\\Epilepesy_Sezuire_dtection\\Data\\Epileptic Seizure Recognition.csv'
import pandas as pd 
df = pd.read_csv(data_address, header=0, index_col=0)

#探索机器学习数据集
df.head()
df.info()

#将目标变量转换为癫痫(y列编码为1)与非癫痫(2-5)
df["seizure"] = 0 
for i in range(11500): 
    if df["y"][i] == 1: 
        df["seizure"][i] = 1 
    else:
        df["seizure"][i] = 0
        
#绘制并观察一些脑电波
import matplotlib.pyplot as plt 
# plotting an epileptic wave form 
plt.plot(range(178), df.iloc[11496,0:178]) 
plt.show()

#将把数据准备成神经网络可以接受的形式。首先解析数据，然后标准化值，最后创建目标数组
# create df1 which only contains the waveform data points 
df1 = df.drop(["seizure", "y"], axis=1) #删除两列
# 1. parse the data 
import numpy as np 
wave = np.zeros((11500, 178)) 

z=0
for index, row in df1.iterrows():
    #row = pd.DataFrame(row,dtype=np.float64)
    wave[z,:] = row
    z +=1

# print the wave.shape to make sure we parsed the data correctly 
print(wave.shape) 
# 2. standardize the data such that it has mean of 0 and standard deviation of 1 
mean = wave.mean(axis=0) 
wave -= mean 
std = wave.std(axis=0) 
wave /= std 
# 3. create the target numpy array 
target = df["seizure"].values
# ============================== Load the Dataset and Data Pre-processing END ===================================================


# ============================== Develop NN Models ===================================================
#使用Keras构建了一个具有正则化和dropout的dense 网络，以减少过度拟合
from keras.models import Sequential 
from keras import layers 
from keras import regularizers 
from keras.utils import plot_model

model = Sequential() 
model.add(layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l1(0.001), input_shape = (178,))) 
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l1(0.001))) 
model.add(layers.Dropout(0.5)) 
model.add(layers.Dense(1, activation="sigmoid")) 
model.summary()
plot_model(model, to_file='ModelStructure_ANN_Regularizer.png', show_shapes=True)

#sklearn的train_test_split函数帮助我们创建训练和测试集。
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(wave, target, test_size=0.2, random_state=42)

#编译机器学习模型并训练它100个epochs。
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"]) 
history = model.fit(x_train, y_train, epochs=100, batch_size=128, validation_split=0.2, verbose=2)
# ============================== Develop NN Models END ===================================================


# ============================== show the model performance ===================================================
#将模型释放到测试集中，绘制ROC曲线，计算AUC。
from sklearn.metrics import roc_curve, auc 
y_pred = model.predict(x_test).ravel() 
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred) 
AUC = auc(fpr_keras, tpr_keras) 
plt.plot(fpr_keras, tpr_keras, label='Keras Model(area = {:.3f})'.format(AUC)) 
plt.xlabel('False positive Rate') 
plt.ylabel('True positive Rate') 
plt.title('ROC curve') 
plt.legend(loc='best') 
plt.show()
# ============================== show the model performance END ===================================================
