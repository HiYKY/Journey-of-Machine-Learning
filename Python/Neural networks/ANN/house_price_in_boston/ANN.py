# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 16:15:15 2020

@author: YuKaiyu
"""

from keras import models
from keras import layers
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils import plot_model
from ann_visualizer.visualize import ann_viz

#======================================== 数据加载 ========================================
'''
本节将要预测 20 世纪 70 年代中期波士顿郊区房屋价格的中位数，已知当时郊区的一些数据点，比如犯罪率、当地房产税率等。

只有 506 个，分为 404 个训练样本和 102 个测试样本。输入数据的每个特征（比如犯罪率）都有不同的取值范围。，有些特性是比例，取值范围为 0~1；有的取值范围为 1~12；还有的取值范围为 0~100
'''

boston = load_boston()

x = boston.data
print(x.shape)
y = boston.target
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

print(x_train.shape)
print(x_test.shape)
#======================================== 数据加载 END ========================================


#======================================== 数据归一化 ========================================
# 神经网络,所以需要数据归一化

train_data = x_train
train_targets = y_train
test_data = x_test
test_targets = y_test

#test_data标准化采用的是train_data的均值和标准差，is this reaonable?
mean = train_data.mean(axis = 0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

# 测试数据的标准化，也只能使用训练数据的mean,std
test_data -= mean
test_data /= std
#======================================== 数据归一化 END ========================================


#======================================== 构建网络 ========================================
'''
一般来说，训练数据越少，过拟合会越严重，而较小的网络可以降低过拟合。 网络的主体： - 两个中间层，每层都有 64 个隐藏单元，使用relu作为激活函数； - 第三层输出一个标量，是一个线性层，不需要激活函数这样可以实现任意值的预测。

注意的点： - loss函数：用的是 mse 损失函数，即均方误差（MSE，mean squared error），预测值与目标值之差的平方。这是回归问题常用的损失函数； - 监控一个新指标：：平均绝对误差（MAE，mean absolute error）。它是预测值与目标值之差的绝对值。
'''

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
#   注意没有激活层，是一个线性层，因为回归的是一个标量
    model.add(layers.Dense(1))

#   mse:均方误差
#   mae:平均绝对误差
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True) 
    ann_viz(model, title="model structure")
    return model  

#======================================== 构建网络 END ========================================


#======================================== 利用K折验证法 ========================================
'''
在coding使用一个numpy中的函数，也就是数据堆叠。 concatenate的使用

每次运行模型得到的验证分数有很大差异，从 2.6 到 3.2 不等。平均分数（3.0）是比单一分数更可靠的指标——这就是 K 折交叉验证的关键。
'''

# 设定K为4，数据折成4段，也需要循环4次，
k_flod = 4

num_val_samples = len(train_data) // k_flod
num_epochs = 100
all_scores = []

for i in range(k_flod):
    print('Processing fold #', i)
    val_data = train_data[i*num_val_samples : (i+1)*num_val_samples]
    val_targets = train_targets[i*num_val_samples : (i+1)*num_val_samples]
    
    # 数据合成
    partial_train_data = np.concatenate([train_data[:i*num_val_samples], train_data[:(i+1)*num_val_samples]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i*num_val_samples], train_targets[:(i+1)*num_val_samples]], axis=0)
    
    # 创建模型
    model = build_model()
    
    # 开始训练
    model.fit(partial_train_data,
              partial_train_targets,
              epochs = num_epochs,
              batch_size = 16,
              verbose = 0
            )
    
    # 进行验证
    val_mse,val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
    
# 训练最终模型
model = build_model()
model.fit(train_data,
          train_targets,
          epochs = 80,
          batch_size = 16,
          verbose = 0
          )

test_mes_score,test_mae_score = model.evaluate(test_data, test_targets)
print(test_mae_score)
#======================================== 利用K折验证法 END ========================================


'''
小结
回归问题使用的损失函数与分类问题不同。回归常用的损失函数是均方误差（MSE）。

回归问题使用的评估指标也与分类问题不同。显而易见，精度的概念不适用于回归问题。常见的回归指标是平均绝对误差（MAE）。

如果输入数据的特征具有不同的取值范围，应该先进行预处理，对每个特征单独进行缩放。

如果可用的数据很少，使用 K 折验证可以可靠地评估模型。

如果可用的训练数据很少，最好使用隐藏层较少（通常只有一到两个）的小型网络，以避免严重的过拟合。
'''