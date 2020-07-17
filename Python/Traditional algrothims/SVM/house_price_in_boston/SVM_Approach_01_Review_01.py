from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd  ##用于做数据分析
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model  ##加载线性模型
from sklearn.model_selection import cross_val_predict
from sklearn import metrics  ###这个模块中含有评分函数，性能度量，距离计算等


# 博文：http://www.cnblogs.com/Lin-Yi/p/8971845.html

#======================================== 1 准备数据 ========================================
# 读取波士顿地区房价信息
boston = load_boston()

# 查看数据描述
print(boston.DESCR)   # 共506条波士顿地区房价信息，每条13项数值特征描述和目标房价

# 查看数据的差异情况
print("最大房价：", np.max(boston.target))   # 50
print("最小房价：",np.min(boston.target))    # 5
print("平均房价：", np.mean(boston.target))   # 22.532806324110677

x = boston.data
print(x.shape)
y = boston.target
print(y.shape)


# ---------------------------------- exploratory data analysis ---------------------------------
# 查看数据组成
print(boston.keys()) # 分别代表，数据，目标，特征名称，描述信息
# dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])

# 将描述信息打印输出
print(boston['DESCR']) # 样本条数， 特征个数 Median Value (attribute 14) is usually the target.目标

# ..........................original data .....................................
# 查看数据集中各个特征的数量级分布情况,使用散点图进行可视化
width = 8
height = 8
fig = plt.figure(figsize=(width,height))

# matplotlib inline
plt.plot(x.min(axis = 0), 'v', label='min')
plt.plot(x.max(axis = 0), '^', label='max')
# 设定纵坐标为对数形式
plt.yscale('log')
# 设置图注位置为最佳
plt.legend(loc='best')
# 设置最标轴标题
plt.xlabel('features')
plt.ylabel('features distribute')
# 显示图像
plt.show()
# ..........................original data END .....................................

# ---------------------------------- exploratory data analysis END ---------------------------------
#======================================== 1 准备数据 END ========================================


#======================================== 2 分割训练数据和测试数据 ========================================
# 随机采样25%作为测试 75%作为训练
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=33)

print(x_train.shape)
print(x_test.shape)
#======================================== 2 分割训练数据和测试数据 END ========================================


#======================================== 3 训练数据和测试数据进行标准化处理 ========================================
ss_x = StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)

ss_y = StandardScaler()
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1, 1))

# 处理之后的数据进行可视化
# Define the figure and setting dimensions width and height
width = 8
height = 8
fig = plt.figure(figsize=(width,height))

plt.plot(x_train.min(axis=0),'v',label = 'train set min')
plt.plot(x_train.max(axis=0),'^',label = 'train set max')
plt.plot(x_test.min(axis=0),'v',label = 'test set min')
plt.plot(x_test.max(axis=0),'^',label = 'test set max')
plt.yscale('log')
plt.legend(loc = 'best') # 设置图标注位置，为最佳

# 设置坐标轴标题
plt.xlabel('scaled features')
plt.ylabel('scaled features distribute')
plt.show()
#======================================== 3 训练数据和测试数据进行标准化处理 END ========================================


#======================================== 4 建模预测 ========================================

#--------------------------------------- 4.1 支持向量机模型进行学习和预测 ---------------------------------------
# 线性核函数配置支持向量机
linear_svr = SVR(kernel="linear")
linear_svr.fit(x_train, y_train) # 训练

# coef_: 每个特征系数（重要性），只有核函数是Linear的时候可用
print(linear_svr.coef_)

linear_svr_y_predict = linear_svr.predict(x_test) # 预测 保存预测结果

# 多项式核函数配置支持向量机
poly_svr = SVR(kernel="poly")
poly_svr.fit(x_train, y_train) # 训练
poly_svr_y_predict = linear_svr.predict(x_test) # 预测 保存预测结果

# rbf核函数配置支持向量机
rbf_svr = SVR(kernel="rbf")
rbf_svr.fit(x_train, y_train) # 训练
rbf_svr_y_predict = rbf_svr.predict(x_test) # 预测 保存预测结果
#--------------------------------------- 4.1 支持向量机模型进行学习和预测 END ---------------------------------------

#--------------------------------------- 4.2 LinearRegression ---------------------------------------
# ....................................... 线性回归模型 .......................................
##加载线性回归模型
model_lr=linear_model.LinearRegression()
##将训练数据传入开始训练
model_lr.fit(x_train, y_train)

print(model_lr.coef_)     #系数，有些模型没有系数（如k近邻）
print(model_lr.intercept_) #与y轴交点，即截距

# 交叉验证
predicted = cross_val_predict(model_lr, x, y, cv=10)
print("使用交叉验证的均方误差为:",metrics.mean_squared_error(y, predicted))
'''
我们可以发现这里的均方误差比上面的大，那是因为上面只针对测试集求了MSE，而这里对每一折的测试集都求了MSE。
'''
# ....................................... 线性回归模型 END .......................................

# ....................................... Ridge Regression模型  .......................................
model_ridge=linear_model.Ridge(alpha = .5)
model_ridge.fit(x_train, y_train)

print(model_ridge.coef_)
print(model_ridge.intercept_)
# ....................................... Ridge Regression模型  END .......................................

# 最后我们可以用图像来直观表示预测值与真实值的关系
plt.figure('model')
plt.plot(y, predicted, '.')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.scatter(y, predicted)
plt.show()
'''
离虚线越近的点误差越小
'''

#--------------------------------------- 4.2 LinearRegression END ---------------------------------------


#======================================== 4 建模预测 END ========================================


#======================================== 5 模型评估 ========================================
# 线性核函数模型评估
print("线性核函数支持向量机的默认评估值为：\n", linear_svr.score(x_test, y_test))
print("线性核函数支持向量机的R_squared值为：\n", r2_score(y_test, linear_svr_y_predict))
print("线性核函数支持向量机的均方误差(MSE)为: \n", mean_squared_error(ss_y.inverse_transform(y_test),
                                              ss_y.inverse_transform(linear_svr_y_predict)) )
print("线性核函数支持向量机的平均绝对误差为: \n", mean_absolute_error(ss_y.inverse_transform(y_test),
                                                 ss_y.inverse_transform(linear_svr_y_predict)))

# 对多项式核函数模型评估
print("对多项式核函数的默认评估值为：\n", poly_svr.score(x_test, y_test))
print("对多项式核函数的R_squared值为：\n", r2_score(y_test, poly_svr_y_predict))
print("对多项式核函数的均方误差为: \n", mean_squared_error(ss_y.inverse_transform(y_test),
                                           ss_y.inverse_transform(poly_svr_y_predict)))
print("对多项式核函数的平均绝对误差为: \n", mean_absolute_error(ss_y.inverse_transform(y_test),
                                              ss_y.inverse_transform(poly_svr_y_predict)))

# 对rbf核函数模型评估
print("对rbf核函数的默认评估值为：\n", rbf_svr.score(x_test, y_test))
print("对rbf核函数的R_squared值为：\n", r2_score(y_test, rbf_svr_y_predict))
print("对rbf核函数的均方误差为: \n", mean_squared_error(ss_y.inverse_transform(y_test),
                                           ss_y.inverse_transform(rbf_svr_y_predict)))
print("对rbf核函数的平均绝对误差为: \n", mean_absolute_error(ss_y.inverse_transform(y_test),
                                              ss_y.inverse_transform(rbf_svr_y_predict)))
#======================================== 5 模型评估 END ========================================




'''
线性核函数支持向量机的默认评估值为： 0.651717097429608
线性核函数支持向量机的R_squared值为： 0.651717097429608
线性核函数支持向量机的均方误差为: 27.0063071393243
线性核函数支持向量机的平均绝对误差为: 3.426672916872753
对多项式核函数的默认评估值为： 0.40445405800289286
对多项式核函数的R_squared值为： 0.651717097429608
对多项式核函数的均方误差为: 27.0063071393243
对多项式核函数的平均绝对误差为: 3.426672916872753
'''

