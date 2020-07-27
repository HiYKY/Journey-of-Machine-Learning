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
# relative path ../ 表示当前文件所在的目录的上一级目录
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
# -------------------------- 2. Standardize features by removing the mean and scaling to unit variance END ----------------------------------------------------

# ------------------------ 3. Splitting the Dataset into the Training set and Test set ------------------------------
from sklearn.model_selection import train_test_split, cross_val_score
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=824)
# ------------------------ 3. Splitting the Dataset into the Training set and Test set END ------------------------------
# ============================== Data Pre-processing END ===================================================


# ============================== Building Machine Learning Models ===================================================
# -------------------------------------------------------------------------------------------------------------------
# ------------------------ build traditional models and evaluate the model ------------------------------
# -------------------------------------------------------------------------------------------------------------------

# ................ (1)Logistic Regression ................
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(x_train, y_train)
y_pred_log_reg = clf.predict(x_test)
acc_log_reg = round(clf.score(x_train, y_train) * 100, 2)
print (str(acc_log_reg) + ' %')
# ................ (1)Logistic Regression END ................

# ................ (2)Support Vector Machine (SVM) ................
from sklearn.svm import SVC

clf = SVC()
clf.fit(x_train, y_train)
y_pred_svc = clf.predict(x_test)
acc_svc = round(clf.score(x_train, y_train) * 100, 2)
print (str(acc_svc) + '%')

# Linear SVM
from sklearn.svm import SVC, LinearSVC

clf = LinearSVC()
clf.fit(x_train, y_train)
y_pred_linear_svc = clf.predict(x_test)
acc_linear_svc = round(clf.score(x_train, y_train) * 100, 2)
print (str(acc_linear_svc) + '%')
# ................ (2)Support Vector Machine (SVM) END ................

# ................ (3)k-Nearest Neighbors ................
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()
clf.fit(x_train, y_train)
y_pred_knn = clf.predict(x_test)
acc_knn = round(clf.score(x_train, y_train) * 100, 2)
print (str(acc_knn)+'%')
# ................ (3)k-Nearest Neighbors END ................

# ................ (4)Gaussian Naive Bayes ................
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(x_train, y_train)
y_pred_gnb = clf.predict(x_test)
acc_gnb = round(clf.score(x_train, y_train) * 100, 2)
print (str(acc_gnb) + '%')
# ................ (4)Gaussian Naive Bayes END ................

# ................ (5)Principal Component Analysis (PCA) ................
from sklearn.decomposition import PCA

pca = PCA()
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)
acc_PCA = round(pca.score(x_train, x_test) )
print (str(acc_PCA) + '%')
# ................ (5)Principal Component Analysis (PCA) END ................

# .......................... (6)SGD ..........................
# refered from: https://www.kaggle.com/m0nika/ml-seizure

from evaluate_metrix import *
from sklearn.linear_model import SGDClassifier

import warnings
warnings.filterwarnings('ignore')

x_valid = x_train
y_valid = y_train

sgdc= SGDClassifier(loss = 'log',alpha = 0.1)
sgdc.fit(x_train, y_train)

y_train_preds = sgdc.predict_proba(x_train)[:,1]
y_valid_preds = sgdc.predict_proba(x_train)[:,1]

print('SGDC')
print('Training:')
sgdc_train_auc, sgdc_train_accuracy, sgdc_train_recall, \
    sgdc_train_precision, sgdc_train_specificity = print_report(y_train,y_train_preds, thresh)
print('Validation:')
sgdc_valid_auc, sgdc_valid_accuracy, sgdc_valid_recall, \
    sgdc_valid_precision, sgdc_valid_specificity = print_report(y_valid,y_valid_preds, thresh)
# .......................... (6)SGD END ..........................

# .......................... (7)DecisionTree ..........................
from evaluate_metrix import *
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth = 10, random_state = 69)
tree.fit(x_train, y_train)

x_valid = x_train
y_valid = y_train

y_train_preds = tree.predict_proba(x_train)[:,1]
y_valid_preds = tree.predict_proba(x_valid)[:,1]

print('Decision Tree')
print('Training:')
tree_train_auc, tree_train_accuracy, tree_train_recall, tree_train_precision, \
tree_train_specificity =print_report(y_train,y_train_preds, thresh)
print('Validation:')
tree_valid_auc, tree_valid_accuracy, tree_valid_recall, tree_valid_precision, \
tree_valid_specificity = print_report(y_valid,y_valid_preds, thresh)
# .......................... (7)DecisionTree END ..........................

# .......................... (8)RandomForest ..........................
from evaluate_metrix import *
from sklearn.ensemble import RandomForestClassifier

x_valid = x_train
y_valid = y_train

rf = RandomForestClassifier(max_depth = 6, random_state = 69)
rf.fit(x_train, y_train)

y_train_preds = rf.predict_proba(x_train)[:,1]
y_valid_preds = rf.predict_proba(x_valid)[:,1]

print('Random Forest')
print('Training:')
rf_train_auc, rf_train_accuracy, rf_train_recall, rf_train_precision, \
rf_train_specificity =print_report(y_train,y_train_preds, thresh)
print('Validation:')
rf_valid_auc, rf_valid_accuracy, rf_valid_recall, rf_valid_precision, \
rf_valid_specificity = print_report(y_valid,y_valid_preds, thresh)
# .......................... (8)RandomForest END ..........................

# .......................... (9)GradientBoosting ..........................
from evaluate_metrix import *
from sklearn.ensemble import GradientBoostingClassifier

x_valid = x_train
y_valid = y_train

gbc = GradientBoostingClassifier(
    n_estimators=100, learning_rate=1.0, max_depth=3, random_state=69)
gbc.fit(x_train, y_train)

y_train_preds = gbc.predict_proba(x_train)[:,1]
y_valid_preds = gbc.predict_proba(x_valid)[:,1]

print('Gradient Boosting Classifier')
print('Training:')
gbc_train_auc, gbc_train_accuracy, gbc_train_recall, gbc_train_precision, \
gbc_train_specificity = print_report(y_train,y_train_preds, thresh)
print('Validation:')
gbc_valid_auc, gbc_valid_accuracy, gbc_valid_recall, gbc_valid_precision, \
gbc_valid_specificity = print_report(y_valid,y_valid_preds, thresh)
# .......................... (9)GradientBoosting END ..........................

# .......................... (10)ExtraTrees ..........................
from evaluate_metrix import *
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

x_valid = x_train
y_valid = y_train

etc = ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=1.0,
                           min_samples_leaf=3, min_samples_split=20, n_estimators=100)
etc.fit(x_train, y_train)

y_train_preds = etc.predict_proba(x_train)[:, 1]
y_valid_preds = etc.predict_proba(x_valid)[:, 1]

print('Extra Trees Classifier')
print('Training:')
etc_train_auc, etc_train_accuracy, etc_train_recall, etc_train_precision, \
etc_train_specificity = print_report(y_train, y_train_preds, thresh)
print('Validation:')
etc_valid_auc, etc_valid_accuracy, etc_valid_recall, etc_valid_precision, \
etc_valid_specificity = print_report(y_valid, y_valid_preds, thresh)
# .......................... (10)ExtraTrees END ..........................

# .......................... (11)xgboost ..........................
from evaluate_metrix import *
from xgboost import XGBClassifier
import xgboost as xgb

x_valid = x_train
y_valid = y_train

xgbc = XGBClassifier()
xgbc.fit(x_train, y_train)

y_train_preds = xgbc.predict_proba(x_train)[:,1]
y_valid_preds = xgbc.predict_proba(x_valid)[:,1]

print('Xtreme Gradient Boosting Classifier')
print('Training:')
xgbc_train_auc, xgbc_train_accuracy, xgbc_train_recall, xgbc_train_precision, \
xgbc_train_specificity = print_report(y_train,y_train_preds, thresh)
print('Validation:')
xgbc_valid_auc, xgbc_valid_accuracy, xgbc_valid_recall, xgbc_valid_precision, \
xgbc_valid_specificity = print_report(y_valid,y_valid_preds, thresh)
# .......................... (11)xgboost END ..........................

# -------------------------------------------------------------------------------------------------------------------
# ------------------------ build traditional models and evaluate the model END ------------------------------
# -------------------------------------------------------------------------------------------------------------------

# ============================== Building Machine Learning Models END ===================================================


# ============================== Compare different models ===================================================
#df_results = pd.DataFrame({'classifier':['SGD','SGD','NB','NB','DT','DT','RF','RF','GB','GB','XGBC','XGBC','ETC','ETC'],
#                          'data_set':['train','valid']*9,
#                          'auc':[knn_train_auc, knn_valid_auc,lr_train_auc,lr_valid_auc,sgdc_train_auc,sgdc_valid_auc,nb_train_auc,nb_valid_auc,tree_train_auc,tree_valid_auc,rf_train_auc,rf_valid_auc,gbc_train_auc,gbc_valid_auc,xgbc_train_auc,xgbc_valid_auc,etc_train_auc,etc_valid_auc],
#                          'accuracy':[knn_train_accuracy, knn_valid_accuracy,lr_train_accuracy,lr_valid_accuracy,sgdc_train_accuracy,sgdc_valid_accuracy,nb_train_accuracy,nb_valid_accuracy,tree_train_accuracy,tree_valid_accuracy,rf_train_accuracy,rf_valid_accuracy,gbc_train_accuracy,gbc_valid_accuracy,xgbc_train_accuracy,xgbc_valid_accuracy,etc_train_accuracy,etc_valid_accuracy],
#                          'recall':[knn_train_recall, knn_valid_recall,lr_train_recall,lr_valid_recall,sgdc_train_recall,sgdc_valid_recall,nb_train_recall,nb_valid_recall,tree_train_recall,tree_valid_recall,rf_train_recall,rf_valid_recall,gbc_train_recall,gbc_valid_recall,xgbc_train_recall,xgbc_valid_recall,etc_train_recall,etc_valid_recall],
#                          'precision':[knn_train_precision, knn_valid_precision,lr_train_precision,lr_valid_precision,sgdc_train_precision,sgdc_valid_precision,nb_train_precision,nb_valid_precision,tree_train_precision,tree_valid_precision,rf_train_precision,rf_valid_precision,gbc_train_precision,gbc_valid_precision,xgbc_train_precision,xgbc_valid_precision,etc_train_precision,etc_valid_precision],
#                          'specificity':[knn_train_specificity, knn_valid_specificity,lr_train_specificity,lr_valid_specificity,sgdc_train_specificity,sgdc_valid_specificity,nb_train_specificity,nb_valid_specificity,tree_train_specificity,tree_valid_specificity,rf_train_specificity,rf_valid_specificity,gbc_train_specificity,gbc_valid_specificity,xgbc_train_specificity,xgbc_valid_specificity,etc_train_specificity,etc_valid_specificity]})

df_results = pd.DataFrame({'classifier':['SGD','SGD','DT','DT','RF','RF','GB','GB','XGBC','XGBC','ETC','ETC'],
                          'data_set':['train','valid']*6,
                          'auc':[sgdc_train_auc,sgdc_valid_auc,tree_train_auc,tree_valid_auc,rf_train_auc,rf_valid_auc,gbc_train_auc,gbc_valid_auc,xgbc_train_auc,xgbc_valid_auc,etc_train_auc,etc_valid_auc],
                          'accuracy':[sgdc_train_accuracy,sgdc_valid_accuracy,tree_train_accuracy,tree_valid_accuracy,rf_train_accuracy,rf_valid_accuracy,gbc_train_accuracy,gbc_valid_accuracy,xgbc_train_accuracy,xgbc_valid_accuracy,etc_train_accuracy,etc_valid_accuracy],
                          'recall':[sgdc_train_recall,sgdc_valid_recall,tree_train_recall,tree_valid_recall,rf_train_recall,rf_valid_recall,gbc_train_recall,gbc_valid_recall,xgbc_train_recall,xgbc_valid_recall,etc_train_recall,etc_valid_recall],
                          'precision':[sgdc_train_precision,sgdc_valid_precision,tree_train_precision,tree_valid_precision,rf_train_precision,rf_valid_precision,gbc_train_precision,gbc_valid_precision,xgbc_train_precision,xgbc_valid_precision,etc_train_precision,etc_valid_precision],
                          'specificity':[sgdc_train_specificity,sgdc_valid_specificity,tree_train_specificity,tree_valid_specificity,rf_train_specificity,rf_valid_specificity,gbc_train_specificity,gbc_valid_specificity,xgbc_train_specificity,xgbc_valid_specificity,etc_train_specificity,etc_valid_specificity]})

#SGD DecisionTree RandomForest GradientBoosting ExtraTrees xgboost

import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline
sns.set(style="whitegrid")

sns.set_style("whitegrid")
plt.figure(figsize=(16, 8))
ax = sns.barplot(x = 'classifier', y = 'auc', hue = 'data_set', data = df_results)
ax.set_xlabel('Classifier', fontsize = 15)
ax.set_ylabel('AUC', fontsize = 15)
ax.tick_params(labelsize = 15)

#Separate legend from graph
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0., fontsize = 15)
#保存图片
plt.savefig('Comparationof different models.png', dpi=1080 ,bbox_inches='tight')
# ============================== Compare different models END ===================================================


# ============================== Compare feature importances ===================================================
df = ESR

df["OUTPUT_LABEL"] = df.y == 1
df["OUTPUT_LABEL"] = df["OUTPUT_LABEL"].astype(int)
df.pop('y')
df.drop(df.columns[0], axis=1, inplace=True)

collist = df.columns.tolist()
cols_input = collist[0:178]
df_data = df[cols_input + ["OUTPUT_LABEL"]]

'''
# check for duplicated columns in cols_input
dup_cols = set([x for x in cols_input if cols_input.count(x) > 1])
print(dup_cols)
assert len(dup_cols) == 0, "you have duplicated columns in cols_input"

# check for duplicated columns in df_data
cols_df_data = list(df_data.columns)
dup_cols = set([x for x in cols_df_data if cols_df_data.count(x) > 1])
print(dup_cols)
assert len(dup_cols) == 0,'you have duplicated columns in df_data'

# check the size of df_data makes sense
assert (len(cols_input) + 1) == len(
    df_data.columns
), "issue with dimensions of df_data or cols_input"
'''

# ------------------------ ExtraTrees ------------------------------
feature_importances = pd.DataFrame(etc.feature_importances_,
                                   index = cols_input,
                                    columns=['importance']).sort_values('importance', ascending=False)

feature_importances.head()

pos_features = feature_importances.loc[feature_importances.importance > 0]

num = np.min([50, len(pos_features)])
ylocs = np.arange(num)
# get the feature importance for top num and sort in reverse order
values_to_plot = pos_features.iloc[:num].values.ravel()[::-1]
feature_labels = list(pos_features.iloc[:num].index)[::-1]

plt.figure(num=None, figsize=(8, 15), dpi=80, facecolor='w', edgecolor='k');
plt.barh(ylocs, values_to_plot, align = 'center')
plt.ylabel('Features')
plt.xlabel('Importance Score')
plt.title('Positive Feature Importance Score - ExtraTrees Classifier')
plt.yticks(ylocs, feature_labels)
plt.show()
# ------------------------ ExtraTrees END ------------------------------

# ------------------------ xgboost ------------------------------
ax = xgb.plot_importance(xgbc)
fig = ax.figure
fig.set_size_inches(10, 35)
#plt.savefig('xgbcf.png')

feature_importances = pd.DataFrame(xgbc.feature_importances_,
                                   index = cols_input,
                                    columns=['importance']).sort_values('importance',
                                                                        ascending=False)

pos_features = feature_importances.loc[feature_importances.importance > 0]

num = np.min([50, len(pos_features)])
ylocs = np.arange(num)
# get the feature importance for top num and sort in reverse order
values_to_plot = pos_features.iloc[:num].values.ravel()[::-1]
feature_labels = list(pos_features.iloc[:num].index)[::-1]

plt.figure(num=None, figsize=(8, 15), dpi=80, facecolor='w', edgecolor='k');
plt.barh(ylocs, values_to_plot, align = 'center')
plt.ylabel('Features')
plt.xlabel('Importance Score')
plt.title('Positive Feature Importance Score - XGBoost Classifier')
plt.yticks(ylocs, feature_labels)
#plt.savefig('xgbc.png')
plt.show()
# ------------------------ xgboost END ------------------------------

# ============================== Compare feature importances END ===================================================


# ============================== Feature selection and test ===================================================
import sklearn.metrics as metrik

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


from sklearn.feature_selection import SelectKBest, chi2
selekt15 = SelectKBest(chi2, k=15)
x_train_new = selekt15.fit_transform(x_train, y_train)
x_test_new = selekt15.transform(x_test)

# ------------------------ RandomForest ------------------------------
from sklearn.ensemble import RandomForestClassifier
ranforest = RandomForestClassifier()
ranforest.fit(x_train_new, y_train)
ypred = ranforest.predict(x_test_new)

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))
print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
# ------------------------ RandomForest END ------------------------------

# .......................... xgboost ..........................
from xgboost import XGBClassifier

xgbc = XGBClassifier()
xgbc.fit(x_train_new, y_train)
ypred = xgbc.predict(x_test_new)

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))
print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
# .......................... xgboost END ..........................

# .......................... ExtraTrees ..........................
from sklearn.ensemble import ExtraTreesClassifier

etc = ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=1.0,
                           min_samples_leaf=3, min_samples_split=20, n_estimators=100)
etc.fit(x_train_new, y_train)

ypred = xgbc.predict(x_test_new)

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))
print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
# .......................... ExtraTrees END ..........................

# .......................... GradientBoosting ..........................
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(
    n_estimators=100, learning_rate=1.0, max_depth=3, random_state=69)
gbc.fit(x_train_new, y_train)

ypred = xgbc.predict(x_test_new)

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))
print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
# .......................... GradientBoosting END ..........................

# .......................... SVM ..........................
from sklearn.svm import SVC

svm = SVC()
svm.fit(x_train_new, y_train)
ypred=svm.predict(x_test_new)

print(metrik.accuracy_score(y_pred=ypred,y_true=y_test))
print(metrik.confusion_matrix(y_pred=ypred,y_true=y_test))
# .......................... SVM END ..........................

# ============================== Feature selection and test END ===================================================

