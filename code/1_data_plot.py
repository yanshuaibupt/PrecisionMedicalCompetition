#!/usr/bin/env python
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from sklearn import model_selection
from sklearn.linear_model import LinearRegression,RidgeCV,LassoCV
from sklearn.ensemble import BaggingRegressor,AdaBoostRegressor
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']

data_train = pd.read_csv("C:/Tool/Pycharm/TianChi/d_train_20180102.csv",encoding='gb2312')

def set_missing_message(df,column):
    df[column].fillna(df[column].median(),inplace=True)
    return

def error(actual,predicted):
    return np.sum(np.square(actual-predicted))/(2*len(actual))

def log_transform(feature):
    data_train[feature] = np.log1p(data_train[feature].values)

columns = ['年龄','*天门冬氨酸氨基转换酶','*丙氨酸氨基转换酶','*碱性磷酸酶','*r-谷氨酰基转换酶','*总蛋白','白蛋白','*球蛋白','白球比例','甘油三酯','总胆固醇','高密度脂蛋白胆固醇','低密度脂蛋白胆固醇','尿素','肌酐',
           '尿酸','白细胞计数','红细胞计数','血红蛋白','红细胞压积','红细胞平均体积','红细胞平均血红蛋白量','红细胞平均血红蛋白浓度','红细胞体积分布宽度','血小板计数','血小板平均体积','血小板体积分布宽度','血小板比积',
           '中性粒细胞%','淋巴细胞%','单核细胞%','嗜酸细胞%','嗜碱细胞%','血糖']
for item in columns:
    set_missing_message(data_train,item)


# data_train[columns] = np.square(data_train[columns].values)
for item in columns:
    log_transform(item)

# 特征归一化
scaler = preprocessing.StandardScaler()
for item in columns:
    data_train[item] = scaler.fit_transform(data_train[item].values.reshape(-1,1))

# 特征因子化
dummies_Sex = pd.get_dummies(data_train['性别'],prefix='性别')
data_train = pd.concat([data_train,dummies_Sex],axis=1)

# fig,axes = plt.subplots(nrows=1,ncols=10)
# for i in range(10):
#     data_train.plot.scatter(x=columns[i],y='血糖',ax=axes[i])
# data_train.plot.scatter(x='血红蛋白',y='血糖',ax=axes[0,0])

melt_X = pd.melt(data_train, value_vars=columns)
g = sns.FacetGrid(melt_X, col="variable",  col_wrap=5, size=1,sharex=False, sharey=False)
g = g.map(sns.distplot, "value")  # 以melt_X['value']作为数据
plt.show()

# 数据分割
df_train,df_cv = model_selection.train_test_split(data_train,test_size=0.3,random_state=0)
columns.append('性别_男')
columns.append('性别_女')
features = ('|'.join(columns)).replace('*','[a-zA-Z0-9]*')
train_X = df_train.filter(regex=features)
train_y = df_train['血糖']
cv_X = df_cv.filter(regex=features)
cv_y = df_cv['血糖']
