# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from sklearn import model_selection
from sklearn.linear_model import LinearRegression,RidgeCV,LassoCV
from sklearn.ensemble import BaggingRegressor,AdaBoostRegressor
plt.rcParams['font.sans-serif'] = ['SimHei']

data_train = pd.read_csv("C:/Tool/Pycharm/TianChi/d_train_2018010.csv",encoding='gb2312')

def set_missing_message(df,column):
    df[column].fillna(df[column].mean(),inplace=True)
    return

def error(actual,predicted):
    return np.sum(np.square(actual-predicted))/(2*len(actual))

columns = ['年龄','*天门冬氨酸氨基转换酶','*丙氨酸氨基转换酶','*碱性磷酸酶','*r-谷氨酰基转换酶','*总蛋白','白蛋白','*球蛋白','白球比例','甘油三酯','总胆固醇','高密度脂蛋白胆固醇','低密度脂蛋白胆固醇','尿素','肌酐',
           '尿酸','白细胞计数','红细胞计数','血红蛋白','红细胞压积','红细胞平均体积','红细胞平均血红蛋白量','红细胞平均血红蛋白浓度','红细胞体积分布宽度','血小板计数','血小板平均体积','血小板体积分布宽度','血小板比积',
           '中性粒细胞%','淋巴细胞%','单核细胞%','嗜酸细胞%','嗜碱细胞%']
for item in columns:
    set_missing_message(data_train,item)

# data_train[columns] = np.square(np.log1p(data_train[columns].values))
#
# # 特征归一化
# scaler = preprocessing.StandardScaler()
# for item in columns:
#     data_train[item] = scaler.fit_transform(data_train[item].values.reshape(-1,1))

# 特征因子化
dummies_Sex = pd.get_dummies(data_train['性别'],prefix='性别')
data_train = pd.concat([data_train,dummies_Sex],axis=1)

# 数据分割
df_train,df_cv = model_selection.train_test_split(data_train,test_size=0.3,random_state=0)

# df_train['血糖'] = (np.log(df_train['血糖'].values))
columns.append('性别_男')
columns.append('性别_女')
features = ('|'.join(columns)).replace('*','[a-zA-Z0-9]*')
train_X = df_train.filter(regex=features)
train_y = df_train['血糖']
cv_X = df_cv.filter(regex=features)
cv_y = df_cv['血糖']

# 线性回归模型
lr = LinearRegression()
lr.fit(train_X,(train_y))
print('lr_error:',error(cv_y,lr.predict(cv_X)))

# 岭回归
ridge = RidgeCV(alphas=[151],cv=10)
ridge.fit(train_X,train_y)
print('ridge_error:',error(cv_y,ridge.predict(cv_X)))

lasso = LassoCV(alphas=[0.003],max_iter=10000,cv=10)
lasso.fit(train_X,train_y)
print('lasso_error:',error(cv_y,np.exp(lasso.predict(cv_X))))

br_lr = BaggingRegressor(base_estimator = lr,n_estimators = 3)
br_lr.fit(train_X,train_y)
print('br_lr_error:',error(cv_y,br_lr.predict(cv_X)))

br_ridge = BaggingRegressor(base_estimator = ridge,n_estimators = 3)
br_ridge.fit(train_X,train_y)
print('br_ridge_error:',error(cv_y,br_ridge.predict(cv_X)))

br_lasso = BaggingRegressor(base_estimator = lasso,n_estimators = 7)
br_lasso.fit(train_X,train_y)
print('br_lasso_error:',error(cv_y,br_lasso.predict(cv_X)))

# 测试集预处理
data_test = pd.read_csv("C:/Tool/Pycharm/TianChi/d_test_A_20180102.csv",encoding='gb2312')
test_columns = ['年龄','*天门冬氨酸氨基转换酶','*丙氨酸氨基转换酶','*碱性磷酸酶','*r-谷氨酰基转换酶','*总蛋白','白蛋白','*球蛋白','白球比例','甘油三酯','总胆固醇','高密度脂蛋白胆固醇','低密度脂蛋白胆固醇','尿素','肌酐',
           '尿酸','白细胞计数','红细胞计数','血红蛋白','红细胞压积','红细胞平均体积','红细胞平均血红蛋白量','红细胞平均血红蛋白浓度','红细胞体积分布宽度','血小板计数','血小板平均体积','血小板体积分布宽度','血小板比积',
           '中性粒细胞%','淋巴细胞%','单核细胞%','嗜酸细胞%','嗜碱细胞%']
for item in test_columns:
    set_missing_message(data_test,item)
# data_test[test_columns] = np.square(np.log1p(data_test[test_columns].values))
# for item in test_columns:
#     data_test[item] = scaler.fit_transform(data_test[item].values.reshape(-1, 1))
test_dummies_Sex = pd.get_dummies(data_test['性别'],prefix='性别')
data_test = pd.concat([data_test,test_dummies_Sex],axis=1)
test_columns.append('性别_男')
test_columns.append('性别_女')
test_features = ('|'.join(test_columns)).replace('*','[a-zA-Z0-9]*')
test_X = data_test.filter(regex=features)
result = pd.DataFrame(lr.predict(test_X))
# result.to_csv("C:/Tool/Pycharm/TianChi/lr.csv",header=None,index=None)
# print(result)