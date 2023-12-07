# -*- coding: utf-8 -*-
"""
@Time ： 2023/2/16 21:26
@Auth ： 兰宏富
@File ：adaboost_decisiontree_gridsearch.py
@IDE ：PyCharm
"""
import sklearn.ensemble
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, GradientBoostingRegressor, \
    StackingRegressor, StackingClassifier, RandomForestClassifier  # AdaBoost分类器
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier  # 决策树分类器
import sklearn.metrics.pairwise

# --------------------------------------数据集读取原始版本------------------------------------------
# wine = load_wine()
# print(f"所有特征：{wine.feature_names}")
# X = pd.DataFrame(wine.data, columns=wine.feature_names)
# y = pd.Series(wine.target)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
# --------------------------------------数据集读取原始版本------------------------------------------


# --------------------------------------数据集读取我的版本start------------------------------------------
# model_name = 'distilbert-base-uncased'
model_name = 'bert-base-uncased'
csn_path = '../dataset/Commit_dataset_final.xlsx'

import logging

logging.basicConfig(level=logging.INFO)
import torch
from transformers import AutoTokenizer, \
    AutoModel

# Load pre-trained model tokenizer (vocabulary)
tokenizer = AutoTokenizer.from_pretrained(model_name)

import pandas as pd

df = pd.read_excel(csn_path,
                   header=0,  # 不读取第一行的表头
                   index_col=False,  # 设置为False就会读取第一列，设置为0不会读取第一列
                   )
# Tokenize input
tokenized_text = df['comment'].apply(
    (lambda x: tokenizer.encode(x, max_length=80, add_special_tokens=True, truncation=True)))

import numpy as np

max_len = 0
for i in tokenized_text.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized_text.values])

np.array(padded).shape
attention_mask = np.where(padded != 0, 1, 0)
attention_mask.shape

model = AutoModel.from_pretrained(model_name)
input_ids = torch.tensor(padded)
attention_mask = torch.tensor(attention_mask)

with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

bert_features = last_hidden_states[0][:, 0, :].numpy()  # BERT的输出：768维度的语义向量
print(bert_features.shape)
bert_features.shape

change = {'p': 0, 'a': 1, 'c': 2}  # 替换的值
labels = df['3_labels'].map(change).values  # 标签值(类型为numpy array)
cc = df.drop(columns=['3_labels', 'comment', 'project', 'commit_id', 'comment_diff'], axis=1).replace(np.nan,
                                                                                                      0)  # 将空的格子设置为0
print(cc)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
cc = sc.fit_transform(cc)
print(cc)

all_input_features = np.concatenate((bert_features, cc), axis=1)  # 拼接语义和特征
all_input_features[0].shape

# encoder = LabelBinarizer() # 转换成独热编码
# labels = encoder.fit_transform(labels)
# print(labels)

from sklearn.model_selection import train_test_split, cross_val_score

# stratified train_test_split
X_train, X_test, y_train, y_test = train_test_split(all_input_features, labels, test_size=0.2, random_state=42,
                                                    stratify=labels)  # 80% 训练，20%测试结果

# --------------------------------------数据集读取我的版本end------------------------------------------


# --------------------------------------预测的代码start------------------------------------------
# 实例化
model1 = sklearn.svm.SVC(probability=True)  # SVM
model2 = KNeighborsClassifier()  # KNN
model3 = GradientBoostingClassifier(random_state=37)  # GBDT
model4 = DecisionTreeClassifier(random_state=666)  # DT
model5 = LGBMClassifier(learning_rate=0.1, n_estimators=100)  # GBM
model6 = MLPClassifier(hidden_layer_sizes=(100,), random_state=420, max_iter=1000)  # MLP
model7 = LogisticRegression(max_iter=1000)  # LogisticRegression
model8 = RandomForestClassifier()  # RF随机森林

LIST = []

model1.fit(X_train, y_train)
model1_preds = model1.predict(X_test)
LIST.append(model1_preds)
model2.fit(X_train, y_train)
model2_preds = model2.predict(X_test)
LIST.append(model2_preds)
model3.fit(X_train, y_train)
model3_preds = model3.predict(X_test)
LIST.append(model3_preds)
model4.fit(X_train, y_train)
model4_preds = model4.predict(X_test)
LIST.append(model4_preds)
model5.fit(X_train, y_train)
model5_preds = model5.predict(X_test)
LIST.append(model5_preds)
model6.fit(X_train, y_train)
model6_preds = model6.predict(X_test)
LIST.append(model6_preds)
model7.fit(X_train, y_train)
model7_preds = model7.predict(X_test)
LIST.append(model7_preds)
model8.fit(X_train, y_train)
model8_preds = model8.predict(X_test)
LIST.append(model8_preds)


def get_matrix(model1_predsS, model2_predsS):
    q, w, e, r = 0, 0, 0, 0
    for i in range(len(model1_predsS)):
        A = model1_predsS[i]
        B = model2_predsS[i]
        # A对B对
        if ((A == y_test[i]) == True) and ((B == y_test[i]) == True):
            q += 1
        # A对B错
        if ((A == y_test[i]) == True) and ((B == y_test[i]) == False):
            w += 1
        # A错B对
        if ((A == y_test[i]) == False) and ((B == y_test[i]) == True):
            e += 1
        # A错B错
        if ((A == y_test[i]) == False) and ((B == y_test[i]) == False):
            r += 1
    return q, w, e, r

print("Q值:")
for i in LIST:
    index = 0
    for j in LIST:
        index += 1
        a, c, b, d = get_matrix(i, j)
        Q= (a*d-b*c)/(a*d+b*c)
        if Q == 0:
            Q = 1
        print(format(Q, '.2f'), "", end='')
        if index == 8:
            print("")

print("分歧度量:")


for i in LIST:
    index = 0
    for j in LIST:
        index += 1
        a, c, b, d = get_matrix(i, j)
        FQ = (b+c) / len(model8_preds)  # 计算FQ
        print(format(FQ, '.2f'), "", end='')
        if index == 8:
            print("")

print("DF:")

for i in LIST:
    index = 0
    for j in LIST:
        index += 1
        a, c, b, d = get_matrix(i, j)
        DF = d / len(model8_preds)  # 计算DF
        print(format(DF, '.2f'), "", end='')
        if index == 8:
            print("")


