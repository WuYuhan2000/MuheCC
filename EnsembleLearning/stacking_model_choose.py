# -*- coding: utf-8 -*-
"""
@Time ： 2023/2/23 22:13
@Auth ： 兰宏富
@File ：stacking_model_choose.py
@IDE ：PyCharm
用来进行stacking基模型选择
"""
import sklearn.ensemble
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, GradientBoostingRegressor, \
    StackingRegressor, StackingClassifier, RandomForestClassifier, AdaBoostClassifier  # AdaBoost分类器
from sklearn.linear_model import LogisticRegression, RidgeCV, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier  # 决策树分类器
import itertools
import time
# model_name = 'huggingface/CodeBERTa-small-v1'  # 准确率最高
model_name = 'distilbert-base-uncased'  # 准确率第二
# model_name = 'bert-base-uncased'  # 准确率第三
csn_path = 'Commit_dataset_final.xlsx'

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
test_project = df[df['project'] == 'JetBrains/intellij-community']
start_time = time.time()
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

# labels = df['3_labels']
# print(labels)
change = {'p': 0, 'a': 1, 'c': 2}  # 替换的值
labels = df['3_labels'].map(change).values  # 标签值
cc = df.drop(columns=['3_labels', 'comment', 'project', 'commit_id', 'comment_diff'], axis=1).replace(np.nan,
                                                                                                      0)  # 将空的格子设置为0
print(cc)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
cc = sc.fit_transform(cc)
print(cc)


# all_input_features = cc
all_input_features = np.concatenate((bert_features, cc), axis=1)  # 拼接语义和特征
all_input_features[0].shape

# encoder = LabelBinarizer() # 转换成独热编码
# labels = encoder.fit_transform(labels)
# print(labels)

from sklearn.model_selection import train_test_split, cross_val_score

test_project_labels = test_project['3_labels'].map(change).values
# Tokenize input
tokenized_text = test_project['comment'].apply(
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

bert_features_project = last_hidden_states[0][:, 0, :].numpy()  # BERT的输出：768维度的语义向量
test_project = test_project.drop(columns=['3_labels', 'comment', 'project', 'commit_id', 'comment_diff'], axis=1).replace(np.nan,
                                                                                                      0)  # 将空的格子设置为0

test_project_all_feature = np.concatenate((bert_features_project, test_project), axis=1)
X_train, X_test_1, y_train, y_test_1 = train_test_split(test_project_all_feature, test_project_labels, test_size=0.2, random_state=42,
                                                    stratify=test_project_labels)




# stratified train_test_split
X_train, X_test, y_train, y_test = train_test_split(all_input_features, labels, test_size=0.2, random_state=42,
                                                    stratify=labels)  # 80% 训练，20%测试结果

para_decision_tree = {'max_depth': 899, 'min_samples_leaf': 0.0275, 'min_samples_split': 0.0236}
para_logistic_regression = {'C': 1}
para_svm = {'C': 1.8}
para_knn = {'leaf_size': 10, 'n_neighbors': 25}
para_native_bayes = {'var_smoothing': 3}
para_ridge = {'alpha': 2.0}
para_SGD = {'alpha': 2, 'l1_ratio': 0}
para_MLP = {'hidden_layer_sizes': 50, 'max_iter': 500}

para_random_forest = {'max_depth': 14, 'n_estimators': 27}
para_adaboost = {'learning_rate': 0.15, 'n_estimators': 27}
para_GBDT = {'learning_rate': 0.15, 'n_estimators': 40}
para_lightGBM = {'learning_rate': 0.1218080101678672, 'n_estimators': 20}

MODEL1 = DecisionTreeClassifier(**para_decision_tree)
MODEL2 = LogisticRegression(**para_logistic_regression, max_iter=1000, n_jobs=-1)
MODEL3 = sklearn.svm.SVC(**para_svm, probability=False)
MODEL4 = KNeighborsClassifier(**para_knn,n_jobs=-1)
MODEL5 = GaussianNB(**para_native_bayes)
MODEL6 = RidgeClassifier(**para_ridge)
MODEL7 = SGDClassifier(**para_SGD,n_jobs=-1)
MODEL8 = MLPClassifier(**para_MLP)

MODEL_1 = RandomForestClassifier(**para_random_forest, n_jobs=-1)
MODEL_2 = AdaBoostClassifier(**para_adaboost)
MODEL_3 = GradientBoostingClassifier(**para_GBDT)
MODEL_4 = LGBMClassifier(**para_lightGBM, n_jobs=-1)

# 开始写模型选择的代码
# model_list = [  # 传统分类器
#     ('DecisionTreeClassifier', MODEL1),
#     ('LogisticRegression', MODEL2),
#     ('svm', MODEL3),
#     ('KNeighborsClassifier', MODEL4),
#     ('GaussianNB', MODEL5),
#     ('RidgeClassifier', MODEL6),
#     ('SGDClassifier', MODEL7),
#     ('MLPClassifier', MODEL8),
#
# ]

# model_list = [  # 集成分类器
#     ('RandomForestClassifier', MODEL_1),
#     ('AdaBoostClassifier', MODEL_2),
#     ('GradientBoostingClassifier', MODEL_3),
#     ('LGBMClassifier', MODEL_4),
# ]


# x = 0  # 用来统计组合的数量
# best_acc = -100  # 用来保存最好的best_acc
# best_models = []  # 用来保存最好的模型组合
#
# for i in range(len(model_list)):
#     if i + 1 == 1:
#         continue
#     print("组合的个数为：", i + 1)
#     model_combination = list(itertools.combinations(model_list, i + 1))  # 获取组合
#     for j in model_combination:
#         x += 1
#         print(x,list(j))
#         models = list(j)  # 把这个models传入stacking里面即可
#         reg = StackingClassifier(  # 集成学习分类器
#             estimators=models,
#             final_estimator=LogisticRegression(n_jobs=-1,max_iter=1000))
#         reg.fit(X_train, y_train)
#         acc_final = reg.score(X_test, y_test)
#         print(acc_final)
#         if acc_final>best_acc: # 保存最大准确率
#             best_acc = acc_final
#             best_models.clear()
#             best_models.append(models)
#         #  在这里就可以写训练的代码了，最好是封装成一个函数写在这里
#
# print("总的组合个数：", x)
# print("最优准确率",best_acc)
# print("最好模型",best_models)

# MODEL_1.fit(X_train, y_train)
# acc1 = MODEL_1.score(X_test, y_test)
# print("SVM", acc1)
#
# MODEL_2.fit(X_train, y_train)
# acc2 = MODEL_2.score(X_test, y_test)
# print("KNN", acc2)
#
# MODEL_3.fit(X_train, y_train)
# acc3 = MODEL_3.score(X_test, y_test)
# print("GBDT", acc3)
#
# MODEL_4.fit(X_train, y_train)
# acc4 = MODEL_4.score(X_test, y_test)
# print("DTREE", acc4)


st = [  # 调参完毕的模型

    #4种随机组合
    # ('GaussianNB', MODEL5),
    # ('LGBMClassifier', MODEL_4),
    # ('RandomForestClassifier', MODEL_1),
    # ('MLPClassifier', MODEL8),

    #最佳组合
    ('RidgeClassifier', MODEL6),
    ('LGBMClassifier', MODEL_4),
    # ('SGDClassifier', MODEL7),
    ('AdaBoostClassifier', MODEL_2),
    ('GradientBoostingClassifier', MODEL_3)
]
#
# # st = [  # 未调参的模型
# #     ('RidgeClassifier', RidgeClassifier()),
# #     ('SGDClassifier', SGDClassifier()),
# #     ('AdaBoostClassifier', AdaBoostClassifier()),
# #     ('GradientBoostingClassifier', GradientBoostingClassifier())
# # ]
#
# # 下面的代码用来测试挑选之前的模型的性能
for i in range(1, 11):
    reg = StackingClassifier(  # 集成学习分类器
        estimators=st,
        final_estimator=LogisticRegression(n_jobs=-1, max_iter=1000))
    # if i < 11:
    #     X_train, X_test, y_train, y_test = train_test_split(all_input_features, labels, test_size=0.2, random_state=40,
    #                                                     stratify=labels)  # 80% 训练，20%测试结果
    # elif i < 21:
    #     X_train, X_test, y_train, y_test = train_test_split(all_input_features, labels, test_size=0.2, random_state=39,
    #                                                     stratify=labels)  # 80% 训练，20%测试结果
    # elif i < 31:
    #     X_train, X_test, y_train, y_test = train_test_split(all_input_features, labels, test_size=0.2, random_state=41,
    #                                                     stratify=labels)  # 80% 训练，20%测试结果
    # elif i < 41:
    #     X_train, X_test, y_train, y_test = train_test_split(all_input_features, labels, test_size=0.2, random_state=43,
    #                                                     stratify=labels)  # 80% 训练，20%测试结果
    # else:
    X_train, X_test, y_train, y_test = train_test_split(all_input_features, labels, test_size=0.2, random_state=42,
                                                        stratify=labels)  # 80% 训练，20%测试结果
    reg.fit(X_train, y_train)
    acc_final = reg.score(X_test_1, y_test_1)
    test_predict = reg.predict(X_test_1)
    print("-------------第{}轮结果----------------".format(i))
    print("集成学习的结果是：", acc_final)
    end_time = time.time()
    time_consume = end_time - start_time
    print("模型运行总耗时：{}".format(time_consume))
    confusion_matrix_result = metrics.confusion_matrix(test_predict, y_test_1, labels=[0, 1, 2])
    print(confusion_matrix_result)
    print(classification_report(test_predict, y_test_1, digits=4))  # 保留四位小数点