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
from sklearn import metrics
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, GradientBoostingRegressor, \
    StackingRegressor, StackingClassifier, RandomForestClassifier  # AdaBoost分类器
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier  # 决策树分类器
import sys
import codecs
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
# --------------------------------------数据集读取原始版本------------------------------------------
# wine = load_wine()
# print(f"所有特征：{wine.feature_names}")
# X = pd.DataFrame(wine.data, columns=wine.feature_names)
# y = pd.Series(wine.target)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
# --------------------------------------数据集读取原始版本------------------------------------------


# --------------------------------------数据集读取我的版本start------------------------------------------
model_name = 'distilbert-base-uncased'  # 准确率第二
csn_path = 'Commit_dataset_final.xlsx'
from sklearn.model_selection import StratifiedKFold
import logging
import warnings
warnings.filterwarnings("ignore")


logging.basicConfig(level=logging.INFO)
import torch
from transformers import AutoTokenizer, \
    AutoModel

# Load pre-trained model tokenizer (vocabulary)
tokenizer = AutoTokenizer.from_pretrained(model_name)

import pandas as pd

df = pd.read_excel(csn_path,
                   header=0,  # 不读取第一行的表头
                   index_col=False,
                   # 设置为False就会读取第一列，设置为0不会读取第一列
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
print("before feature concatenate")
all_input_features = np.concatenate((bert_features, cc), axis=1)  # 拼接语义和特征
all_input_features[0].shape

# encoder = LabelBinarizer() # 转换成独热编码
# labels = encoder.fit_transform(labels)
# print(labels)

from sklearn.model_selection import train_test_split, cross_val_score
# #使用五折交叉验证
# # 定义折数（这里是五折交叉验证）
# k = 5
#
# # 创建 StratifiedKFold 对象，并指定折数 k
# skf = StratifiedKFold(n_splits=k)
#
# # 创建空列表存储每个折的训练集和验证集索引
# train_indices_list = []
# val_indices_list = []
#
# # 使用 StratifiedKFold 对象划分数据集
# for train_index, val_index in skf.split(all_input_features, labels):
#     train_indices_list.append(train_index)
#     val_indices_list.append(val_index)
    # #对每一折进行训练和验证
    # for fold in range(k):
    #     print("第{}次交叉验证\n".format(fold))
    #     train_indices = train_indices_list[fold]
    #     val_indices = val_indices_list[fold]
    #
    #     X_train, X_test = all_input_features[train_indices], all_input_features[val_indices]
    #     y_train, y_test = labels[train_indices], labels[val_indices]



for i in range(0, 21):
    # # stratified train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(all_input_features, labels, test_size=0.2, random_state=42,
    #                                                     stratify=labels)  # 80% 训练，20%测试结果
    X_train, X_test, y_train, y_test = train_test_split(all_input_features, labels, test_size=0.2, random_state=36,
                                                stratify=labels)  # 80% 训练，20%测试结果
    # --------------------------------------数据集读取我的版本end------------------------------------------

    # ------------------------设置基础模型---------------------------------------
    # model1 = sklearn.svm.SVC(probability=False)  # SVM
    # model2 = KNeighborsClassifier()  # KNN
    # model3 = GradientBoostingClassifier(random_state=37)  # GBDT
    # model4 = DecisionTreeClassifier(random_state=666)  # DT
    # model5 = LGBMClassifier(learning_rate=0.1,n_estimators=100)  # GBM
    # model6 = MLPClassifier(hidden_layer_sizes=(100,),random_state=420,max_iter=1000)  # MLP
    # model7 = LogisticRegression(max_iter=1000)  # LogisticRegression
    # model8 = RandomForestClassifier()  # RF随机森林

    # 下面的是调参后的
    para_svm = {'C': 0.0224433889890122, 'coef0': 2.248410437719281, 'gamma': 12.923005804808009, 'kernel': 'linear'}
    para_knn = {'n_neighbors': 26}
    para_GBM = {'boosting_type': 'dart', 'learning_rate': 0.1218080101678672, 'num_leaves': 20}
    para_MLP = {'hidden_layer_sizes': 50}
    para_LR = {'C': 1}
    para_RF = {'criterion': "gini", 'max_depth': 92, 'max_features': 3, 'min_samples_split': 3, 'n_estimators': 77}

    model1 = sklearn.svm.SVC(**para_svm)
    model2 = KNeighborsClassifier(**para_knn)
    model3 = GradientBoostingClassifier(random_state=37)  # GBDT 没有参数
    model4 = DecisionTreeClassifier(random_state=666)  # DT 没有参数
    model5 = LGBMClassifier(**para_GBM)  # GBM
    model6 = MLPClassifier(hidden_layer_sizes=50, random_state=420, max_iter=1000)  # MLP
    model7 = LogisticRegression(**para_LR, max_iter=1000)  # LogisticRegression
    model8 = RandomForestClassifier()  # RF随机森林（添加了参数偶尔比没添加参数更低）

    # ------------------------设置stacking的参数---------------------------------------
    reg = StackingClassifier(
        estimators=[
            ('SVM', model1),
            ('KNN', model2),
            ('GBDT', model3),
            ('DT', model4),
            ('GBM', model5),
            ('MLP', model6),
            ('LogisticRegression', model7),
            ('RF', model8),
        ],
        final_estimator=LogisticRegression(C=1, n_jobs=-1))

    # voting_clf = VotingClassifier(
    #     estimators=[
    #         # ('SVM', model1),
    #                 ('KNN', model2),
    #                 ('GBDT', model3),
    #                 ('DT', model4),
    #                 # ('GBM', model5),
    #                 ('MLP', model6),
    #                 ('LogisticRegression', model7),
    #                 ('RF', model8),
    #                 ], voting='hard')

    # ------------------------训练并输出结果---------------------------------------
    # model1.fit(X_train, y_train)
    # acc1 = model1.score(X_test, y_test)
    # print("SVM", acc1)
    #
    # model2.fit(X_train, y_train)
    # acc2 = model2.score(X_test, y_test)
    # print("KNN", acc2)
    #
    # model3.fit(X_train, y_train)
    # acc3 = model3.score(X_test, y_test)
    # print("GBDT", acc3)
    #
    # model4.fit(X_train, y_train)
    # acc4 = model4.score(X_test, y_test)
    # print("DTREE", acc4)
    #
    # model5.fit(X_train, y_train)
    # acc5 = model5.score(X_test, y_test)
    # print("gbm", acc5)
    #
    # model6.fit(X_train, y_train)
    # acc6 = model6.score(X_test, y_test)
    # print("MLP", acc6)
    #
    # model7.fit(X_train, y_train)
    # acc7 = model7.score(X_test, y_test)
    # print("LogisticRegression", acc7)
    #
    # model8.fit(X_train, y_train)
    # acc8 = model8.score(X_test, y_test)
    # print("RF", acc8)

    reg.fit(X_train, y_train)
    acc_final = reg.score(X_test, y_test)
    test_predict = reg.predict(X_test)
    print("-----------------轮次=", i)
    print("集成学习的结果是：", acc_final)

    confusion_matrix_result = metrics.confusion_matrix(test_predict, y_test, labels=[0, 1, 2])
    print(confusion_matrix_result)
    print(classification_report(test_predict, y_test, digits=4))  # 保留四位小数点

# ------------------------5折交叉验证---------------------------------------
# for clf, label in zip([model1, model2, model3, model4, model5,model6,model7,model8,reg],
#                       ['SVM', 'KNN', 'GBDT', 'DTREE','GBM','MLP','LR','RF','集成学习']):
#     scores = cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5)  # cross_val_score训练模型打分函数,
#     # 参数scoring：accuracy cv：5 将数据集分为大小相同的5份，四份训练，一份测试
#     # scores.mean()分数、scores.std()误差
#     print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), label))


# estimators = [ ('rf', clf2), ('gnb', clf3), ('svm', clf4)]
# final_estimator = GradientBoostingRegressor(
#     n_estimators=25, subsample=0.5, min_samples_leaf=25, max_features=1,
#     random_state=42)
# reg = StackingRegressor(
#     estimators=estimators,
#     final_estimator=final_estimator)
# reg.fit(X_train, y_train)
# train_predict = reg.predict(X_train)
# test_predict = reg.predict(X_test)
#
# ## 查看混淆矩阵 (预测值和真实值的各类情况统计矩阵)
# confusion_matrix_result = metrics.confusion_matrix(test_predict, y_test, labels=[0, 1, 2])
#
# print('The confusion matrix result:\n', confusion_matrix_result)
#
# print(classification_report(y_test, test_predict))


# # 决策树分类器
# # max_depth决策树最大深度
# # criterion = gini/entropy 可以用来选择用基尼指数或者熵来做损失函数。
# base_model = DecisionTreeClassifier(max_depth=1, criterion='gini', random_state=1).fit(X_train, y_train)
# y_pred = base_model.predict(X_test)  # 预测模型结果
# print(f"决策树的准确率：{accuracy_score(y_test, y_pred):.3f}")
#
# from sklearn.ensemble import AdaBoostClassifier
#
# model = AdaBoostClassifier(base_estimator=base_model,
#                            n_estimators=50,
#                            learning_rate=0.5,
#                            algorithm='SAMME.R',
#                            random_state=1)
# model.fit(X_train, y_train)  # 报错：ValueError: y should be a 1d array, got an array of shape (1252, 3) instead.
# y_pred = model.predict(X_test)
# print(f"AdaBoost的准确率：{accuracy_score(y_test, y_pred):.3f}")
#
# # 测试估计器个数的影响
# x = list(range(2, 20, 2))  # 从2开始，间隔是2，到102，一般是n_estimators到了90多会取得最高的ACC
# y = []
#
# for i in x:
#     print("第", i, "轮：", "测试n_estimators...", "准确率为：", end="")
#     model = AdaBoostClassifier(base_estimator=base_model,
#                                n_estimators=i,  # 代表随机森林里面树的个数
#                                learning_rate=0.5,
#                                algorithm='SAMME.R',
#                                random_state=1)
#     model.fit(X_train, y_train)
#     model_test_sc = accuracy_score(y_test, model.predict(X_test))
#     y.append(model_test_sc)
#     print(model_test_sc)
#
# plt.style.use('ggplot')
# plt.title("Effect of n_estimators", pad=20)
# plt.xlabel("Number of base estimators")
# plt.ylabel("Test accuracy of AdaBoost")
# plt.plot(x, y)
# plt.show()
#
# # 使用GridSearchCV自动调参
# # GridSearch和CV，即网格搜索和交叉验证。
# # 网格搜索，搜索的是参数，即在指定的参数范围内，按步长依次调整参数，利用调整的参数训练学习器，从所有的参数中找到在验证集上精度最高的参数，这其实是一个训练和比较的过程。
# hyperparameter_space = {'n_estimators': list(range(2, 102, 2)),
#                         'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
#
# gs = GridSearchCV(AdaBoostClassifier(base_estimator=base_model,
#                                      algorithm='SAMME.R',
#                                      random_state=1),
#                   param_grid=hyperparameter_space,
#                   scoring="accuracy", n_jobs=-1, cv=5, return_train_score=True, verbose=20)
#
# gs.fit(X_train, y_train)
# print("---------调参结束，最终结果如下---------")
# print("最优超参数:", gs.best_params_)
# print("best_score_:", gs.best_score_)
# print("best_index_:", gs.best_index_)
# print("best_estimator_:", gs.best_estimator_)
