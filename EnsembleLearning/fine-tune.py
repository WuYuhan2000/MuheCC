# # -*- coding: utf-8 -*-
# """
# @Time ： 2023/2/16 21:26
# @Auth ： 兰宏富
# @File ：adaboost_decisiontree_gridsearch.py
# @IDE ：PyCharm
# """
# import sklearn.ensemble
# from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
# from hyperopt.early_stop import no_progress_loss
# from lightgbm import LGBMClassifier
# from matplotlib import pyplot as plt
# from sklearn import metrics
# from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, GradientBoostingRegressor, \
#     StackingRegressor, StackingClassifier, RandomForestClassifier, AdaBoostClassifier  # AdaBoost分类器
# from sklearn.linear_model import LogisticRegression, RidgeCV, SGDClassifier, Ridge, RidgeClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier  # 决策树分类器
# from bayes_opt import BayesianOptimization  # 贝叶斯优化
# from sklearn.model_selection import cross_val_score
#
# # --------------------------------------数据集读取原始版本------------------------------------------
# # wine = load_wine()
# # print(f"所有特征：{wine.feature_names}")
# # X = pd.DataFrame(wine.data, columns=wine.feature_names)
# # y = pd.Series(wine.target)
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
# # --------------------------------------数据集读取原始版本------------------------------------------
#
#
# # --------------------------------------数据集读取我的版本start------------------------------------------
# # model_name = 'huggingface/CodeBERTa-small-v1'  # 准确率最高
# model_name = 'distilbert-base-uncased'  # 准确率第二
# # model_name = 'bert-base-uncased'  # 准确率第三
# csn_path = '../dataset/Commit_dataset_final.xlsx'
#
# import logging
#
# logging.basicConfig(level=logging.INFO)
# import torch
# from transformers import AutoTokenizer, \
#     AutoModel
#
# # Load pre-trained model tokenizer (vocabulary)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# import pandas as pd
#
# df = pd.read_excel(csn_path,
#                    header=0,  # 不读取第一行的表头
#                    index_col=False,  # 设置为False就会读取第一列，设置为0不会读取第一列
#                    )
# # Tokenize input
# tokenized_text = df['comment'].apply(
#     (lambda x: tokenizer.encode(x, max_length=80, add_special_tokens=True, truncation=True)))
#
# import numpy as np
#
# max_len = 0
# for i in tokenized_text.values:
#     if len(i) > max_len:
#         max_len = len(i)
#
# padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized_text.values])
#
# np.array(padded).shape
# attention_mask = np.where(padded != 0, 1, 0)
# attention_mask.shape
#
# model = AutoModel.from_pretrained(model_name)
# input_ids = torch.tensor(padded)
# attention_mask = torch.tensor(attention_mask)
#
# with torch.no_grad():
#     last_hidden_states = model(input_ids, attention_mask=attention_mask)
#
# bert_features = last_hidden_states[0][:, 0, :].numpy()  # BERT的输出：768维度的语义向量
# print(bert_features.shape)
# bert_features.shape
#
# # labels = df['3_labels']
# # print(labels)
# change = {'p': 0, 'a': 1, 'c': 2}  # 替换的值
# labels = df['3_labels'].map(change).values  # 标签值
# cc = df.drop(columns=['3_labels', 'comment', 'project', 'commit_id', 'comment_diff'], axis=1).replace(np.nan,
#                                                                                                       0)  # 将空的格子设置为0
# print(cc)
#
# from sklearn.preprocessing import StandardScaler, normalize, scale
#
# sc = StandardScaler()
# cc = sc.fit_transform(cc)
# print(cc)
#
# all_input_features = np.concatenate((bert_features, cc), axis=1)  # 拼接语义和特征
# all_input_features[0].shape
#
# # encoder = LabelBinarizer() # 转换成独热编码
# # labels = encoder.fit_transform(labels)
# # print(labels)
#
# from sklearn.model_selection import train_test_split, cross_val_score
#
# # stratified train_test_split
# X_train, X_test, y_train, y_test = train_test_split(all_input_features, labels, test_size=0.2, random_state=42,
#                                                     stratify=labels)  # 80% 训练，20%测试结果
#
# # --------------------------------------数据集读取我的版本end------------------------------------------
#
#
# # ------------------------设置基础模型---------------------------------------
# # model1 = sklearn.svm.SVC(probability=False)  # SVM
# # model2 = KNeighborsClassifier()  # KNN
# # model3 = GradientBoostingClassifier(random_state=37)  # GBDT
# # model4 = DecisionTreeClassifier(random_state=666)  # DT
# # model5 = LGBMClassifier(learning_rate=0.1, n_estimators=100)  # GBM
# # model6 = MLPClassifier(hidden_layer_sizes=(100,), random_state=420, max_iter=1000)  # MLP
# # model7 = LogisticRegression(max_iter=1000)  # LogisticRegression
# # model8 = RandomForestClassifier()  # RF随机森林
#
# # para_svm = {'C': 0.0224433889890122, 'coef0': 2.248410437719281, 'gamma': 12.923005804808009, 'kernel': 'linear'}
# # para_knn = {'n_neighbors': 26}
# # para_GBM = {'boosting_type': 'dart', 'learning_rate': 0.1218080101678672, 'num_leaves': 20}
# # para_MLP={'hidden_layer_sizes': 50}
# # para_LR={'C': 1}
# # para_RF = {'criterion': "gini", 'max_depth': 92, 'max_features': 3, 'min_samples_split': 3, 'n_estimators': 77}
# # x = sklearn.svm.SVC(**para_svm)
# # x = KNeighborsClassifier(**para_svm)
# # x = LGBMClassifier(**para_GBM)  # GBM
# # x = MLPClassifier(hidden_layer_sizes=50, random_state=420, max_iter=1000)  # MLP
# # x = LogisticRegression(C=1,max_iter=1000)  # LogisticRegression
# # x = RandomForestClassifier(**para_RF)  # RF随机森林（添加了参数比没添加参数更低）
# # scores = cross_val_score(x, all_input_features, labels, scoring='accuracy', cv=5)
# # print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
# # x.fit(X_train, y_train)
# # test_predict = x.predict(X_test)
# # confusion_matrix_result = metrics.confusion_matrix(test_predict, y_test, labels=[0, 1, 2])
# # print(confusion_matrix_result)
# # print(classification_report(test_predict, y_test, digits=4))  # 保留四位小数点
#
# #------------------------SVM调参---------------------------------------
# model = sklearn.svm.SVC(probability=False)
# scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5)
# print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
#
# def hyperopt_train_test(params):
#     X_ = all_input_features[:]
#
#     # if 'normalize' in params:
#     #     if params['normalize'] == 1:
#     #         X_ = normalize(X_)
#     #         del params['normalize']
#
#     # if 'scale' in params:
#     #     if params['scale'] == 1:
#     #         X_ = scale(X_)
#     #         del params['scale']
#
#     clf = SVC(**params)
#     return cross_val_score(clf, X_, labels,scoring='accuracy', cv=5).mean()
#
# space4svm = {
#     'C': hp.quniform('C', 0.1, 2,0.05),
#     # 'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
#     # 'gamma': hp.uniform('gamma', 0, 20),
#     # 'coef0': hp.uniform('coef0', 0, 10),
#     # 'scale': hp.choice('scale', [0, 1]),
#     # 'normalize': hp.choice('normalize', [0, 1])
# }
#
# def f(params):
#     acc = hyperopt_train_test(params)
#     return {'loss': -acc, 'status': STATUS_OK}
#
# trials = Trials()
# best = fmin(f, space4svm, algo=tpe.suggest, max_evals=100, trials=trials)
# print('best:')
# print(best)
#
# # ------------------------KNN调参---------------------------------------
# model = KNeighborsClassifier()  # KNN
# scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5)
# print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
#
# def hyperopt_train_test(params):
#     clf = KNeighborsClassifier(**params)
#     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5).mean()
#
#
# space4knn = {
#     'n_neighbors': hp.choice('n_neighbors', range(2, 31)),
#     'leaf_size': hp.quniform('leaf_size', 10, 50, 10)
# }
#
#
# def f(params):
#     acc = hyperopt_train_test(params)
#     return {'loss': -acc, 'status': STATUS_OK}
#
#
# trials = Trials()
# best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
# print('best:')
# print(best)
#
# # ------------------------SGD调参---------------------------------------
# model = SGDClassifier()  # KNN
# scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30)
# print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
#
# def hyperopt_train_test(params):
#     clf = SGDClassifier(**params)
#     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30).mean()
#
#
# space4knn = {
#     'l1_ratio': hp.choice('l1_ratio', [0.1,0.15,0.2,0.25,0.3,0.35,0.4]),
#     'alpha': hp.choice('alpha', [0.0001,0.001,0.01,0.1])
# }
#
#
# def f(params):
#     acc = hyperopt_train_test(params)
#     return {'loss': -acc, 'status': STATUS_OK}# -*- coding: utf-8 -*-
# """
# @Time ： 2023/2/16 21:26
# @Auth ： 兰宏富
# @File ：adaboost_decisiontree_gridsearch.py
# @IDE ：PyCharm
# """
# import sklearn.ensemble
# from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
# from hyperopt.early_stop import no_progress_loss
# from lightgbm import LGBMClassifier
# from matplotlib import pyplot as plt
# from sklearn import metrics
# from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, GradientBoostingRegressor, \
#     StackingRegressor, StackingClassifier, RandomForestClassifier, AdaBoostClassifier  # AdaBoost分类器
# from sklearn.linear_model import LogisticRegression, RidgeCV, SGDClassifier, Ridge, RidgeClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier  # 决策树分类器
# from bayes_opt import BayesianOptimization  # 贝叶斯优化
# from sklearn.model_selection import cross_val_score
#
# # --------------------------------------数据集读取原始版本------------------------------------------
# # wine = load_wine()
# # print(f"所有特征：{wine.feature_names}")
# # X = pd.DataFrame(wine.data, columns=wine.feature_names)
# # y = pd.Series(wine.target)
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
# # --------------------------------------数据集读取原始版本------------------------------------------
#
#
# # --------------------------------------数据集读取我的版本start------------------------------------------
# # model_name = 'huggingface/CodeBERTa-small-v1'  # 准确率最高
# model_name = 'distilbert-base-uncased'  # 准确率第二
# # model_name = 'bert-base-uncased'  # 准确率第三
# csn_path = '../dataset/Commit_dataset_final.xlsx'
#
# import logging
#
# logging.basicConfig(level=logging.INFO)
# import torch
# from transformers import AutoTokenizer, \
#     AutoModel
#
# # Load pre-trained model tokenizer (vocabulary)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# import pandas as pd
#
# df = pd.read_excel(csn_path,
#                    header=0,  # 不读取第一行的表头
#                    index_col=False,  # 设置为False就会读取第一列，设置为0不会读取第一列
#                    )
# # Tokenize input
# tokenized_text = df['comment'].apply(
#     (lambda x: tokenizer.encode(x, max_length=80, add_special_tokens=True, truncation=True)))
#
# import numpy as np
#
# max_len = 0
# for i in tokenized_text.values:
#     if len(i) > max_len:
#         max_len = len(i)
#
# padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized_text.values])
#
# np.array(padded).shape
# attention_mask = np.where(padded != 0, 1, 0)
# attention_mask.shape
#
# model = AutoModel.from_pretrained(model_name)
# input_ids = torch.tensor(padded)
# attention_mask = torch.tensor(attention_mask)
#
# with torch.no_grad():
#     last_hidden_states = model(input_ids, attention_mask=attention_mask)
#
# bert_features = last_hidden_states[0][:, 0, :].numpy()  # BERT的输出：768维度的语义向量
# print(bert_features.shape)
# bert_features.shape
#
# # labels = df['3_labels']
# # print(labels)
# change = {'p': 0, 'a': 1, 'c': 2}  # 替换的值
# labels = df['3_labels'].map(change).values  # 标签值
# cc = df.drop(columns=['3_labels', 'comment', 'project', 'commit_id', 'comment_diff'], axis=1).replace(np.nan,
#                                                                                                       0)  # 将空的格子设置为0
# print(cc)
#
# from sklearn.preprocessing import StandardScaler, normalize, scale
#
# sc = StandardScaler()
# cc = sc.fit_transform(cc)
# print(cc)
#
# all_input_features = np.concatenate((bert_features, cc), axis=1)  # 拼接语义和特征
# all_input_features[0].shape
#
# # encoder = LabelBinarizer() # 转换成独热编码
# # labels = encoder.fit_transform(labels)
# # print(labels)
#
# from sklearn.model_selection import train_test_split, cross_val_score
#
# # stratified train_test_split
# X_train, X_test, y_train, y_test = train_test_split(all_input_features, labels, test_size=0.2, random_state=42,
#                                                     stratify=labels)  # 80% 训练，20%测试结果
#
# # --------------------------------------数据集读取我的版本end------------------------------------------
#
#
# # ------------------------设置基础模型---------------------------------------
# # model1 = sklearn.svm.SVC(probability=False)  # SVM
# # model2 = KNeighborsClassifier()  # KNN
# # model3 = GradientBoostingClassifier(random_state=37)  # GBDT
# # model4 = DecisionTreeClassifier(random_state=666)  # DT
# # model5 = LGBMClassifier(learning_rate=0.1, n_estimators=100)  # GBM
# # model6 = MLPClassifier(hidden_layer_sizes=(100,), random_state=420, max_iter=1000)  # MLP
# # model7 = LogisticRegression(max_iter=1000)  # LogisticRegression
# # model8 = RandomForestClassifier()  # RF随机森林
#
# # para_svm = {'C': 0.0224433889890122, 'coef0': 2.248410437719281, 'gamma': 12.923005804808009, 'kernel': 'linear'}
# # para_knn = {'n_neighbors': 26}
# # para_GBM = {'boosting_type': 'dart', 'learning_rate': 0.1218080101678672, 'num_leaves': 20}
# # para_MLP={'hidden_layer_sizes': 50}
# # para_LR={'C': 1}
# # para_RF = {'criterion': "gini", 'max_depth': 92, 'max_features': 3, 'min_samples_split': 3, 'n_estimators': 77}
# # x = sklearn.svm.SVC(**para_svm)
# # x = KNeighborsClassifier(**para_svm)
# # x = LGBMClassifier(**para_GBM)  # GBM
# # x = MLPClassifier(hidden_layer_sizes=50, random_state=420, max_iter=1000)  # MLP
# # x = LogisticRegression(C=1,max_iter=1000)  # LogisticRegression
# # x = RandomForestClassifier(**para_RF)  # RF随机森林（添加了参数比没添加参数更低）
# # scores = cross_val_score(x, all_input_features, labels, scoring='accuracy', cv=5)
# # print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
# # x.fit(X_train, y_train)
# # test_predict = x.predict(X_test)
# # confusion_matrix_result = metrics.confusion_matrix(test_predict, y_test, labels=[0, 1, 2])
# # print(confusion_matrix_result)
# # print(classification_report(test_predict, y_test, digits=4))  # 保留四位小数点
#
# # ------------------------SVM调参---------------------------------------
# # model = sklearn.svm.SVC(probability=False)
# # scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5)
# # print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
# #
# # def hyperopt_train_test(params):
# #     X_ = all_input_features[:]
# #
# #     # if 'normalize' in params:
# #     #     if params['normalize'] == 1:
# #     #         X_ = normalize(X_)
# #     #         del params['normalize']
# #
# #     # if 'scale' in params:
# #     #     if params['scale'] == 1:
# #     #         X_ = scale(X_)
# #     #         del params['scale']
# #
# #     clf = SVC(**params)
# #     return cross_val_score(clf, X_, labels,scoring='accuracy', cv=5).mean()
# #
# # space4svm = {
# #     'C': hp.quniform('C', 0.1, 2,0.05),
# #     # 'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
# #     # 'gamma': hp.uniform('gamma', 0, 20),
# #     # 'coef0': hp.uniform('coef0', 0, 10),
# #     # 'scale': hp.choice('scale', [0, 1]),
# #     # 'normalize': hp.choice('normalize', [0, 1])
# # }
# #
# # def f(params):
# #     acc = hyperopt_train_test(params)
# #     return {'loss': -acc, 'status': STATUS_OK}
# #
# # trials = Trials()
# # best = fmin(f, space4svm, algo=tpe.suggest, max_evals=100, trials=trials)
# # print('best:')
# # print(best)
#
# # ------------------------KNN调参---------------------------------------
# model = KNeighborsClassifier()  # KNN
# scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5)
# print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
#
# def hyperopt_train_test(params):
#     clf = KNeighborsClassifier(**params)
#     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5).mean()
#
#
# space4knn = {
#     'n_neighbors': hp.choice('n_neighbors', range(2, 31)),
#     'leaf_size': hp.quniform('leaf_size', 10, 50, 10)
# }
#
#
# def f(params):
#     acc = hyperopt_train_test(params)
#     return {'loss': -acc, 'status': STATUS_OK}
#
#
# trials = Trials()
# best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
# print('best:')
# print(best)
#
# # ------------------------SGD调参---------------------------------------
# # model = SGDClassifier()  # KNN
# # scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30)
# # print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
# #
# # def hyperopt_train_test(params):
# #     clf = SGDClassifier(**params)
# #     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30).mean()
# #
# #
# # space4knn = {
# #     'l1_ratio': hp.choice('l1_ratio', [0.1,0.15,0.2,0.25,0.3,0.35,0.4]),
# #     'alpha': hp.choice('alpha', [0.0001,0.001,0.01,0.1])
# # }
# #
# #
# # def f(params):
# #     acc = hyperopt_train_test(params)
# #     return {'loss': -acc, 'status': STATUS_OK}
# #
# #
# # trials = Trials()
# # best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
# # print('best:')
# # print(best)
#
# # -----------------岭回归分类器调参---------------------------------------
# # model = RidgeClassifier()  # 注意，如果导入的是Ridge那么将报错
# # scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5)
# # print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
# #
# #
# # def hyperopt_train_test(params):
# #     clf = RidgeClassifier(**params)
# #     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5).mean()
# #
# #
# # space4knn = {
# #     'alpha': hp.quniform('alpha',0,2,0.05)
# # }
# #
# #
# # def f(params):
# #     acc = hyperopt_train_test(params)
# #     return {'loss': -acc, 'status': STATUS_OK}
# #
# #
# # trials = Trials()
# # best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
# # print('best:')
# # print(best)
#
# # ------------------------NB调参---------------------------------------
# # model = GaussianNB()  # GaussianNB
# # scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30)
# # print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
# #
# # def hyperopt_train_test(params):
# #     clf = GaussianNB(**params)
# #     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30).mean()
# #
# #
# # space4knn = {
# #     'var_smoothing': hp.choice('var_smoothing', [1,0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001,0.00000001,0.000000001])
# # }
# #
# #
# # def f(params):
# #     acc = hyperopt_train_test(params)
# #     return {'loss': -acc, 'status': STATUS_OK}
# #
# #
# # trials = Trials()
# # best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
# # print('best:')
# # print(best)
#
# # ------------------------GBDT调参---------------------------------------
#
# # model = GradientBoostingClassifier(random_state=37)  # KNN
# # scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1)
# # print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
# #
# # def hyperopt_train_test(params):
# #     X_ = all_input_features[:]
# #     # if 'normalize' in params:
# #     #     if params['normalize'] == 1:
# #     #         X_ = normalize(X_)
# #     #         del params['normalize']
# #     #
# #     # if 'scale' in params:
# #     #     if params['scale'] == 1:
# #     #         X_ = scale(X_)
# #     #         del params['scale']
# #     clf = GradientBoostingClassifier(n_estimators=int(params["n_estimators"])
# #                                      , learning_rate=params["learning_rate"]
# #                                      # , criterion=params["criterion"]
# #                                      # , max_depth=int(params["max_depth"])
# #                                      # , max_features=int(params["max_features"])
# #                                      # , subsample=params["subsample"]
# #                                      # , min_impurity_decrease=params["min_impurity_decrease"]
# #                                      , random_state=1412
# #                                      , verbose=False)
# #     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1).mean()
# #
# #
# # space4dt = {'n_estimators': hp.quniform("n_estimators", 10, 200, 10)
# #             , "learning_rate": hp.quniform("learning_rate", 0.05, 1, 0.05)
# #             # , "criterion": hp.choice("criterion", ["friedman_mse", "squared_error", "squared_error", "absolute_error"])
# #             # , "max_depth": hp.quniform("max_depth", 10, 100, 1)
# #             # , "subsample": hp.quniform("subsample", 0.5, 1, 0.05)
# #             # , "max_features": hp.quniform("max_features", 0, 30, 1)
# #             # , "min_impurity_decrease": hp.quniform("min_impurity_decrease", 0, 5, 0.5)
# #             }
# #
# #
# # def f(params):
# #     acc = hyperopt_train_test(params)
# #     return {'loss': -acc, 'status': STATUS_OK}
# #
# #
# # def param_hyperopt(max_evals=100):
# #     # 保存迭代过程
# #     trials = Trials()
# #
# #     # 设置提前停止
# #     # early_stop_fn = no_progress_loss(100)
# #
# #     # 定义代理模型
# #     params_best = fmin(f
# #                        , space=space4dt
# #                        , algo=tpe.suggest
# #                        , max_evals=max_evals
# #                        , verbose=True
# #                        , trials=trials
# #                        # , early_stop_fn=early_stop_fn
# #                        )
# #
# #     # 打印最优参数，fmin会自动打印最佳分数
# #     print("\n", "\n", "best params: ", params_best,
# #           "\n")
# #     return params_best, trials
# #
# #
# # best, trials = param_hyperopt(60)
# # print('best:')
# # print(best)
#
# # -----------DT调参---------------------------------------
# # para = {'max_depth': 899, 'min_samples_leaf': 0.0275, 'min_samples_split': 0.0236}
# # model = DecisionTreeClassifier(**para)  # KNN
# # scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1)
# # print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
# #
# # def hyperopt_train_test(params):
# #     clf = DecisionTreeClassifier(**params)
# #     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1).mean()
# #
# #
# # space4dt = {
# #     'max_depth': hp.choice('max_depth', range(1, 1000)),
# #     'min_samples_leaf':hp.uniform('min_samples_leaf',0,0.5),
# #     'min_samples_split':hp.uniform('min_samples_split',0,1.0)
# # }
# #
# #
# # def f(params):
# #     acc = hyperopt_train_test(params)
# #     return {'loss': -acc, 'status': STATUS_OK}
# #
# #
# # def param_hyperopt(max_evals=100):
# #     # 保存迭代过程
# #     trials = Trials()
# #
# #     # 设置提前停止
# #     # early_stop_fn = no_progress_loss(300)
# #
# #     # 定义代理模型
# #     params_best = fmin(f
# #                        , space=space4dt
# #                        , algo=tpe.suggest
# #                        , max_evals=max_evals
# #                        , verbose=True
# #                        , trials=trials
# #                        # , early_stop_fn=early_stop_fn
# #                        )
# #
# #     # 打印最优参数，fmin会自动打印最佳分数
# #     print("\n", "\n", "best params: ", params_best,
# #           "\n")
# #     return params_best, trials
# #
# #
# # best, trials = param_hyperopt(800)
# # print('best:')
# # print(best)
# # -----------LGBM调参---------------------------------------
# # para_lightGBM = {'learning_rate': 0.05, 'n_estimators': 7,'num_leaves': 20}
# # print("LGBM调参：")
# # model = LGBMClassifier(**para_lightGBM)  # adaboost
# # scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5, n_jobs=-1, verbose=30)
# # print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
# #
# #
# # def hyperopt_train_test(params):
# #     clf = LGBMClassifier(**params)
# #     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5, n_jobs=-1, verbose=30).mean()
# #
# #
# # space4dt = {
# #     'num_leaves': hp.choice('num_leaves', range(5, 200, 5)),  # ??
# #     "learning_rate": hp.quniform("learning_rate", 0.05, 1, 0.05),  # 学习率
# #     'n_estimators': hp.choice('n_estimators', range(5, 250, 5)),  # 学习器的数量
# #
# # }
# #
# #
# # def f(params):
# #     acc = hyperopt_train_test(params)
# #     return {'loss': -acc, 'status': STATUS_OK}
# #
# #
# # def param_hyperopt(max_evals=100):
# #     # 保存迭代过程
# #     trials = Trials()
# #
# #     # 设置提前停止
# #     # early_stop_fn = no_progress_loss(300)
# #
# #     # 定义代理模型
# #     params_best = fmin(f
# #                        , space=space4dt
# #                        , algo=tpe.suggest
# #                        , max_evals=max_evals
# #                        , verbose=True
# #                        , trials=trials
# #                        # , early_stop_fn=early_stop_fn
# #                        )
# #
# #     # 打印最优参数，fmin会自动打印最佳分数
# #     print("\n", "\n", "best params: ", params_best,
# #           "\n")
# #     return params_best, trials
# #
# #
# # best, trials = param_hyperopt(60)
# # print('best:')
# # print(best)
#
# # -----------MLP调参---------------------------------------
#
# # def hyperopt_train_test(params):
# #     clf = MLPClassifier(**params,max_iter=500)
# #     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5).mean()
# #
# #
# # space4dt = {
# #           "hidden_layer_sizes":hp.choice("hidden_layer_sizes",range(1,1000,10))
# #           # "learning_rate":hp.loguniform("learning_rate",np.log(0.001),np.log(0.5))
# #           }
# #
# #
# # def f(params):
# #     acc = hyperopt_train_test(params)
# #     return {'loss': -acc, 'status': STATUS_OK}
# #
# #
# # def param_hyperopt(max_evals=100):
# #     # 保存迭代过程
# #     trials = Trials()
# #
# #     # 设置提前停止
# #     # early_stop_fn = no_progress_loss(300)
# #
# #     # 定义代理模型
# #     params_best = fmin(f
# #                        , space=space4dt
# #                        , algo=tpe.suggest
# #                        , max_evals=max_evals
# #                        , verbose=True
# #                        , trials=trials
# #                        # , early_stop_fn=early_stop_fn
# #                        )
# #
# #     # 打印最优参数，fmin会自动打印最佳分数
# #     print("\n", "\n", "best params: ", params_best,
# #           "\n")
# #     return params_best, trials
# #
# #
# # best, trials = param_hyperopt(50)
# # print('best:')
# # print(best)
#
#
# # ------lr调参---------------------------------------
#
# # def hyperopt_train_test(params):
# #     clf = LogisticRegression(**params)
# #     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5).mean()
# #
# #
# # space4dt = {
# #           "C":hp.choice("C",[0.001,0.01,0.1,1,5,10,100])
# #           # "learning_rate":hp.loguniform("learning_rate",np.log(0.001),np.log(0.5))
# #           }
# #
# #
# # def f(params):
# #     acc = hyperopt_train_test(params)
# #     return {'loss': -acc, 'status': STATUS_OK}
# #
# #
# # def param_hyperopt(max_evals=100):
# #     # 保存迭代过程
# #     trials = Trials()
# #
# #     # 设置提前停止
# #     # early_stop_fn = no_progress_loss(300)
# #
# #     # 定义代理模型
# #     params_best = fmin(f
# #                        , space=space4dt
# #                        , algo=tpe.suggest
# #                        , max_evals=max_evals
# #                        , verbose=True
# #                        , trials=trials
# #                        # , early_stop_fn=early_stop_fn
# #                        )
# #
# #     # 打印最优参数，fmin会自动打印最佳分数
# #     print("\n", "\n", "best params: ", params_best,
# #           "\n")
# #     return params_best, trials
# #
# #
# # best, trials = param_hyperopt(50)
# # print('best:')
# # print(best)
#
# # ---------------------------RF随机森林调参---------------------------------------
# # model = RandomForestClassifier()  # KNN
# # scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30)
# # print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
# #
# # def hyperopt_train_test(params):
# #     clf = RandomForestClassifier(**params)
# #     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30).mean()
# #
# #
# # space4dt = {
# #     'max_depth': hp.choice('max_depth', range(5,200,5)),
# #     'n_estimators': hp.choice('n_estimators', range(5,200,5)),
# # }
# #
# #
# # def f(params):
# #     acc = hyperopt_train_test(params)
# #     return {'loss': -acc, 'status': STATUS_OK}
# #
# #
# # def param_hyperopt(max_evals=100):
# #     # 保存迭代过程
# #     trials = Trials()
# #
# #     # 设置提前停止
# #     # early_stop_fn = no_progress_loss(300)
# #
# #     # 定义代理模型
# #     params_best = fmin(f
# #                        , space=space4dt
# #                        , algo=tpe.suggest
# #                        , max_evals=max_evals
# #                        , verbose=True
# #                        , trials=trials
# #                        # , early_stop_fn=early_stop_fn
# #                        )
# #
# #     # 打印最优参数，fmin会自动打印最佳分数
# #     print("\n", "\n", "best params: ", params_best,
# #           "\n")
# #     return params_best, trials
# #
# #
# # best, trials = param_hyperopt(200)
# # print('best:')
# # print(best)
#
# # ---------------------------adaboost调参---------------------------------------
# # print("adaboost调参：")
# # model = AdaBoostClassifier()  # adaboost
# # scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30)
# # print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
# #
# # def hyperopt_train_test(params):
# #     clf = AdaBoostClassifier(**params)
# #     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30).mean()
# #
# #
# # space4dt = {
# #     "learning_rate": hp.quniform("learning_rate", 0.05, 1, 0.05),
# #     'n_estimators': hp.choice('n_estimators', range(5,250,5))
# # }
# #
# #
# # def f(params):
# #     acc = hyperopt_train_test(params)
# #     return {'loss': -acc, 'status': STATUS_OK}
# #
# #
# # def param_hyperopt(max_evals=100):
# #     # 保存迭代过程
# #     trials = Trials()
# #
# #     # 设置提前停止
# #     # early_stop_fn = no_progress_loss(300)
# #
# #     # 定义代理模型
# #     params_best = fmin(f
# #                        , space=space4dt
# #                        , algo=tpe.suggest
# #                        , max_evals=max_evals
# #                        , verbose=True
# #                        , trials=trials
# #                        # , early_stop_fn=early_stop_fn
# #                        )
# #
# #     # 打印最优参数，fmin会自动打印最佳分数
# #     print("\n", "\n", "best params: ", params_best,
# #           "\n")
# #     return params_best, trials
# #
# #
# # best, trials = param_hyperopt(60)
# # print('best:')
# # print(best)
#
#
# trials = Trials()
# best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
# print('best:')
# print(best)
#
# -----------------岭回归分类器调参---------------------------------------
# model = RidgeClassifier()  # 注意，如果导入的是Ridge那么将报错
# scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5)
# print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
#
#
# def hyperopt_train_test(params):
#     clf = RidgeClassifier(**params)
#     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5).mean()
#
#
# space4knn = {
#     'alpha': hp.quniform('alpha',0,2,0.05)
# }
#
#
# def f(params):
#     acc = hyperopt_train_test(params)
#     return {'loss': -acc, 'status': STATUS_OK}
#
#
# trials = Trials()
# best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
# print('best:')
# print(best)
#
# ------------------------NB调参---------------------------------------
# model = GaussianNB()  # GaussianNB
# scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30)
# print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
#
# def hyperopt_train_test(params):
#     clf = GaussianNB(**params)
#     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30).mean()
#
#
# space4knn = {
#     'var_smoothing': hp.choice('var_smoothing', [1,0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001,0.00000001,0.000000001])
# }
#
#
# def f(params):
#     acc = hyperopt_train_test(params)
#     return {'loss': -acc, 'status': STATUS_OK}
#
#
# trials = Trials()
# best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
# print('best:')
# print(best)
#
# ------------------------GBDT调参---------------------------------------
#
# model = GradientBoostingClassifier(random_state=37)  # KNN
# scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1)
# print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
#
# def hyperopt_train_test(params):
#     X_ = all_input_features[:]
#     # if 'normalize' in params:
#     #     if params['normalize'] == 1:
#     #         X_ = normalize(X_)
#     #         del params['normalize']
#     #
#     # if 'scale' in params:
#     #     if params['scale'] == 1:
#     #         X_ = scale(X_)
#     #         del params['scale']
#     clf = GradientBoostingClassifier(n_estimators=int(params["n_estimators"])
#                                      , learning_rate=params["learning_rate"]
#                                      # , criterion=params["criterion"]
#                                      # , max_depth=int(params["max_depth"])
#                                      # , max_features=int(params["max_features"])
#                                      # , subsample=params["subsample"]
#                                      # , min_impurity_decrease=params["min_impurity_decrease"]
#                                      , random_state=1412
#                                      , verbose=False)
#     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1).mean()
#
#
# space4dt = {'n_estimators': hp.quniform("n_estimators", 10, 200, 10)
#             , "learning_rate": hp.quniform("learning_rate", 0.05, 1, 0.05)
#             # , "criterion": hp.choice("criterion", ["friedman_mse", "squared_error", "squared_error", "absolute_error"])
#             # , "max_depth": hp.quniform("max_depth", 10, 100, 1)
#             # , "subsample": hp.quniform("subsample", 0.5, 1, 0.05)
#             # , "max_features": hp.quniform("max_features", 0, 30, 1)
#             # , "min_impurity_decrease": hp.quniform("min_impurity_decrease", 0, 5, 0.5)
#             }
#
#
# def f(params):
#     acc = hyperopt_train_test(params)
#     return {'loss': -acc, 'status': STATUS_OK}
#
#
# def param_hyperopt(max_evals=100):
#     # 保存迭代过程
#     trials = Trials()
#
#     # 设置提前停止
#     # early_stop_fn = no_progress_loss(100)
#
#     # 定义代理模型
#     params_best = fmin(f
#                        , space=space4dt
#                        , algo=tpe.suggest
#                        , max_evals=max_evals
#                        , verbose=True
#                        , trials=trials
#                        # , early_stop_fn=early_stop_fn
#                        )
#
#     # 打印最优参数，fmin会自动打印最佳分数
#     print("\n", "\n", "best params: ", params_best,
#           "\n")
#     return params_best, trials
#
#
# best, trials = param_hyperopt(60)
# print('best:')
# print(best)
#
# -----------DT调参---------------------------------------
# para = {'max_depth': 899, 'min_samples_leaf': 0.0275, 'min_samples_split': 0.0236}
# model = DecisionTreeClassifier(**para)  # KNN
# scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1)
# print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
#
# def hyperopt_train_test(params):
#     clf = DecisionTreeClassifier(**params)
#     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1).mean()
#
#
# space4dt = {
#     'max_depth': hp.choice('max_depth', range(1, 1000)),
#     'min_samples_leaf':hp.uniform('min_samples_leaf',0,0.5),
#     'min_samples_split':hp.uniform('min_samples_split',0,1.0)
# }
#
#
# def f(params):
#     acc = hyperopt_train_test(params)
#     return {'loss': -acc, 'status': STATUS_OK}
#
#
# def param_hyperopt(max_evals=100):
#     # 保存迭代过程
#     trials = Trials()
#
#     # 设置提前停止
#     # early_stop_fn = no_progress_loss(300)
#
#     # 定义代理模型
#     params_best = fmin(f
#                        , space=space4dt
#                        , algo=tpe.suggest
#                        , max_evals=max_evals
#                        , verbose=True
#                        , trials=trials
#                        # , early_stop_fn=early_stop_fn
#                        )
#
#     # 打印最优参数，fmin会自动打印最佳分数
#     print("\n", "\n", "best params: ", params_best,
#           "\n")
#     return params_best, trials
#
#
# best, trials = param_hyperopt(800)
# print('best:')
# print(best)
# -----------LGBM调参---------------------------------------
# para_lightGBM = {'learning_rate': 0.05, 'n_estimators': 7,'num_leaves': 20}
# print("LGBM调参：")
# model = LGBMClassifier(**para_lightGBM)  # adaboost
# scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5, n_jobs=-1, verbose=30)
# print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
#
#
# def hyperopt_train_test(params):
#     clf = LGBMClassifier(**params)
#     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5, n_jobs=-1, verbose=30).mean()
#
#
# space4dt = {
#     'num_leaves': hp.choice('num_leaves', range(5, 200, 5)),  # ??
#     "learning_rate": hp.quniform("learning_rate", 0.05, 1, 0.05),  # 学习率
#     'n_estimators': hp.choice('n_estimators', range(5, 250, 5)),  # 学习器的数量
#
# }
#
#
# def f(params):
#     acc = hyperopt_train_test(params)
#     return {'loss': -acc, 'status': STATUS_OK}
#
#
# def param_hyperopt(max_evals=100):
#     # 保存迭代过程
#     trials = Trials()
#
#     # 设置提前停止
#     # early_stop_fn = no_progress_loss(300)
#
#     # 定义代理模型
#     params_best = fmin(f
#                        , space=space4dt
#                        , algo=tpe.suggest
#                        , max_evals=max_evals
#                        , verbose=True
#                        , trials=trials
#                        # , early_stop_fn=early_stop_fn
#                        )
#
#     # 打印最优参数，fmin会自动打印最佳分数
#     print("\n", "\n", "best params: ", params_best,
#           "\n")
#     return params_best, trials
#
#
# best, trials = param_hyperopt(60)
# print('best:')
# print(best)
#
# -----------MLP调参---------------------------------------
#
# def hyperopt_train_test(params):
#     clf = MLPClassifier(**params,max_iter=500)
#     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5).mean()
#
#
# space4dt = {
#           "hidden_layer_sizes":hp.choice("hidden_layer_sizes",range(1,1000,10))
#           # "learning_rate":hp.loguniform("learning_rate",np.log(0.001),np.log(0.5))
#           }
#
#
# def f(params):
#     acc = hyperopt_train_test(params)
#     return {'loss': -acc, 'status': STATUS_OK}
#
#
# def param_hyperopt(max_evals=100):
#     # 保存迭代过程
#     trials = Trials()
#
#     # 设置提前停止
#     # early_stop_fn = no_progress_loss(300)
#
#     # 定义代理模型
#     params_best = fmin(f
#                        , space=space4dt
#                        , algo=tpe.suggest
#                        , max_evals=max_evals
#                        , verbose=True
#                        , trials=trials
#                        # , early_stop_fn=early_stop_fn
#                        )
#
#     # 打印最优参数，fmin会自动打印最佳分数
#     print("\n", "\n", "best params: ", params_best,
#           "\n")
#     return params_best, trials
#
#
# best, trials = param_hyperopt(50)
# print('best:')
# print(best)
#
#
# ------lr调参---------------------------------------
#
# def hyperopt_train_test(params):
#     clf = LogisticRegression(**params)
#     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5).mean()
#
#
# space4dt = {
#           "C":hp.choice("C",[0.001,0.01,0.1,1,5,10,100])
#           # "learning_rate":hp.loguniform("learning_rate",np.log(0.001),np.log(0.5))
#           }
#
#
# def f(params):
#     acc = hyperopt_train_test(params)
#     return {'loss': -acc, 'status': STATUS_OK}
#
#
# def param_hyperopt(max_evals=100):
#     # 保存迭代过程
#     trials = Trials()
#
#     # 设置提前停止
#     # early_stop_fn = no_progress_loss(300)
#
#     # 定义代理模型
#     params_best = fmin(f
#                        , space=space4dt
#                        , algo=tpe.suggest
#                        , max_evals=max_evals
#                        , verbose=True
#                        , trials=trials
#                        # , early_stop_fn=early_stop_fn
#                        )
#
#     # 打印最优参数，fmin会自动打印最佳分数
#     print("\n", "\n", "best params: ", params_best,
#           "\n")
#     return params_best, trials
#
#
# best, trials = param_hyperopt(50)
# print('best:')
# print(best)
#
# ---------------------------RF随机森林调参---------------------------------------
# model = RandomForestClassifier()  # KNN
# scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30)
# print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
#
# def hyperopt_train_test(params):
#     clf = RandomForestClassifier(**params)
#     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30).mean()
#
#
# space4dt = {
#     'max_depth': hp.choice('max_depth', range(5,200,5)),
#     'n_estimators': hp.choice('n_estimators', range(5,200,5)),
# }
#
#
# def f(params):
#     acc = hyperopt_train_test(params)
#     return {'loss': -acc, 'status': STATUS_OK}
#
#
# def param_hyperopt(max_evals=100):
#     # 保存迭代过程
#     trials = Trials()
#
#     # 设置提前停止
#     # early_stop_fn = no_progress_loss(300)
#
#     # 定义代理模型
#     params_best = fmin(f
#                        , space=space4dt
#                        , algo=tpe.suggest
#                        , max_evals=max_evals
#                        , verbose=True
#                        , trials=trials
#                        # , early_stop_fn=early_stop_fn
#                        )
#
#     # 打印最优参数，fmin会自动打印最佳分数
#     print("\n", "\n", "best params: ", params_best,
#           "\n")
#     return params_best, trials
#
#
# best, trials = param_hyperopt(200)
# print('best:')
# print(best)
#
# ---------------------------adaboost调参---------------------------------------
# print("adaboost调参：")
# model = AdaBoostClassifier()  # adaboost
# scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30)
# print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
#
# def hyperopt_train_test(params):
#     clf = AdaBoostClassifier(**params)
#     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30).mean()
#
#
# space4dt = {
#     "learning_rate": hp.quniform("learning_rate", 0.05, 1, 0.05),
#     'n_estimators': hp.choice('n_estimators', range(5,250,5))
# }
#
#
# def f(params):
#     acc = hyperopt_train_test(params)
#     return {'loss': -acc, 'status': STATUS_OK}
#
#
# def param_hyperopt(max_evals=100):
#     # 保存迭代过程
#     trials = Trials()
#
#     # 设置提前停止
#     # early_stop_fn = no_progress_loss(300)
#
#     # 定义代理模型
#     params_best = fmin(f
#                        , space=space4dt
#                        , algo=tpe.suggest
#                        , max_evals=max_evals
#                        , verbose=True
#                        , trials=trials
#                        # , early_stop_fn=early_stop_fn
#                        )
#
#     # 打印最优参数，fmin会自动打印最佳分数
#     print("\n", "\n", "best params: ", params_best,
#           "\n")
#     return params_best, trials
#
#
# best, trials = param_hyperopt(60)
# print('best:')
# print(best)
# # -*- coding: utf-8 -*-
# """
# @Time ： 2023/2/16 21:26
# @Auth ： 兰宏富
# @File ：adaboost_decisiontree_gridsearch.py
# @IDE ：PyCharm
# """
# import sklearn.ensemble
# from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
# from hyperopt.early_stop import no_progress_loss
# from lightgbm import LGBMClassifier
# from matplotlib import pyplot as plt
# from sklearn import metrics
# from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, GradientBoostingRegressor, \
#     StackingRegressor, StackingClassifier, RandomForestClassifier, AdaBoostClassifier  # AdaBoost分类器
# from sklearn.linear_model import LogisticRegression, RidgeCV, SGDClassifier, Ridge, RidgeClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier  # 决策树分类器
# from bayes_opt import BayesianOptimization  # 贝叶斯优化
# from sklearn.model_selection import cross_val_score
#
# # --------------------------------------数据集读取原始版本------------------------------------------
# # wine = load_wine()
# # print(f"所有特征：{wine.feature_names}")
# # X = pd.DataFrame(wine.data, columns=wine.feature_names)
# # y = pd.Series(wine.target)
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
# # --------------------------------------数据集读取原始版本------------------------------------------
#
#
# # --------------------------------------数据集读取我的版本start------------------------------------------
# # model_name = 'huggingface/CodeBERTa-small-v1'  # 准确率最高
# model_name = 'distilbert-base-uncased'  # 准确率第二
# # model_name = 'bert-base-uncased'  # 准确率第三
# csn_path = '../dataset/Commit_dataset_final.xlsx'
#
# import logging
#
# logging.basicConfig(level=logging.INFO)
# import torch
# from transformers import AutoTokenizer, \
#     AutoModel
#
# # Load pre-trained model tokenizer (vocabulary)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# import pandas as pd
#
# df = pd.read_excel(csn_path,
#                    header=0,  # 不读取第一行的表头
#                    index_col=False,  # 设置为False就会读取第一列，设置为0不会读取第一列
#                    )
# # Tokenize input
# tokenized_text = df['comment'].apply(
#     (lambda x: tokenizer.encode(x, max_length=80, add_special_tokens=True, truncation=True)))
#
# import numpy as np
#
# max_len = 0
# for i in tokenized_text.values:
#     if len(i) > max_len:
#         max_len = len(i)
#
# padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized_text.values])
#
# np.array(padded).shape
# attention_mask = np.where(padded != 0, 1, 0)
# attention_mask.shape
#
# model = AutoModel.from_pretrained(model_name)
# input_ids = torch.tensor(padded)
# attention_mask = torch.tensor(attention_mask)
#
# with torch.no_grad():
#     last_hidden_states = model(input_ids, attention_mask=attention_mask)
#
# bert_features = last_hidden_states[0][:, 0, :].numpy()  # BERT的输出：768维度的语义向量
# print(bert_features.shape)
# bert_features.shape
#
# # labels = df['3_labels']
# # print(labels)
# change = {'p': 0, 'a': 1, 'c': 2}  # 替换的值
# labels = df['3_labels'].map(change).values  # 标签值
# cc = df.drop(columns=['3_labels', 'comment', 'project', 'commit_id', 'comment_diff'], axis=1).replace(np.nan,
#                                                                                                       0)  # 将空的格子设置为0
# print(cc)
#
# from sklearn.preprocessing import StandardScaler, normalize, scale
#
# sc = StandardScaler()
# cc = sc.fit_transform(cc)
# print(cc)
#
# all_input_features = np.concatenate((bert_features, cc), axis=1)  # 拼接语义和特征
# all_input_features[0].shape
#
# # encoder = LabelBinarizer() # 转换成独热编码
# # labels = encoder.fit_transform(labels)
# # print(labels)
#
# from sklearn.model_selection import train_test_split, cross_val_score
#
# # stratified train_test_split
# X_train, X_test, y_train, y_test = train_test_split(all_input_features, labels, test_size=0.2, random_state=42,
#                                                     stratify=labels)  # 80% 训练，20%测试结果
#
# # --------------------------------------数据集读取我的版本end------------------------------------------
#
#
# # ------------------------设置基础模型---------------------------------------
# # model1 = sklearn.svm.SVC(probability=False)  # SVM
# # model2 = KNeighborsClassifier()  # KNN
# # model3 = GradientBoostingClassifier(random_state=37)  # GBDT
# # model4 = DecisionTreeClassifier(random_state=666)  # DT
# # model5 = LGBMClassifier(learning_rate=0.1, n_estimators=100)  # GBM
# # model6 = MLPClassifier(hidden_layer_sizes=(100,), random_state=420, max_iter=1000)  # MLP
# # model7 = LogisticRegression(max_iter=1000)  # LogisticRegression
# # model8 = RandomForestClassifier()  # RF随机森林
#
# # para_svm = {'C': 0.0224433889890122, 'coef0': 2.248410437719281, 'gamma': 12.923005804808009, 'kernel': 'linear'}
# # para_knn = {'n_neighbors': 26}
# # para_GBM = {'boosting_type': 'dart', 'learning_rate': 0.1218080101678672, 'num_leaves': 20}
# # para_MLP={'hidden_layer_sizes': 50}
# # para_LR={'C': 1}
# # para_RF = {'criterion': "gini", 'max_depth': 92, 'max_features': 3, 'min_samples_split': 3, 'n_estimators': 77}
# # x = sklearn.svm.SVC(**para_svm)
# # x = KNeighborsClassifier(**para_svm)
# # x = LGBMClassifier(**para_GBM)  # GBM
# # x = MLPClassifier(hidden_layer_sizes=50, random_state=420, max_iter=1000)  # MLP
# # x = LogisticRegression(C=1,max_iter=1000)  # LogisticRegression
# # x = RandomForestClassifier(**para_RF)  # RF随机森林（添加了参数比没添加参数更低）
# # scores = cross_val_score(x, all_input_features, labels, scoring='accuracy', cv=5)
# # print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
# # x.fit(X_train, y_train)
# # test_predict = x.predict(X_test)
# # confusion_matrix_result = metrics.confusion_matrix(test_predict, y_test, labels=[0, 1, 2])
# # print(confusion_matrix_result)
# # print(classification_report(test_predict, y_test, digits=4))  # 保留四位小数点
#
# #------------------------SVM调参---------------------------------------
# model = sklearn.svm.SVC(probability=False)
# scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5)
# print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
#
# def hyperopt_train_test(params):
#     X_ = all_input_features[:]
#
#     # if 'normalize' in params:
#     #     if params['normalize'] == 1:
#     #         X_ = normalize(X_)
#     #         del params['normalize']
#
#     # if 'scale' in params:
#     #     if params['scale'] == 1:
#     #         X_ = scale(X_)
#     #         del params['scale']
#
#     clf = SVC(**params)
#     return cross_val_score(clf, X_, labels,scoring='accuracy', cv=5).mean()
#
# space4svm = {
#     'C': hp.quniform('C', 0.1, 2,0.05),
#     # 'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
#     # 'gamma': hp.uniform('gamma', 0, 20),
#     # 'coef0': hp.uniform('coef0', 0, 10),
#     # 'scale': hp.choice('scale', [0, 1]),
#     # 'normalize': hp.choice('normalize', [0, 1])
# }
#
# def f(params):
#     acc = hyperopt_train_test(params)
#     return {'loss': -acc, 'status': STATUS_OK}
#
# trials = Trials()
# best = fmin(f, space4svm, algo=tpe.suggest, max_evals=100, trials=trials)
# print('best:')
# print(best)
#
# # ------------------------KNN调参---------------------------------------
# model = KNeighborsClassifier()  # KNN
# scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5)
# print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
#
# def hyperopt_train_test(params):
#     clf = KNeighborsClassifier(**params)
#     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5).mean()
#
#
# space4knn = {
#     'n_neighbors': hp.choice('n_neighbors', range(2, 31)),
#     'leaf_size': hp.quniform('leaf_size', 10, 50, 10)
# }
#
#
# def f(params):
#     acc = hyperopt_train_test(params)
#     return {'loss': -acc, 'status': STATUS_OK}
#
#
# trials = Trials()
# best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
# print('best:')
# print(best)
#
# # ------------------------SGD调参---------------------------------------
# model = SGDClassifier()  # KNN
# scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30)
# print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
#
# def hyperopt_train_test(params):
#     clf = SGDClassifier(**params)
#     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30).mean()
#
#
# space4knn = {
#     'l1_ratio': hp.choice('l1_ratio', [0.1,0.15,0.2,0.25,0.3,0.35,0.4]),
#     'alpha': hp.choice('alpha', [0.0001,0.001,0.01,0.1])
# }
#
#
# def f(params):
#     acc = hyperopt_train_test(params)
#     return {'loss': -acc, 'status': STATUS_OK}# -*- coding: utf-8 -*-
# """
# @Time ： 2023/2/16 21:26
# @Auth ： 兰宏富
# @File ：adaboost_decisiontree_gridsearch.py
# @IDE ：PyCharm
# """
# import sklearn.ensemble
# from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
# from hyperopt.early_stop import no_progress_loss
# from lightgbm import LGBMClassifier
# from matplotlib import pyplot as plt
# from sklearn import metrics
# from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, GradientBoostingRegressor, \
#     StackingRegressor, StackingClassifier, RandomForestClassifier, AdaBoostClassifier  # AdaBoost分类器
# from sklearn.linear_model import LogisticRegression, RidgeCV, SGDClassifier, Ridge, RidgeClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier  # 决策树分类器
# from bayes_opt import BayesianOptimization  # 贝叶斯优化
# from sklearn.model_selection import cross_val_score
#
# # --------------------------------------数据集读取原始版本------------------------------------------
# # wine = load_wine()
# # print(f"所有特征：{wine.feature_names}")
# # X = pd.DataFrame(wine.data, columns=wine.feature_names)
# # y = pd.Series(wine.target)
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
# # --------------------------------------数据集读取原始版本------------------------------------------
#
#
# # --------------------------------------数据集读取我的版本start------------------------------------------
# # model_name = 'huggingface/CodeBERTa-small-v1'  # 准确率最高
# model_name = 'distilbert-base-uncased'  # 准确率第二
# # model_name = 'bert-base-uncased'  # 准确率第三
# csn_path = '../dataset/Commit_dataset_final.xlsx'
#
# import logging
#
# logging.basicConfig(level=logging.INFO)
# import torch
# from transformers import AutoTokenizer, \
#     AutoModel
#
# # Load pre-trained model tokenizer (vocabulary)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# import pandas as pd
#
# df = pd.read_excel(csn_path,
#                    header=0,  # 不读取第一行的表头
#                    index_col=False,  # 设置为False就会读取第一列，设置为0不会读取第一列
#                    )
# # Tokenize input
# tokenized_text = df['comment'].apply(
#     (lambda x: tokenizer.encode(x, max_length=80, add_special_tokens=True, truncation=True)))
#
# import numpy as np
#
# max_len = 0
# for i in tokenized_text.values:
#     if len(i) > max_len:
#         max_len = len(i)
#
# padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized_text.values])
#
# np.array(padded).shape
# attention_mask = np.where(padded != 0, 1, 0)
# attention_mask.shape
#
# model = AutoModel.from_pretrained(model_name)
# input_ids = torch.tensor(padded)
# attention_mask = torch.tensor(attention_mask)
#
# with torch.no_grad():
#     last_hidden_states = model(input_ids, attention_mask=attention_mask)
#
# bert_features = last_hidden_states[0][:, 0, :].numpy()  # BERT的输出：768维度的语义向量
# print(bert_features.shape)
# bert_features.shape
#
# # labels = df['3_labels']
# # print(labels)
# change = {'p': 0, 'a': 1, 'c': 2}  # 替换的值
# labels = df['3_labels'].map(change).values  # 标签值
# cc = df.drop(columns=['3_labels', 'comment', 'project', 'commit_id', 'comment_diff'], axis=1).replace(np.nan,
#                                                                                                       0)  # 将空的格子设置为0
# print(cc)
#
# from sklearn.preprocessing import StandardScaler, normalize, scale
#
# sc = StandardScaler()
# cc = sc.fit_transform(cc)
# print(cc)
#
# all_input_features = np.concatenate((bert_features, cc), axis=1)  # 拼接语义和特征
# all_input_features[0].shape
#
# # encoder = LabelBinarizer() # 转换成独热编码
# # labels = encoder.fit_transform(labels)
# # print(labels)
#
# from sklearn.model_selection import train_test_split, cross_val_score
#
# # stratified train_test_split
# X_train, X_test, y_train, y_test = train_test_split(all_input_features, labels, test_size=0.2, random_state=42,
#                                                     stratify=labels)  # 80% 训练，20%测试结果
#
# # --------------------------------------数据集读取我的版本end------------------------------------------
#
#
# # ------------------------设置基础模型---------------------------------------
# # model1 = sklearn.svm.SVC(probability=False)  # SVM
# # model2 = KNeighborsClassifier()  # KNN
# # model3 = GradientBoostingClassifier(random_state=37)  # GBDT
# # model4 = DecisionTreeClassifier(random_state=666)  # DT
# # model5 = LGBMClassifier(learning_rate=0.1, n_estimators=100)  # GBM
# # model6 = MLPClassifier(hidden_layer_sizes=(100,), random_state=420, max_iter=1000)  # MLP
# # model7 = LogisticRegression(max_iter=1000)  # LogisticRegression
# # model8 = RandomForestClassifier()  # RF随机森林
#
# # para_svm = {'C': 0.0224433889890122, 'coef0': 2.248410437719281, 'gamma': 12.923005804808009, 'kernel': 'linear'}
# # para_knn = {'n_neighbors': 26}
# # para_GBM = {'boosting_type': 'dart', 'learning_rate': 0.1218080101678672, 'num_leaves': 20}
# # para_MLP={'hidden_layer_sizes': 50}
# # para_LR={'C': 1}
# # para_RF = {'criterion': "gini", 'max_depth': 92, 'max_features': 3, 'min_samples_split': 3, 'n_estimators': 77}
# # x = sklearn.svm.SVC(**para_svm)
# # x = KNeighborsClassifier(**para_svm)
# # x = LGBMClassifier(**para_GBM)  # GBM
# # x = MLPClassifier(hidden_layer_sizes=50, random_state=420, max_iter=1000)  # MLP
# # x = LogisticRegression(C=1,max_iter=1000)  # LogisticRegression
# # x = RandomForestClassifier(**para_RF)  # RF随机森林（添加了参数比没添加参数更低）
# # scores = cross_val_score(x, all_input_features, labels, scoring='accuracy', cv=5)
# # print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
# # x.fit(X_train, y_train)
# # test_predict = x.predict(X_test)
# # confusion_matrix_result = metrics.confusion_matrix(test_predict, y_test, labels=[0, 1, 2])
# # print(confusion_matrix_result)
# # print(classification_report(test_predict, y_test, digits=4))  # 保留四位小数点
#
# # ------------------------SVM调参---------------------------------------
# # model = sklearn.svm.SVC(probability=False)
# # scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5)
# # print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
# #
# # def hyperopt_train_test(params):
# #     X_ = all_input_features[:]
# #
# #     # if 'normalize' in params:
# #     #     if params['normalize'] == 1:
# #     #         X_ = normalize(X_)
# #     #         del params['normalize']
# #
# #     # if 'scale' in params:
# #     #     if params['scale'] == 1:
# #     #         X_ = scale(X_)
# #     #         del params['scale']
# #
# #     clf = SVC(**params)
# #     return cross_val_score(clf, X_, labels,scoring='accuracy', cv=5).mean()
# #
# # space4svm = {
# #     'C': hp.quniform('C', 0.1, 2,0.05),
# #     # 'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
# #     # 'gamma': hp.uniform('gamma', 0, 20),
# #     # 'coef0': hp.uniform('coef0', 0, 10),
# #     # 'scale': hp.choice('scale', [0, 1]),
# #     # 'normalize': hp.choice('normalize', [0, 1])
# # }
# #
# # def f(params):
# #     acc = hyperopt_train_test(params)
# #     return {'loss': -acc, 'status': STATUS_OK}
# #
# # trials = Trials()
# # best = fmin(f, space4svm, algo=tpe.suggest, max_evals=100, trials=trials)
# # print('best:')
# # print(best)
#
# # ------------------------KNN调参---------------------------------------
# model = KNeighborsClassifier()  # KNN
# scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5)
# print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
#
# def hyperopt_train_test(params):
#     clf = KNeighborsClassifier(**params)
#     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5).mean()
#
#
# space4knn = {
#     'n_neighbors': hp.choice('n_neighbors', range(2, 31)),
#     'leaf_size': hp.quniform('leaf_size', 10, 50, 10)
# }
#
#
# def f(params):
#     acc = hyperopt_train_test(params)
#     return {'loss': -acc, 'status': STATUS_OK}
#
#
# trials = Trials()
# best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
# print('best:')
# print(best)
#
# # ------------------------SGD调参---------------------------------------
# # model = SGDClassifier()  # KNN
# # scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30)
# # print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
# #
# # def hyperopt_train_test(params):
# #     clf = SGDClassifier(**params)
# #     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30).mean()
# #
# #
# # space4knn = {
# #     'l1_ratio': hp.choice('l1_ratio', [0.1,0.15,0.2,0.25,0.3,0.35,0.4]),
# #     'alpha': hp.choice('alpha', [0.0001,0.001,0.01,0.1])
# # }
# #
# #
# # def f(params):
# #     acc = hyperopt_train_test(params)
# #     return {'loss': -acc, 'status': STATUS_OK}
# #
# #
# # trials = Trials()
# # best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
# # print('best:')
# # print(best)
#
# # -----------------岭回归分类器调参---------------------------------------
# # model = RidgeClassifier()  # 注意，如果导入的是Ridge那么将报错
# # scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5)
# # print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
# #
# #
# # def hyperopt_train_test(params):
# #     clf = RidgeClassifier(**params)
# #     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5).mean()
# #
# #
# # space4knn = {
# #     'alpha': hp.quniform('alpha',0,2,0.05)
# # }
# #
# #
# # def f(params):
# #     acc = hyperopt_train_test(params)
# #     return {'loss': -acc, 'status': STATUS_OK}
# #
# #
# # trials = Trials()
# # best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
# # print('best:')
# # print(best)
#
# # ------------------------NB调参---------------------------------------
# # model = GaussianNB()  # GaussianNB
# # scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30)
# # print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
# #
# # def hyperopt_train_test(params):
# #     clf = GaussianNB(**params)
# #     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30).mean()
# #
# #
# # space4knn = {
# #     'var_smoothing': hp.choice('var_smoothing', [1,0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001,0.00000001,0.000000001])
# # }
# #
# #
# # def f(params):
# #     acc = hyperopt_train_test(params)
# #     return {'loss': -acc, 'status': STATUS_OK}
# #
# #
# # trials = Trials()
# # best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
# # print('best:')
# # print(best)
#
# # ------------------------GBDT调参---------------------------------------
#
# # model = GradientBoostingClassifier(random_state=37)  # KNN
# # scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1)
# # print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
# #
# # def hyperopt_train_test(params):
# #     X_ = all_input_features[:]
# #     # if 'normalize' in params:
# #     #     if params['normalize'] == 1:
# #     #         X_ = normalize(X_)
# #     #         del params['normalize']
# #     #
# #     # if 'scale' in params:
# #     #     if params['scale'] == 1:
# #     #         X_ = scale(X_)
# #     #         del params['scale']
# #     clf = GradientBoostingClassifier(n_estimators=int(params["n_estimators"])
# #                                      , learning_rate=params["learning_rate"]
# #                                      # , criterion=params["criterion"]
# #                                      # , max_depth=int(params["max_depth"])
# #                                      # , max_features=int(params["max_features"])
# #                                      # , subsample=params["subsample"]
# #                                      # , min_impurity_decrease=params["min_impurity_decrease"]
# #                                      , random_state=1412
# #                                      , verbose=False)
# #     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1).mean()
# #
# #
# # space4dt = {'n_estimators': hp.quniform("n_estimators", 10, 200, 10)
# #             , "learning_rate": hp.quniform("learning_rate", 0.05, 1, 0.05)
# #             # , "criterion": hp.choice("criterion", ["friedman_mse", "squared_error", "squared_error", "absolute_error"])
# #             # , "max_depth": hp.quniform("max_depth", 10, 100, 1)
# #             # , "subsample": hp.quniform("subsample", 0.5, 1, 0.05)
# #             # , "max_features": hp.quniform("max_features", 0, 30, 1)
# #             # , "min_impurity_decrease": hp.quniform("min_impurity_decrease", 0, 5, 0.5)
# #             }
# #
# #
# # def f(params):
# #     acc = hyperopt_train_test(params)
# #     return {'loss': -acc, 'status': STATUS_OK}
# #
# #
# # def param_hyperopt(max_evals=100):
# #     # 保存迭代过程
# #     trials = Trials()
# #
# #     # 设置提前停止
# #     # early_stop_fn = no_progress_loss(100)
# #
# #     # 定义代理模型
# #     params_best = fmin(f
# #                        , space=space4dt
# #                        , algo=tpe.suggest
# #                        , max_evals=max_evals
# #                        , verbose=True
# #                        , trials=trials
# #                        # , early_stop_fn=early_stop_fn
# #                        )
# #
# #     # 打印最优参数，fmin会自动打印最佳分数
# #     print("\n", "\n", "best params: ", params_best,
# #           "\n")
# #     return params_best, trials
# #
# #
# # best, trials = param_hyperopt(60)
# # print('best:')
# # print(best)
#
# # -----------DT调参---------------------------------------
# # para = {'max_depth': 899, 'min_samples_leaf': 0.0275, 'min_samples_split': 0.0236}
# # model = DecisionTreeClassifier(**para)  # KNN
# # scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1)
# # print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
# #
# # def hyperopt_train_test(params):
# #     clf = DecisionTreeClassifier(**params)
# #     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1).mean()
# #
# #
# # space4dt = {
# #     'max_depth': hp.choice('max_depth', range(1, 1000)),
# #     'min_samples_leaf':hp.uniform('min_samples_leaf',0,0.5),
# #     'min_samples_split':hp.uniform('min_samples_split',0,1.0)
# # }
# #
# #
# # def f(params):
# #     acc = hyperopt_train_test(params)
# #     return {'loss': -acc, 'status': STATUS_OK}
# #
# #
# # def param_hyperopt(max_evals=100):
# #     # 保存迭代过程
# #     trials = Trials()
# #
# #     # 设置提前停止
# #     # early_stop_fn = no_progress_loss(300)
# #
# #     # 定义代理模型
# #     params_best = fmin(f
# #                        , space=space4dt
# #                        , algo=tpe.suggest
# #                        , max_evals=max_evals
# #                        , verbose=True
# #                        , trials=trials
# #                        # , early_stop_fn=early_stop_fn
# #                        )
# #
# #     # 打印最优参数，fmin会自动打印最佳分数
# #     print("\n", "\n", "best params: ", params_best,
# #           "\n")
# #     return params_best, trials
# #
# #
# # best, trials = param_hyperopt(800)
# # print('best:')
# # print(best)
# # -----------LGBM调参---------------------------------------
# # para_lightGBM = {'learning_rate': 0.05, 'n_estimators': 7,'num_leaves': 20}
# # print("LGBM调参：")
# # model = LGBMClassifier(**para_lightGBM)  # adaboost
# # scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5, n_jobs=-1, verbose=30)
# # print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
# #
# #
# # def hyperopt_train_test(params):
# #     clf = LGBMClassifier(**params)
# #     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5, n_jobs=-1, verbose=30).mean()
# #
# #
# # space4dt = {
# #     'num_leaves': hp.choice('num_leaves', range(5, 200, 5)),  # ??
# #     "learning_rate": hp.quniform("learning_rate", 0.05, 1, 0.05),  # 学习率
# #     'n_estimators': hp.choice('n_estimators', range(5, 250, 5)),  # 学习器的数量
# #
# # }
# #
# #
# # def f(params):
# #     acc = hyperopt_train_test(params)
# #     return {'loss': -acc, 'status': STATUS_OK}
# #
# #
# # def param_hyperopt(max_evals=100):
# #     # 保存迭代过程
# #     trials = Trials()
# #
# #     # 设置提前停止
# #     # early_stop_fn = no_progress_loss(300)
# #
# #     # 定义代理模型
# #     params_best = fmin(f
# #                        , space=space4dt
# #                        , algo=tpe.suggest
# #                        , max_evals=max_evals
# #                        , verbose=True
# #                        , trials=trials
# #                        # , early_stop_fn=early_stop_fn
# #                        )
# #
# #     # 打印最优参数，fmin会自动打印最佳分数
# #     print("\n", "\n", "best params: ", params_best,
# #           "\n")
# #     return params_best, trials
# #
# #
# # best, trials = param_hyperopt(60)
# # print('best:')
# # print(best)
#
# # -----------MLP调参---------------------------------------
#
# # def hyperopt_train_test(params):
# #     clf = MLPClassifier(**params,max_iter=500)
# #     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5).mean()
# #
# #
# # space4dt = {
# #           "hidden_layer_sizes":hp.choice("hidden_layer_sizes",range(1,1000,10))
# #           # "learning_rate":hp.loguniform("learning_rate",np.log(0.001),np.log(0.5))
# #           }
# #
# #
# # def f(params):
# #     acc = hyperopt_train_test(params)
# #     return {'loss': -acc, 'status': STATUS_OK}
# #
# #
# # def param_hyperopt(max_evals=100):
# #     # 保存迭代过程
# #     trials = Trials()
# #
# #     # 设置提前停止
# #     # early_stop_fn = no_progress_loss(300)
# #
# #     # 定义代理模型
# #     params_best = fmin(f
# #                        , space=space4dt
# #                        , algo=tpe.suggest
# #                        , max_evals=max_evals
# #                        , verbose=True
# #                        , trials=trials
# #                        # , early_stop_fn=early_stop_fn
# #                        )
# #
# #     # 打印最优参数，fmin会自动打印最佳分数
# #     print("\n", "\n", "best params: ", params_best,
# #           "\n")
# #     return params_best, trials
# #
# #
# # best, trials = param_hyperopt(50)
# # print('best:')
# # print(best)
#
#
# # ------lr调参---------------------------------------
#
# # def hyperopt_train_test(params):
# #     clf = LogisticRegression(**params)
# #     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5).mean()
# #
# #
# # space4dt = {
# #           "C":hp.choice("C",[0.001,0.01,0.1,1,5,10,100])
# #           # "learning_rate":hp.loguniform("learning_rate",np.log(0.001),np.log(0.5))
# #           }
# #
# #
# # def f(params):
# #     acc = hyperopt_train_test(params)
# #     return {'loss': -acc, 'status': STATUS_OK}
# #
# #
# # def param_hyperopt(max_evals=100):
# #     # 保存迭代过程
# #     trials = Trials()
# #
# #     # 设置提前停止
# #     # early_stop_fn = no_progress_loss(300)
# #
# #     # 定义代理模型
# #     params_best = fmin(f
# #                        , space=space4dt
# #                        , algo=tpe.suggest
# #                        , max_evals=max_evals
# #                        , verbose=True
# #                        , trials=trials
# #                        # , early_stop_fn=early_stop_fn
# #                        )
# #
# #     # 打印最优参数，fmin会自动打印最佳分数
# #     print("\n", "\n", "best params: ", params_best,
# #           "\n")
# #     return params_best, trials
# #
# #
# # best, trials = param_hyperopt(50)
# # print('best:')
# # print(best)
#
# # ---------------------------RF随机森林调参---------------------------------------
# # model = RandomForestClassifier()  # KNN
# # scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30)
# # print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
# #
# # def hyperopt_train_test(params):
# #     clf = RandomForestClassifier(**params)
# #     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30).mean()
# #
# #
# # space4dt = {
# #     'max_depth': hp.choice('max_depth', range(5,200,5)),
# #     'n_estimators': hp.choice('n_estimators', range(5,200,5)),
# # }
# #
# #
# # def f(params):
# #     acc = hyperopt_train_test(params)
# #     return {'loss': -acc, 'status': STATUS_OK}
# #
# #
# # def param_hyperopt(max_evals=100):
# #     # 保存迭代过程
# #     trials = Trials()
# #
# #     # 设置提前停止
# #     # early_stop_fn = no_progress_loss(300)
# #
# #     # 定义代理模型
# #     params_best = fmin(f
# #                        , space=space4dt
# #                        , algo=tpe.suggest
# #                        , max_evals=max_evals
# #                        , verbose=True
# #                        , trials=trials
# #                        # , early_stop_fn=early_stop_fn
# #                        )
# #
# #     # 打印最优参数，fmin会自动打印最佳分数
# #     print("\n", "\n", "best params: ", params_best,
# #           "\n")
# #     return params_best, trials
# #
# #
# # best, trials = param_hyperopt(200)
# # print('best:')
# # print(best)
#
# # ---------------------------adaboost调参---------------------------------------
# # print("adaboost调参：")
# # model = AdaBoostClassifier()  # adaboost
# # scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30)
# # print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
# #
# # def hyperopt_train_test(params):
# #     clf = AdaBoostClassifier(**params)
# #     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30).mean()
# #
# #
# # space4dt = {
# #     "learning_rate": hp.quniform("learning_rate", 0.05, 1, 0.05),
# #     'n_estimators': hp.choice('n_estimators', range(5,250,5))
# # }
# #
# #
# # def f(params):
# #     acc = hyperopt_train_test(params)
# #     return {'loss': -acc, 'status': STATUS_OK}
# #
# #
# # def param_hyperopt(max_evals=100):
# #     # 保存迭代过程
# #     trials = Trials()
# #
# #     # 设置提前停止
# #     # early_stop_fn = no_progress_loss(300)
# #
# #     # 定义代理模型
# #     params_best = fmin(f
# #                        , space=space4dt
# #                        , algo=tpe.suggest
# #                        , max_evals=max_evals
# #                        , verbose=True
# #                        , trials=trials
# #                        # , early_stop_fn=early_stop_fn
# #                        )
# #
# #     # 打印最优参数，fmin会自动打印最佳分数
# #     print("\n", "\n", "best params: ", params_best,
# #           "\n")
# #     return params_best, trials
# #
# #
# # best, trials = param_hyperopt(60)
# # print('best:')
# # print(best)
#
#
# trials = Trials()
# best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
# print('best:')
# print(best)
#
# -----------------岭回归分类器调参---------------------------------------
# model = RidgeClassifier()  # 注意，如果导入的是Ridge那么将报错
# scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5)
# print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
#
#
# def hyperopt_train_test(params):
#     clf = RidgeClassifier(**params)
#     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5).mean()
#
#
# space4knn = {
#     'alpha': hp.quniform('alpha',0,2,0.05)
# }
#
#
# def f(params):
#     acc = hyperopt_train_test(params)
#     return {'loss': -acc, 'status': STATUS_OK}
#
#
# trials = Trials()
# best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
# print('best:')
# print(best)
#
# ------------------------NB调参---------------------------------------
# model = GaussianNB()  # GaussianNB
# scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30)
# print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
#
# def hyperopt_train_test(params):
#     clf = GaussianNB(**params)
#     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30).mean()
#
#
# space4knn = {
#     'var_smoothing': hp.choice('var_smoothing', [1,0.1,0.01,0.001,0.0001,0.00001,0.000001,0.0000001,0.00000001,0.000000001])
# }
#
#
# def f(params):
#     acc = hyperopt_train_test(params)
#     return {'loss': -acc, 'status': STATUS_OK}
#
#
# trials = Trials()
# best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
# print('best:')
# print(best)
#
# ------------------------GBDT调参---------------------------------------
#
# model = GradientBoostingClassifier(random_state=37)  # KNN
# scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1)
# print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
#
# def hyperopt_train_test(params):
#     X_ = all_input_features[:]
#     # if 'normalize' in params:
#     #     if params['normalize'] == 1:
#     #         X_ = normalize(X_)
#     #         del params['normalize']
#     #
#     # if 'scale' in params:
#     #     if params['scale'] == 1:
#     #         X_ = scale(X_)
#     #         del params['scale']
#     clf = GradientBoostingClassifier(n_estimators=int(params["n_estimators"])
#                                      , learning_rate=params["learning_rate"]
#                                      # , criterion=params["criterion"]
#                                      # , max_depth=int(params["max_depth"])
#                                      # , max_features=int(params["max_features"])
#                                      # , subsample=params["subsample"]
#                                      # , min_impurity_decrease=params["min_impurity_decrease"]
#                                      , random_state=1412
#                                      , verbose=False)
#     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1).mean()
#
#
# space4dt = {'n_estimators': hp.quniform("n_estimators", 10, 200, 10)
#             , "learning_rate": hp.quniform("learning_rate", 0.05, 1, 0.05)
#             # , "criterion": hp.choice("criterion", ["friedman_mse", "squared_error", "squared_error", "absolute_error"])
#             # , "max_depth": hp.quniform("max_depth", 10, 100, 1)
#             # , "subsample": hp.quniform("subsample", 0.5, 1, 0.05)
#             # , "max_features": hp.quniform("max_features", 0, 30, 1)
#             # , "min_impurity_decrease": hp.quniform("min_impurity_decrease", 0, 5, 0.5)
#             }
#
#
# def f(params):
#     acc = hyperopt_train_test(params)
#     return {'loss': -acc, 'status': STATUS_OK}
#
#
# def param_hyperopt(max_evals=100):
#     # 保存迭代过程
#     trials = Trials()
#
#     # 设置提前停止
#     # early_stop_fn = no_progress_loss(100)
#
#     # 定义代理模型
#     params_best = fmin(f
#                        , space=space4dt
#                        , algo=tpe.suggest
#                        , max_evals=max_evals
#                        , verbose=True
#                        , trials=trials
#                        # , early_stop_fn=early_stop_fn
#                        )
#
#     # 打印最优参数，fmin会自动打印最佳分数
#     print("\n", "\n", "best params: ", params_best,
#           "\n")
#     return params_best, trials
#
#
# best, trials = param_hyperopt(60)
# print('best:')
# print(best)
#
# -----------DT调参---------------------------------------
# para = {'max_depth': 899, 'min_samples_leaf': 0.0275, 'min_samples_split': 0.0236}
# model = DecisionTreeClassifier(**para)  # KNN
# scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1)
# print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
#
# def hyperopt_train_test(params):
#     clf = DecisionTreeClassifier(**params)
#     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1).mean()
#
#
# space4dt = {
#     'max_depth': hp.choice('max_depth', range(1, 1000)),
#     'min_samples_leaf':hp.uniform('min_samples_leaf',0,0.5),
#     'min_samples_split':hp.uniform('min_samples_split',0,1.0)
# }
#
#
# def f(params):
#     acc = hyperopt_train_test(params)
#     return {'loss': -acc, 'status': STATUS_OK}
#
#
# def param_hyperopt(max_evals=100):
#     # 保存迭代过程
#     trials = Trials()
#
#     # 设置提前停止
#     # early_stop_fn = no_progress_loss(300)
#
#     # 定义代理模型
#     params_best = fmin(f
#                        , space=space4dt
#                        , algo=tpe.suggest
#                        , max_evals=max_evals
#                        , verbose=True
#                        , trials=trials
#                        # , early_stop_fn=early_stop_fn
#                        )
#
#     # 打印最优参数，fmin会自动打印最佳分数
#     print("\n", "\n", "best params: ", params_best,
#           "\n")
#     return params_best, trials
#
#
# best, trials = param_hyperopt(800)
# print('best:')
# print(best)
# -----------LGBM调参---------------------------------------
# para_lightGBM = {'learning_rate': 0.05, 'n_estimators': 7,'num_leaves': 20}
# print("LGBM调参：")
# model = LGBMClassifier(**para_lightGBM)  # adaboost
# scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5, n_jobs=-1, verbose=30)
# print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
#
#
# def hyperopt_train_test(params):
#     clf = LGBMClassifier(**params)
#     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5, n_jobs=-1, verbose=30).mean()
#
#
# space4dt = {
#     'num_leaves': hp.choice('num_leaves', range(5, 200, 5)),  # ??
#     "learning_rate": hp.quniform("learning_rate", 0.05, 1, 0.05),  # 学习率
#     'n_estimators': hp.choice('n_estimators', range(5, 250, 5)),  # 学习器的数量
#
# }
#
#
# def f(params):
#     acc = hyperopt_train_test(params)
#     return {'loss': -acc, 'status': STATUS_OK}
#
#
# def param_hyperopt(max_evals=100):
#     # 保存迭代过程
#     trials = Trials()
#
#     # 设置提前停止
#     # early_stop_fn = no_progress_loss(300)
#
#     # 定义代理模型
#     params_best = fmin(f
#                        , space=space4dt
#                        , algo=tpe.suggest
#                        , max_evals=max_evals
#                        , verbose=True
#                        , trials=trials
#                        # , early_stop_fn=early_stop_fn
#                        )
#
#     # 打印最优参数，fmin会自动打印最佳分数
#     print("\n", "\n", "best params: ", params_best,
#           "\n")
#     return params_best, trials
#
#
# best, trials = param_hyperopt(60)
# print('best:')
# print(best)
#
# -----------MLP调参---------------------------------------
#
# def hyperopt_train_test(params):
#     clf = MLPClassifier(**params,max_iter=500)
#     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5).mean()
#
#
# space4dt = {
#           "hidden_layer_sizes":hp.choice("hidden_layer_sizes",range(1,1000,10))
#           # "learning_rate":hp.loguniform("learning_rate",np.log(0.001),np.log(0.5))
#           }
#
#
# def f(params):
#     acc = hyperopt_train_test(params)
#     return {'loss': -acc, 'status': STATUS_OK}
#
#
# def param_hyperopt(max_evals=100):
#     # 保存迭代过程
#     trials = Trials()
#
#     # 设置提前停止
#     # early_stop_fn = no_progress_loss(300)
#
#     # 定义代理模型
#     params_best = fmin(f
#                        , space=space4dt
#                        , algo=tpe.suggest
#                        , max_evals=max_evals
#                        , verbose=True
#                        , trials=trials
#                        # , early_stop_fn=early_stop_fn
#                        )
#
#     # 打印最优参数，fmin会自动打印最佳分数
#     print("\n", "\n", "best params: ", params_best,
#           "\n")
#     return params_best, trials
#
#
# best, trials = param_hyperopt(50)
# print('best:')
# print(best)
#
#
# ------lr调参---------------------------------------
#
# def hyperopt_train_test(params):
#     clf = LogisticRegression(**params)
#     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5).mean()
#
#
# space4dt = {
#           "C":hp.choice("C",[0.001,0.01,0.1,1,5,10,100])
#           # "learning_rate":hp.loguniform("learning_rate",np.log(0.001),np.log(0.5))
#           }
#
#
# def f(params):
#     acc = hyperopt_train_test(params)
#     return {'loss': -acc, 'status': STATUS_OK}
#
#
# def param_hyperopt(max_evals=100):
#     # 保存迭代过程
#     trials = Trials()
#
#     # 设置提前停止
#     # early_stop_fn = no_progress_loss(300)
#
#     # 定义代理模型
#     params_best = fmin(f
#                        , space=space4dt
#                        , algo=tpe.suggest
#                        , max_evals=max_evals
#                        , verbose=True
#                        , trials=trials
#                        # , early_stop_fn=early_stop_fn
#                        )
#
#     # 打印最优参数，fmin会自动打印最佳分数
#     print("\n", "\n", "best params: ", params_best,
#           "\n")
#     return params_best, trials
#
#
# best, trials = param_hyperopt(50)
# print('best:')
# print(best)
#
# ---------------------------RF随机森林调参---------------------------------------
# model = RandomForestClassifier()  # KNN
# scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30)
# print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
#
# def hyperopt_train_test(params):
#     clf = RandomForestClassifier(**params)
#     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30).mean()
#
#
# space4dt = {
#     'max_depth': hp.choice('max_depth', range(5,200,5)),
#     'n_estimators': hp.choice('n_estimators', range(5,200,5)),
# }
#
#
# def f(params):
#     acc = hyperopt_train_test(params)
#     return {'loss': -acc, 'status': STATUS_OK}
#
#
# def param_hyperopt(max_evals=100):
#     # 保存迭代过程
#     trials = Trials()
#
#     # 设置提前停止
#     # early_stop_fn = no_progress_loss(300)
#
#     # 定义代理模型
#     params_best = fmin(f
#                        , space=space4dt
#                        , algo=tpe.suggest
#                        , max_evals=max_evals
#                        , verbose=True
#                        , trials=trials
#                        # , early_stop_fn=early_stop_fn
#                        )
#
#     # 打印最优参数，fmin会自动打印最佳分数
#     print("\n", "\n", "best params: ", params_best,
#           "\n")
#     return params_best, trials
#
#
# best, trials = param_hyperopt(200)
# print('best:')
# print(best)
#
# ---------------------------adaboost调参---------------------------------------
# print("adaboost调参：")
# model = AdaBoostClassifier()  # adaboost
# scores = cross_val_score(model, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30)
# print("Accuracy: %0.4f (+/- %0.4f) [%s]" % (scores.mean(), scores.std(), '最终结果'))
#
# def hyperopt_train_test(params):
#     clf = AdaBoostClassifier(**params)
#     return cross_val_score(clf, all_input_features, labels, scoring='accuracy', cv=5,n_jobs=-1,verbose=30).mean()
#
#
# space4dt = {
#     "learning_rate": hp.quniform("learning_rate", 0.05, 1, 0.05),
#     'n_estimators': hp.choice('n_estimators', range(5,250,5))
# }
#
#
# def f(params):
#     acc = hyperopt_train_test(params)
#     return {'loss': -acc, 'status': STATUS_OK}
#
#
# def param_hyperopt(max_evals=100):
#     # 保存迭代过程
#     trials = Trials()
#
#     # 设置提前停止
#     # early_stop_fn = no_progress_loss(300)
#
#     # 定义代理模型
#     params_best = fmin(f
#                        , space=space4dt
#                        , algo=tpe.suggest
#                        , max_evals=max_evals
#                        , verbose=True
#                        , trials=trials
#                        # , early_stop_fn=early_stop_fn
#                        )
#
#     # 打印最优参数，fmin会自动打印最佳分数
#     print("\n", "\n", "best params: ", params_best,
#           "\n")
#     return params_best, trials
#
#
# best, trials = param_hyperopt(60)
# print('best:')
# print(best)


import sklearn.ensemble
from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, GradientBoostingRegressor, \
    StackingRegressor, StackingClassifier, RandomForestClassifier, AdaBoostClassifier  # AdaBoost分类器
from sklearn.linear_model import LogisticRegression, RidgeCV, SGDClassifier, Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier  # 决策树分类器
from bayes_opt import BayesianOptimization  # 贝叶斯优化
from sklearn.model_selection import cross_val_score, train_test_split, KFold

# --------------------------------------数据集读取我的版本start------------------------------------------
# model_name = 'huggingface/CodeBERTa-small-v1'  # 准确率最高
model_name = 'distilbert-base-uncased'  # 准确率第二
# model_name = 'bert-base-uncased'  # 准确率第三
csn_path = '../dataset/Commit_dataset_final.xlsx'

import logging

logging.basicConfig(level=logging.INFO)
import torch
from transformers import AutoTokenizer, AutoModel

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

# labels = df['3_labels']
# print(labels)
change = {'p': 0, 'a': 1, 'c': 2}  # 替换的值
labels = df['3_labels'].map(change).values  # 标签值
cc = df.drop(columns=['3_labels', 'comment', 'project', 'commit_id', 'comment_diff'], axis=1).replace(np.nan, 0)  # 将空的格子设置为0
print(cc)

from sklearn.preprocessing import StandardScaler, normalize, scale

sc = StandardScaler()
cc = sc.fit_transform(cc)
print(cc)

all_input_features = np.concatenate((bert_features, cc), axis=1)  # 拼接语义和特征
all_input_features[0].shape

# stratified train_test_split
X_train_val, X_test, y_train_val, y_test = train_test_split(all_input_features, labels, test_size=0.2, random_state=42, stratify=labels)  # 80% 训练+验证，20% 测试

# --------------------------------------数据集读取我的版本end------------------------------------------

# 进一步划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125, random_state=42, stratify=y_train_val)  # 7:1

# 用于交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ------------------------模型及调参空间定义---------------------------------------
models = {
    'Decision Tree': (DecisionTreeClassifier(), {'max_depth': hp.choice('max_depth', range(1, 20))}),
    'Logistic Regression': (LogisticRegression(max_iter=1000), {'C': hp.loguniform('C', -4, 4)}),
    'SVM': (SVC(), {'C': hp.loguniform('C', -4, 4), 'kernel': hp.choice('kernel', ['linear', 'rbf'])}),
    'KNN': (KNeighborsClassifier(), {'n_neighbors': hp.choice('n_neighbors', range(1, 31))}),
    'Naive Bayes': (GaussianNB(), {}),
    'Ridge Classifier': (RidgeClassifier(), {'alpha': hp.loguniform('alpha', -4, 4)}),
    'SGD': (SGDClassifier(max_iter=1000), {'alpha': hp.loguniform('alpha', -4, 4)}),
    'MLP': (MLPClassifier(max_iter=1000), {'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [(50,), (100,), (50, 50)])})
}

def hyperopt_train_test(params, clf, X_train, y_train):
    clf.set_params(**params)
    return cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=kf).mean()

def f(params, clf, X_train, y_train):
    acc = hyperopt_train_test(params, clf, X_train, y_train)
    return {'loss': -acc, 'status': STATUS_OK}

# 用于存储最佳参数
best_params = {}

# 进行调参并输出结果
for model_name, (clf, space) in models.items():
    trials = Trials()
    best = fmin(fn=lambda params: f(params, clf, X_train, y_train), space=space, algo=tpe.suggest, max_evals=100, trials=trials)
    best_params[model_name] = best
    print(f'{model_name} best params: {best}')

# 评估最佳模型在验证集上的表现
for model_name, (clf, space) in models.items():
    clf.set_params(**best_params[model_name])
    clf.fit(X_train, y_train)
    val_predict = clf.predict(X_val)
    print(f'{model_name} Validation Accuracy: {accuracy_score(y_val, val_predict)}')
    print(classification_report(y_val, val_predict, digits=4))

# 最终在测试集上评估最佳模型
for model_name, (clf, space) in models.items():
    clf.set_params(**best_params[model_name])
    clf.fit(X_train_val, y_train_val)
    test_predict = clf.predict(X_test)
    print(f'{model_name} Test Accuracy: {accuracy_score(y_test, test_predict)}')
    print(classification_report(y_test, test_predict, digits=4))
