from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
from sklearn.model_selection import cross_val_score, train_test_split, KFold
import torch
from transformers import AutoTokenizer, AutoModel
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize, scale
from itertools import combinations
from EnsembleLearning.GA.optimizer import Optimizer
from EnsembleLearning.GA.GARunner import generate
from EnsembleLearning.GA.solution import Solution  # 导入 Solution 类
import sys
import os

sys.path.append(os.path.dirname(__file__))
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#判断是否停止的类
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_loss):
        if self.best_loss is None:
            self.best_loss = current_loss
        elif current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# # 使用示例
# early_stopping = EarlyStopping(patience=100, min_delta=1e-4)

# --------------------------------------数据集读取我的版本start------------------------------------------
# model_name = 'huggingface/CodeBERTa-small-v1'  # 准确率最高
model_name = 'distilbert-base-uncased'  # 准确率第二
csn_path = './MuheCC_Dataset_with_refactoring_final.xlsx'

logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

df = pd.read_excel(csn_path, header=0, index_col=False)
#去掉comment和diff
# # Process other features
change = {'p': 0, 'a': 1, 'c': 2}  # 替换的值
labels = df['3_labels'].map(change).values  # 标签值
cc = df.drop(columns=['3_labels', 'comment', 'project', 'commit_id', 'refactoring'], axis=1).replace(np.nan, 0)  # 将空的格子设置为0

# Check for NaN values
print(cc.isna().sum())

# Ensure there are no NaN values
cc = cc.fillna(0)

sc = StandardScaler()
cc = sc.fit_transform(cc)

# Since we are removing the comment and refactoring features, we directly use cc as the final features
all_input_features = cc
print(all_input_features[0].shape)

# 加载数据
# data = np.load('./new_data/dataset_split_0.npz')
# X_train = data['X_train']
# X_test = data['X_test']
# y_train = data['y_train']
# y_test = data['y_test']
# X_train, X_test, y_train, y_test = train_test_split(all_input_features, labels, test_size=0.2,
#                                                     random_state=42, stratify=labels)
# 定义最佳模型组合
# log_reg = LogisticRegression(C=0.1, max_iter=1000)
# mlp = MLPClassifier(hidden_layer_sizes=(430,), max_iter=1000)
ada = AdaBoostClassifier(learning_rate=0.8, n_estimators=60)
gbdt = GradientBoostingClassifier(n_estimators=19)
#默认参数
# log_reg = LogisticRegression(C=1, max_iter=1000)
# mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
# ada = AdaBoostClassifier(learning_rate=1.0, n_estimators=50)
# gbdt = GradientBoostingClassifier(n_estimators=100)

estimators = [
    # ('Logistic Regression', log_reg),
    # ('MLP', mlp)
    ('AdaBoost', ada),
    ('GBDT', gbdt)
]
for i in range(20):
    # 构建StackingClassifier
    # 加载数据
    data = np.load(f'./new_data/dataset_split_{i}.npz')
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

    # 训练模型
    stacking_clf.fit(X_train, y_train)

    # 进行预测
    y_pred = stacking_clf.predict(X_test)

    # # 生成混淆矩阵
    # cm = confusion_matrix(y_test, y_pred)
    #
    # # 将混淆矩阵转换为百分比格式
    # cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # # 类别标签
    # labels = ['Perfective', 'Adaptive', 'Corrective']
    #
    # # 绘制热力图
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm_percentage, annot=True, fmt='.4f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    # plt.xlabel('Predicted Label')
    # plt.ylabel('True Label')
    # plt.title('Confusion Matrix Heatmap')
    # plt.show()

    # 输出分类报告
    print(classification_report(y_test, y_pred, digits=4))
