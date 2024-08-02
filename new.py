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
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
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
tokenizer = AutoTokenizer.from_pretrained(model_name)

df = pd.read_excel(csn_path, header=0, index_col=False)

# Tokenize 'comment' column
tokenized_comment = df['comment'].apply(
    (lambda x: tokenizer.encode(x, max_length=80, add_special_tokens=True, truncation=True)))
model_name = 'distilbert-base-uncased'  # 准确率第二
csn_path = './MuheCC_Dataset_with_refactoring_final.xlsx'

logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = AutoTokenizer.from_pretrained(model_name)

df = pd.read_excel(csn_path, header=0, index_col=False)

# Tokenize 'comment' column
tokenized_comment = df['comment'].apply(
    (lambda x: tokenizer.encode(x, max_length=80, add_special_tokens=True, truncation=True)))

# Tokenize 'refactoring' column with max_length=512
# tokenized_refactoring = df['refactoring'].apply(
#     (lambda x: tokenizer.encode(x, max_length=512, add_special_tokens=True, truncation=True) if pd.notnull(x) else []))
# Tokenize 'refactoring' column with max_length=512
tokenized_refactoring = df['refactoring'].apply(
    (lambda x: tokenizer.encode(str(x), max_length=512, add_special_tokens=True, truncation=True) if pd.notnull(x) else []))

# Pad tokenized comments and refactorings
max_len_comment = max([len(i) for i in tokenized_comment])
max_len_refactoring = max([len(i) for i in tokenized_refactoring])

padded_comment = np.array([i + [0] * (max_len_comment - len(i)) for i in tokenized_comment.values])
padded_refactoring = np.array([i + [0] * (max_len_refactoring - len(i)) for i in tokenized_refactoring.values])

# Create attention masks
attention_mask_comment = np.where(padded_comment != 0, 1, 0)
attention_mask_refactoring = np.where(padded_refactoring != 0, 1, 0)

# Convert to torch tensors
input_ids_comment = torch.tensor(padded_comment)
attention_mask_comment = torch.tensor(attention_mask_comment)

input_ids_refactoring = torch.tensor(padded_refactoring)
attention_mask_refactoring = torch.tensor(attention_mask_refactoring)

# Load pre-trained model
model = AutoModel.from_pretrained(model_name)

with torch.no_grad():
    last_hidden_states_comment = model(input_ids_comment, attention_mask=attention_mask_comment)
    last_hidden_states_refactoring = model(input_ids_refactoring, attention_mask=attention_mask_refactoring)

# Extract BERT features
bert_features_comment = last_hidden_states_comment[0][:, 0, :].numpy()
bert_features_refactoring = last_hidden_states_refactoring[0][:, 0, :].numpy()

# Concatenate BERT features
bert_features = np.concatenate((bert_features_comment, bert_features_refactoring), axis=1)

# Process other features
change = {'p': 0, 'a': 1, 'c': 2}  # 替换的值
labels = df['3_labels'].map(change).values  # 标签值
cc = df.drop(columns=['3_labels', 'comment', 'project', 'commit_id', 'refactoring'], axis=1).replace(np.nan, 0)  # 将空的格子设置为0

# Check for NaN values
print(cc.isna().sum())

# Ensure there are no NaN values
cc = cc.fillna(0)

sc = StandardScaler()
cc = sc.fit_transform(cc)

# Concatenate all features
all_input_features = np.concatenate((bert_features, cc), axis=1)  # 拼接语义和特征
print(all_input_features[0].shape)
# Tokenize 'refactoring' column with max_length=512
# tokenized_refactoring = df['refactoring'].apply(
#     (lambda x: tokenizer.encode(x, max_length=512, add_special_tokens=True, truncation=True) if pd.notnull(x) else []))
# Tokenize 'refactoring' column with max_length=512
tokenized_refactoring = df['refactoring'].apply(
    (lambda x: tokenizer.encode(str(x), max_length=512, add_special_tokens=True, truncation=True) if pd.notnull(x) else []))

# Pad tokenized comments and refactorings
max_len_comment = max([len(i) for i in tokenized_comment])
max_len_refactoring = max([len(i) for i in tokenized_refactoring])

padded_comment = np.array([i + [0] * (max_len_comment - len(i)) for i in tokenized_comment.values])
padded_refactoring = np.array([i + [0] * (max_len_refactoring - len(i)) for i in tokenized_refactoring.values])

# Create attention masks
attention_mask_comment = np.where(padded_comment != 0, 1, 0)
attention_mask_refactoring = np.where(padded_refactoring != 0, 1, 0)

# Convert to torch tensors
input_ids_comment = torch.tensor(padded_comment)
attention_mask_comment = torch.tensor(attention_mask_comment)

input_ids_refactoring = torch.tensor(padded_refactoring)
attention_mask_refactoring = torch.tensor(attention_mask_refactoring)

# Load pre-trained model
model = AutoModel.from_pretrained(model_name)

with torch.no_grad():
    last_hidden_states_comment = model(input_ids_comment, attention_mask=attention_mask_comment)
    last_hidden_states_refactoring = model(input_ids_refactoring, attention_mask=attention_mask_refactoring)

# Extract BERT features
bert_features_comment = last_hidden_states_comment[0][:, 0, :].numpy()
bert_features_refactoring = last_hidden_states_refactoring[0][:, 0, :].numpy()

# Concatenate BERT features
bert_features = np.concatenate((bert_features_comment, bert_features_refactoring), axis=1)

# Process other features
change = {'p': 0, 'a': 1, 'c': 2}  # 替换的值
labels = df['3_labels'].map(change).values  # 标签值
cc = df.drop(columns=['3_labels', 'comment', 'project', 'commit_id', 'refactoring'], axis=1).replace(np.nan, 0)  # 将空的格子设置为0

# Check for NaN values
print(cc.isna().sum())

# Ensure there are no NaN values
cc = cc.fillna(0)

sc = StandardScaler()
cc = sc.fit_transform(cc)

# Concatenate all features
all_input_features = np.concatenate((bert_features, cc), axis=1)  # 拼接语义和特征
print(all_input_features[0].shape)


# 用于交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define models and hyperparameter spaces
models = {
    'Decision Tree': (DecisionTreeClassifier(), {'max_depth': hp.choice('max_depth', range(1, 1001)),
                                                 'min_samples_leaf': hp.uniform('min_samples_leaf', 0, 0.5),
                                                 'min_samples_split': hp.uniform('min_samples_split', 0, 1)}),
    'Logistic Regression': (LogisticRegression(max_iter=1000), {'C': hp.choice('C', [0.001, 0.01, 0.1, 1, 5, 10, 100])}),
    'SVM': (SVC(), {'C': hp.choice('C', [round(0.05*i, 2) for i in range(2, 41)])}),
    'KNN': (KNeighborsClassifier(), {'n_neighbors': hp.choice('n_neighbors', range(2, 31)),
                                     'leaf_size': hp.choice('leaf_size', [10, 20, 30, 40, 50])}),
    'Naive Bayes': (GaussianNB(), {'var_smoothing': hp.choice('var_smoothing', [1, 0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9])}),
    'Ridge Classifier': (RidgeClassifier(), {'alpha': hp.choice('alpha', [round(0.15*i, 2) for i in range(14)])}),
    'SGD': (SGDClassifier(max_iter=1000), {'l1_ratio': hp.choice('l1_ratio', [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]),
                                           'alpha': hp.choice('alpha', [0.0001, 0.001, 0.01, 0.1])}),
    'MLP': (MLPClassifier(max_iter=1000), {'hidden_layer_sizes': hp.choice('hidden_layer_sizes', [tuple([i]) for i in range(10, 1001, 10)])}),
    'Random Forest': (RandomForestClassifier(), {'max_depth': hp.choice('max_depth', range(5, 201, 5)),
                                                 'n_estimators': hp.choice('n_estimators', range(5, 201, 5))}),
    'AdaBoost': (AdaBoostClassifier(), {'learning_rate': hp.choice('learning_rate', [round(0.05*i, 2) for i in range(1, 21)]),
                                        'n_estimators': hp.choice('n_estimators', range(5, 251, 5))}),
    'GBDT': (GradientBoostingClassifier(), {'n_estimators': hp.choice('n_estimators', range(10, 21)),
                                            'learning_rate': hp.choice('learning_rate', [round(0.05*i, 2) for i in range(1, 21)])}),
    'lightGBM': (LGBMClassifier(), {'num_leaves': hp.choice('num_leaves', range(5, 201, 5)),
                                    'learning_rate': hp.choice('learning_rate', [round(0.05*i, 2) for i in range(1, 21)]),
                                    'n_estimators': hp.choice('n_estimators', range(5, 251, 5))})
}
all_possible_params = {
    'Decision Tree': {'max_depth': range(1, 1001),
                      'min_samples_leaf': [i/100.0 for i in range(1, 50)],
                      'min_samples_split': [i/100.0 for i in range(1, 100)]},
    'Logistic Regression': {'C': [0.001, 0.01, 0.1, 1, 5, 10, 100]},
    'SVM': {'C': [round(0.05*i, 2) for i in range(2, 41)]},
    'KNN': {'n_neighbors': range(2, 31),
            'leaf_size': [10, 20, 30, 40, 50]},
    'Naive Bayes': {'var_smoothing': [1, 0.1, 0.01, 0.001, 0.0001, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]},
    'Ridge Classifier': {'alpha': [round(0.15*i, 2) for i in range(14)]},
    'SGD': {'l1_ratio': [round(0.05*i, 2) for i in range(2, 9)],
            'alpha': [0.0001, 0.001, 0.01, 0.1]},
    'MLP': {'hidden_layer_sizes': [tuple([i]) for i in range(10, 1001, 10)]},
    'Random Forest': {'max_depth': range(5, 201, 5),
                      'n_estimators': range(5, 201, 5)},
    'AdaBoost': {'learning_rate': [round(0.05*i, 2) for i in range(1, 21)],
                 'n_estimators': range(5, 251, 5)},
    'GBDT': {'n_estimators': range(10, 21),
             'learning_rate': [round(0.05*i, 2) for i in range(1, 21)]},
    'lightGBM': {'num_leaves': range(5, 201, 5),
                 'learning_rate': [round(0.05*i, 2) for i in range(1, 21)],
                 'n_estimators': range(5, 251, 5)}
}


# def hyperopt_train_test(params, clf, X_train, y_train):
#     clf.set_params(**params)
#     return cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=kf).mean()
#
# def f(params, clf, X_train, y_train):
#     acc = hyperopt_train_test(params, clf, X_train, y_train)
#     return {'loss': -acc, 'status': STATUS_OK}
# 实现训练函数
def train_fn(params, data):
    clf = data['clf'].set_params(**params)
    X_train, y_train = data['X_train'], data['y_train']
    score = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=data['kf']).mean()
    return {'entry': {'F1': score}, 'model': clf}

# 用于存储最佳参数
best_params = {}
# 划分传统模型和集成模型
traditional_models = ['Decision Tree', 'Logistic Regression', 'SVM', 'KNN', 'Naive Bayes', 'Ridge Classifier', 'SGD',
                      'MLP']
ensemble_models = ['Random Forest', 'AdaBoost', 'GBDT', 'lightGBM']

for repeat in range(20):
    # 打乱数据并划分训练、验证和测试集
    X_train, X_test, y_train, y_test = train_test_split(all_input_features, labels, test_size=0.2,
                                                                random_state=42 + repeat, stratify=labels)
    # X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.125,
    #                                                   random_state=42 + repeat, stratify=y_train_val)
    # early_stopping = EarlyStopping(patience=100, min_delta=1e-4)

    # 保存每次划分的数据集
    np.savez(f'./new_data2/dataset_split_{repeat}.npz', X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    # 读取数据
    # data = np.load(f'dataset_split_{repeat}.npz')
    # X_train = data['X_train']
    # X_test = data['X_test']
    # y_train = data['y_train']
    # y_test = data['y_test']

    # 进行调参并输出结果
    for model_name, (clf, space) in models.items():
        data = {
            'clf': clf,
            'X_train': X_train,
            'y_train': y_train,
            'kf': kf
        }
        best_p, best_model, entry = generate(all_possible_params[model_name], train_fn, data)
        best_params[model_name] = best_p
        print(f'{model_name} best params: {best_params}')
        print(f'{model_name} 最佳参数: {best_params}')

    # 存储最佳模型和其得分
    best_model = None
    best_score = 0

    # 进行组合
    for trad_combo in combinations(traditional_models, 2):
        for ens_combo in combinations(ensemble_models, 2):
            # 获取模型
            trad_clfs = [(name, models[name][0].set_params(**best_params[name])) for name in trad_combo]
            ens_clfs = [(name, models[name][0].set_params(**best_params[name])) for name in ens_combo]

            # 构建StackingClassifier
            estimators = trad_clfs + ens_clfs
            stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

            stacking_clf.fit(X_train, y_train)
            test_predict = stacking_clf.predict(X_test)
            score = accuracy_score(y_test, test_predict)
            report = classification_report(y_test, test_predict, digits=4)
            if score > best_score:
                best_score = score
                best_combination = estimators
                best_report = report
            # 输出当前最佳组合的效果
            print(f'当前最佳组合: {best_combination}')
            print(f'当前最佳Test Accuracy: {best_score}')
            print(classification_report(y_test, test_predict, digits=4))
            print(report)

    # 输出最佳组合的效果
    print(f'Best Combination: {best_combination}')
    print(f'Test Accuracy: {best_score}')
    print(classification_report(y_test, test_predict, digits=4))
    print(best_report)

