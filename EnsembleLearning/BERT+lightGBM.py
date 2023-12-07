from matplotlib import pyplot as plt
from pyforest import sns
import time
# model_name = 'distilbert-base-uncased'
model_name = 'bert-base-uncased'  # 为了让准确率更低用这个
csn_path = 'Commit_dataset_final_drop_diff.csv'
path = 'Commit_dataset_final.xlsx'

import logging

logging.basicConfig(level=logging.INFO)
import torch
from transformers import AutoTokenizer, AutoModel
from lightgbm import LGBMClassifier

# Load pre-trained model tokenizer (vocabulary)
tokenizer = AutoTokenizer.from_pretrained(model_name)

import pandas as pd
start_time = time.time()
df = pd.read_excel(path)
df.head()
df = df.drop('comment_diff', axis=1)
# Tokenize input
# tokenized_text1 = df[['comment_diff']].apply(
#     (lambda x: tokenizer.encode(x, max_length=80, add_special_tokens=True, truncation=True)))

tokenized_text = df['comment'].apply(
    (lambda x: tokenizer.encode(x, max_length=80, add_special_tokens=True, truncation=True)))

import numpy as np

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
# bert_features = last_hidden_states[0][:, 0, :].numpy()
# print(bert_features.shape)
# bert_features.shape

labels = df['3_labels']
print(labels)

# cc是除了语义特征外的其他特征，共90个，1793行
cc = df.drop(columns=['3_labels', 'comment', 'project', 'commit_id'], axis=1).replace(np.nan, 0)  # 将空的格子设置为0
# cc = cc.iloc[:, 0:46]  # 取前0~45个（46代码更改特征）
# cc = cc.iloc[:, 46:66]  # 取前46~65个（20key words）
# cc = cc.iloc[:, 66:67]  # （1个bug fix特征）
# cc = cc.iloc[:, 67:90]  # （23个重构操作特征）
print("下面输出CC的信息：\n")
print(cc)

print(cc)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
cc = sc.fit_transform(cc)
print(cc)

#all_input_features = np.concatenate((bert_features, cc), axis=1)  # 拼接特征，使用语义+90个特征
all_input_features = cc  # 只使用代码更改特征
#all_input_features = bert_features  # 只使用语义特征
all_input_features[0].shape

from sklearn.preprocessing import LabelBinarizer

# 数据转换成one-hot向量
# encoder = LabelBinarizer()
# labels = encoder.fit_transform(labels)
# print(labels)

from sklearn.model_selection import train_test_split
for i in range(1, 11):
    print("----------EPOCH={}----------------".format(i))
    # stratified train_test_split
    X_train, X_test, y_train, y_test = train_test_split(all_input_features, labels, test_size=0.2, random_state=42,
                                                        stratify=labels)  # 80% training

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125,
                                                      random_state=42)  # 0.125 * 0.8 = 0.10 --> #10% validation and 70% training

    # Training the DNN
    # model = LGBMClassifier(objective='')
    clf = LGBMClassifier()

    clf.fit(X_train, y_train)

    train_predict = clf.predict(X_train)
    test_predict = clf.predict(X_test)

    from sklearn import metrics
    from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, \
        roc_auc_score  # 输出评价指标

    ## 利用accuracy（准确度）【预测正确的样本数目占总预测样本数目的比例】评估模型效果
    print('The accuracy of the Logistic Regression is:', metrics.accuracy_score(y_train, train_predict))
    print('The accuracy of the Logistic Regression is:', metrics.accuracy_score(y_test, test_predict))
    # print('pre=', metrics.precision_score(y_test, test_predict, average='micro'))
    # print('recall=', metrics.recall_score(y_test, test_predict, average='micro'))
    # print('F1=', metrics.f1_score(y_test, test_predict, average='micro'))
    # print(test_predict, y_test)
    print(classification_report(y_test, test_predict, digits=4))

## 查看混淆矩阵 (预测值和真实值的各类情况统计矩阵)
confusion_matrix_result = metrics.confusion_matrix(test_predict, y_test, labels=[0, 1, 2])

print('The confusion matrix result:\n', confusion_matrix_result)

print(classification_report(y_test, test_predict, digits=4))
end_time = time.time()
print("总共花销时间：{}s".format(end_time-start_time))
# 利用热力图对于结果进行可视化
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_result, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()
