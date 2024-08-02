import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import StackingClassifier

# 加载数据
data = np.load('./new_data/dataset_split_0.npz')
X_train = data['X_train']
X_test = data['X_test']
y_train = data['y_train']
y_test = data['y_test']

# 定义最佳模型组合
log_reg = LogisticRegression(C=0.1, max_iter=1000)
mlp = MLPClassifier(hidden_layer_sizes=(430,), max_iter=1000)
ada = AdaBoostClassifier(learning_rate=0.8, n_estimators=60)
gbdt = GradientBoostingClassifier(n_estimators=19)

estimators = [
    ('Logistic Regression', log_reg),
    ('MLP', mlp),
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
    for j in range(3):
        stacking_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

        # 训练模型
        stacking_clf.fit(X_train, y_train)

        # 进行预测
        y_pred = stacking_clf.predict(X_test)

        # 生成混淆矩阵
        cm = confusion_matrix(y_test, y_pred)

        # 将混淆矩阵转换为百分比格式
        conf_matrix = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
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
        # 绘制热力图
        plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')

        # 添加颜色条
        plt.colorbar()

        # 设置x轴和y轴的标签
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        # 设置刻度标签
        tick_marks = np.arange(len(conf_matrix))
        plt.xticks(tick_marks, ['Perfective', 'Adaptive', 'Corrective'])
        plt.yticks(tick_marks, ['Perfective', 'Adaptive', 'Corrective'])

        # 在每个格子内显示数值
        for i in range(len(conf_matrix)):
            for j in range(len(conf_matrix)):
                color = 'white' if i == j else 'black'  # 对角线上的数字设置为白色，其他数字设置为黑色
                plt.text(j, i, "{:.4f}".format(conf_matrix[i, j]), ha='center', va='center', color=color)

        # 显示图形
        plt.show()
        # 输出分类报告
        print(classification_report(y_test, y_pred, digits=4))
