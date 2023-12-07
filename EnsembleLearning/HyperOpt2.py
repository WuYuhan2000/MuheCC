# -*- coding: utf-8 -*-
"""
@Time ： 2023/2/24 12:08
@Auth ： 兰宏富
@File ：HyperOpt2.py
@IDE ：PyCharm

hp.choice(label, options) 其中options应是 python 列表或元组。

hp.normal(label, mu, sigma) 其中mu和sigma分别是均值和标准差。

hp.uniform(label, low, high) 其中low和high是范围的下限和上限。

"""
from hyperopt import fmin, tpe, hp, Trials
from matplotlib import pyplot as plt

trials = Trials()
best = fmin(
    fn=lambda x: (x-1)**2,  # 需要最小化的函数
    space=hp.uniform('x', -100, 100),  # 步长
    algo=tpe.suggest,
    max_evals=2000,
    trials=trials,
    verbose=5)  # 最大评估次数

print(best)
f, ax = plt.subplots(1)
xs = [t['tid'] for t in trials.trials]
ys = [t['misc']['vals']['x'] for t in trials.trials]
ax.set_xlim(xs[0]-10, xs[-1]+10)
ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
ax.set_title('$x$ $vs$ $t$ ', fontsize=18)
ax.set_xlabel('$t$', fontsize=16)
ax.set_ylabel('$x$', fontsize=16)
plt.show()

