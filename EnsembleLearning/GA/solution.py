"""Class that represents the solution to be evolved.如何进化"""
import random
from sklearn.model_selection import cross_val_score, train_test_split, KFold
class Solution():
    def __init__(self, all_possible_params):#初始化数据
        self.entry = {}
        self.score = 0.
        self.all_possible_params = all_possible_params
        self.params = {}  #  represents model parameters to be picked by creat_random method随机选取参数构成模型
        self.model = None
        
    """Create the model random params.创建模型随机参数"""
    def create_random(self):
        for key in self.all_possible_params:
            self.params[key] = random.choice(self.all_possible_params[key])

    def set_params(self, params):
        self.params = params
      
    """
        Train the model and record the score.
    """
    # def train_model(self, fn_train,params_fn):#训练模型
    #
    #     if self.score == 0.:
    #             res = fn_train(self.params,params_fn)
    #             self.score =  res["entry"]["F1"] #1-float(res["validation_loss"])
    #             # self.score = 1-float(res["validation_loss"])
    #             self.model = res["model"]
    #             self.entry = res['entry']
    def train_model(params, data):
        clf = data['clf'].set_params(**params)
        X_train, y_train = data['X_train'], data['y_train']
        score = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=data['kf']).mean()
        return {'entry': {'F1': score}, 'model': clf}
            
    """Print out a network."""
    def print_solution(self):
        print("for params ", self.params , "the score in the train = ",self.score)