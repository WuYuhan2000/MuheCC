#from GA.optimizer import Optimizer
#from optimizer import Optimizer
#from .optimizer import Optimizer
# from optimizer import Optimizer
from EnsembleLearning.GA.optimizer import Optimizer
from tqdm import tqdm
import threading

# import Utils
def train_sol_thread(solution,fn_train,params_fn,i):
    solution.train_model(fn_train,params_fn)
    print("solution ", i," trained")
    
def train_population(pop, fn_train,params_fn): #训练
    pbar = tqdm(total=len(pop))
    threads = list()
    i=1
    for solution in pop:
        x = threading.Thread(target=train_sol_thread, args=(solution,fn_train,params_fn,i))#第一个参数是线程函数变量，第二个参数args是一个数组变量参数
        i=i+1
        threads.append(x)
        x.start()
        pbar.update(1)#更新进度条长度
        
    for index, thread in enumerate(threads):
        thread.join()
    pbar.close()#关闭占用资源


def get_average_score(pop):
    """Get the average score for a group of solutions.获得一代的平均分数"""
    total_scores = 0
    for solution in pop:
        total_scores += solution.score
    return total_scores / len(pop)

"""Generate the optimal params with the genetic algorithm."""
""" Args:
        GA_params: Params for GA
        all_possible_params (dict): Parameter choices for the model
        train_set : training dataset
        fn_train : a function used to compute the prediction accuracy  计算预测精度的函数
"""
def generate(all_possible_params, fn_train , params_fn):   #形参分别是所有可能参数，模型，数据
   
    GA_params = {
            "population_size": 2,#种群大小
            "max_generations": 2,#最大迭代代数？
            "retain": 0.7,  #强者的定义概率，前百分之70
            "random_select":0.1,#弱者的存活概率
            "mutate_chance":0.1#变异率
            }
    
    print("params of GA" , GA_params)
    optimizer = Optimizer(GA_params ,all_possible_params)
    pop = optimizer.create_population(GA_params['population_size'])#创建种群
    # Evolve the generation.进化
    for i in range(GA_params['max_generations']):
        print("*********************************** REP(GA) ",(i+1))
        # Train and get accuracy for solutions.培训并获得解决方案的准确性
        train_population(pop,fn_train,params_fn)
        # Get the average accuracy for this generation.获得这一代人的平均精度
        average_accuracy = get_average_score(pop)
        # Print out the average accuracy each generation.打印出每一代的平均精度
        print("Generation average: %.2f%%" % (average_accuracy * 100))
        # Evolve, except on the last iteration.进化（除了最后一次迭代）
        if i != (GA_params['max_generations']):
            print("Generation evolving..")
            evolved = optimizer.evolve(pop) #生成进化的一代
            if(len(evolved)!=0):
                pop=evolved
        else:
            pop = sorted(pop, key=lambda x: x.score, reverse=True)
    # Print out the top 2 solutions.输出最好的两个
    size = len(pop)
    if size < 3:
        print_pop(pop[:size])
    else:
        print_pop(pop[:3])
    return pop[0].params ,pop[0].model,pop[0].entry

def print_pop(pop):
    for solution in pop:
        solution.print_solution()    
