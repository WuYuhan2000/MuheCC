"""
Class that holds a genetic algorithm for evolving a population of params.该类包含用于进化参数群体的遗传算法
"""
from functools import reduce
from operator import add
import random
#from GA.solution import Solution
#from solution import Solution
#from .solution import Solution
# import solution as SL
# from solution import Solution
# import solution as SL
import EnsembleLearning.GA.solution as SL
"""Class that implements genetic algorithm for Hyper-parameter tuning该类实现了用于超参数优化的遗传算法"""
class Optimizer():
    
    def __init__(self, GA_params, all_possible_params):
        """Create an optimizer."""
        self.random_select = GA_params["random_select"]
        self.mutate_chance = GA_params["mutate_chance"]
        self.retain = GA_params["retain"]
        self.all_possible_params = all_possible_params
    
    def create_population(self, count):      #随机创建种群
        """Create a population of random solutions."""
        pop = []
        for _ in range(0, count):
            # Create a random solution.
            solution = SL.Solution(self.all_possible_params)
            # solution = Solution(self.all_possible_params)
            solution.create_random()#创建模型随机参数（一个个体）
            # Add the solution to our population.把这个个体加入种群
            pop.append(solution)
        return pop

    @staticmethod
    def fitness(solution):
        """Return the score, which is our fitness function.返回分数"""
        return solution.score

    def grade(self, pop):
        """Find average fitness for a population. """
        summed = reduce(add, (self.fitness(solution) for solution in pop))
        return summed / float((len(pop)))

    def crossover(self, mother, father):
        """Make two children as parts of their parents.
        Args:
            mother (dict): parameters
            father (dict): parameters
        Returns:
            (list): combined params
        """
        children = []
        for _ in range(2):
            child = {}
            # Loop through the parameters and pick params for the kid.
            for param in self.all_possible_params:
                child[param] = random.choice([mother.params[param], father.params[param]] )

            solution = SL.Solution(self.all_possible_params)
            # solution = Solution(self.all_possible_params)
            solution.set_params(child)#给个体设置孩子的参数
            # Randomly mutate some of the children.随机变异部分孩子，根据前面定义的变异率
            if self.mutate_chance > random.random():
                solution = self.mutate(solution)
            children.append(solution)
        return children
    
    
    def mutate(self, solution):
        """Randomly mutate one part of the solution."""
        # Choose a random key.
        mutation = random.choice(list(self.all_possible_params.keys()))
        # Mutate one of the params.
        solution.params[mutation] = random.choice(self.all_possible_params[mutation])
        return solution
    
    """Evolve a population of solutions."""
    def evolve(self, pop):
        #Get scores for each solution.获取每个个体的分数
        graded = [(self.fitness(solution), solution) for solution in pop]
        #"Sort on the scores.  #给分数排序
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]#x:x[0]代表字典的键（key）给sort排序，reverse为TRUE表示降序
        #Get the number we want to keep for the next gen. #获得我们想要保留到下一代的，前面规定的70%
        retain_length = int(len(graded)*self.retain)
        # define what we want to keep.定义想要保留的
        parents = graded[:retain_length]
        # For those we aren't keeping, randomly keep some anyway.选择适应性不强，但是幸存的
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)
        # Now find out how many spots we have left to fill.看看选出来之后还有多少需要填上
        parents_length = len(parents)
        desired_length = len(pop) - parents_length#后面生成子代的个数要小于等于这个数，为了维持种群稳定
        
        # Add children, which are bred from two remaining solutions.#加入从保留双亲培育出的子代
        if parents_length > 1 and desired_length> 0:
            children = []
            while len(children) < desired_length:
                if parents_length==2:#双亲就两个的情况
                    male_index = 1
                    female_index = 0
                else:
                    male_index = random.randint(0, parents_length-1)
                    female_index = random.randint(0, parents_length-1)
                
                # Assuming they aren't the same solutions...假设他们不是相同的父代
                if male_index != female_index:
                    print("Get a random mom and dad.")
                    male = parents[male_index]
                    female = parents[female_index]
                    # crossover them.交叉繁殖
                    babies = self.crossover(male, female)
                    # Add the children one at a time.
                    for baby in babies:
                        # Don't grow larger than desired length.
                        if len(children) < desired_length:
                            children.append(baby)
            parents.extend(children)
        return parents