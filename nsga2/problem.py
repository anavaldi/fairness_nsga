from nsga2.individual import Individual
from nsga2.ml import *
import random

class Problem:

    def __init__(self, objectives, num_of_variables, variables_range, expand=True, same_range=False):
        self.num_of_objectives = len(objectives)
        self.num_of_variables = num_of_variables
        self.objectives = objectives
        self.expand = expand
        self.variables_range = []
        if same_range:
            for _ in range(num_of_variables):
                self.variables_range.append(variables_range[0])
        else:
            self.variables_range = variables_range

    def generate_individual(self):
        individual = Individual()
        individual.features = [random.uniform(*x) for x in self.variables_range]
        hyperparameters = ['criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_leaf_nodes', 'min_impurity_decrease', 'class_weight']
        individual.features = dict(zip(hyperparameters, individual.features))
        return individual

    def calculate_objectives(self, individual):
        if self.expand:
            hyperparameters = decode(**individual.features)
            y = evaluate_fairness('adult','sex', **hyperparameters)
            acc = accuracy_diff(y[0], y[1], y[2], y[3])
            dem_fp = dem_fpr(y[0], y[1], y[2], y[3])
            individual.objectives = [acc, dem_fp]
        else:
            hyperparameters = decode(**individual.features)
            y = evaluate_fairness('adult', 'sex', **hyperparameters)
            acc = accuracy_diff(y[0], y[1], y[2], y[3])
            dem_fp = dem_fpr(y[0], y[1], y[3])
            individual.objectives = [acc, dem_fp] 

