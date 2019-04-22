from nsga2.individual import Individual
from nsga2.ml import *
import random
from collections import OrderedDict as od

class Problem:

    def __init__(self, objectives, num_of_variables, variables_range, results_df, expand=True, same_range=False):
        self.num_of_objectives = len(objectives)
        self.num_of_variables = num_of_variables
        self.objectives = objectives
        self.results_df = results_df
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
        #print(individual.features)
        hyperparameters = ['criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_leaf_nodes', 'min_impurity_decrease', 'class_weight']
        individual.features = od(zip(hyperparameters, individual.features))
        return individual

    def calculate_objectives(self, individual, results_df):
        #print(individual.features)
        if self.expand:
            hyperparameters = decode(**individual.features)
            X, y, pred = evaluate('german', 'age', **hyperparameters)
            y_fair = evaluate_fairness(X, y, pred)
            error = accuracy_inv(y, pred)
            dem_fp = dem_fpr(y_fair[0], y_fair[1], y_fair[2], y_fair[3])
            individual.objectives = [error, dem_fp]
            results_aux = pd.DataFrame({'individual': [hyperparameters], 'error': error, 'dem_fp': dem_fp})
            results_df = results_df.append(results_aux, ignore_index = True)
            results_df.to_csv("./results/results_prueba.csv", index = False, mode='a', header=False)
        else:
            hyperparameters = decode(**individual.features)
            X, y, pred = evaluate('german', 'age', **hyperparameters)
            y_fair = evaluate_fairness(X, y, pred)
            error = accuracy_inv(y, pred)
            dem_fp = dem_fpr(y_fair[0], y_fair[1], y_fair[3])
            individual.objectives = [error, dem_fp]
            results_aux = pd.DataFrame({'individual': [hyperparameters], 'error': error, 'dem_fp': dem_fp})
            results_df = results_df.append(results_aux, ignore_index = True)
            results_df.to_csv("results.csv", index=False, mode='a', header=False)
