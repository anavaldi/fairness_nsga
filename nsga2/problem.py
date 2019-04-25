from nsga2.individual import Individual
from nsga2.ml import *
import random
from collections import OrderedDict as od
from collections import Counter

class Problem:

    def __init__(self, objectives, num_of_variables, variables_range, individuals_df, num_of_generations, num_of_individuals, dataset_name, variable_name, expand=True, same_range=False):
        self.num_of_objectives = len(objectives)
        self.num_of_variables = num_of_variables
        self.objectives = objectives
        self.expand = expand
        self.variables_range = []
        self.individuals_df = individuals_df
        self.num_of_generations = num_of_generations
        self.num_of_individuals = num_of_individuals
        self.dataset_name = dataset_name
        self.variable_name = variable_name
        if same_range:
            for _ in range(num_of_variables):
                self.variables_range.append(variables_range[0])
        else:
            self.variables_range = variables_range

    def generate_default_individual_gini(self):
        individual = Individual()
        individual.features = [0, None, 2, 5, None, 1e-7, None]
        hyperparameters = ['criterion','max_depth', 'min_samples_split', 'min_samples_leaf', 'max_leaf_nodes', 'min_impurity_decrease', 'class_weight']
        individual.features = od(zip(hyperparameters, individual.features))
        individual.features = decode(self.variables_range, **individual.features)
        return individual

    def generate_default_individual_entropy(self):
        individual = Individual()
        individual.features = [1, None, 2, 5, None, 1e-7, None]
        hyperparameters = ['criterion','max_depth', 'min_samples_split', 'min_samples_leaf', 'max_leaf_nodes', 'min_impurity_decrease', 'class_weight']
        individual.features = od(zip(hyperparameters, individual.features))
        individual.features = decode(self.variables_range, **individual.features)
        return individual

    def generate_individual(self):
        individual = Individual()
        individual.features = [random.uniform(*x) for x in self.variables_range]
        hyperparameters = ['criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_leaf_nodes', 'min_impurity_decrease', 'class_weight']
        individual.features = od(zip(hyperparameters, individual.features))
        individual.features = decode(self.variables_range, **individual.features)
        return individual

    def calculate_objectives(self, individual):
        if self.expand:
            #hyperparameters = decode(self.variables_range, **individual.features)
            hyperparameters = individual.features
            X, y, pred = evaluate(self.dataset_name, self.variable_name, **hyperparameters)
            y_fair = evaluate_fairness(X, y, pred)
            error = accuracy_inv(y, pred)
            dem_fp = dem_fpr(y_fair[0], y_fair[1], y_fair[2], y_fair[3])
            individual.objectives = [error, dem_fp]
            indiv_list = list(individual.features.items())
            criterion, max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, min_impurity_decrease, class_weight = [item[1] for item in indiv_list]
            individuals_aux = pd.DataFrame({'criterion': [criterion], 'max_depth': [max_depth], 'min_samples_split': [min_samples_split], 'min_samples_leaf': [min_samples_leaf], 'max_leaf_nodes': [max_leaf_nodes], 'min_impurity_decrease': [min_impurity_decrease], 'class_weight': [class_weight], 'error': error, 'dem_fp': dem_fp})
            self.individuals_df = pd.concat([self.individuals_df, individuals_aux])
            self.individuals_df.to_csv('./results/individuals/individuals_' + self.dataset_name + '_' + self.variable_name + '_' + str(self.num_of_generations) + '_' + str(self.num_of_individuals) + '.csv', index = False, header = True, columns = ['error', 'dem_fp', 'criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_leaf_nodes', 'min_impurity_decrease', 'class_weight'])
        else:
            hyperparameters = decode(**individual.features)
            X, y, pred = evaluate('german', 'age', **hyperparameters)
            y_fair = evaluate_fairness(X, y, pred)
            error = accuracy_inv(y, pred)
            dem_fp = dem_fpr(y_fair[0], y_fair[1], y_fair[3])
            individual.objectives = [error, dem_fp]
            #results_aux = pd.DataFrame({'individual': [hyperparameters], 'error': error, 'dem_fp': dem_fp})
            #results_df = results_df.append(results_aux, ignore_index = True)
            #results_df.to_csv("results.csv", index=False, mode='a', header=False)
