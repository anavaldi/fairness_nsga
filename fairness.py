from nsga2.problem import Problem
from nsga2.evolution import Evolution
import matplotlib.pyplot as plt
from nsga2.ml import *
import pandas as pd
from math import ceil


# problem parameters
generations = 100
individuals = 50
dataset = 'german'
variable = 'age'

# number of rows in train
num_rows_train = get_matrices('german')[0].shape[0]

# range of hyperparameters
min_range_criterion = 0
max_range_criterion = 1

min_range_max_depth = 3
max_range_max_depth = 15

min_range_samples_split = 2
max_range_samples_split = num_rows_train

min_range_samples_leaf = 1e-7
max_range_samples_leaf = ceil(0.5*num_rows_train)

min_range_leaf_nodes = 2
max_range_leaf_nodes = 1000

min_range_impurity_decrease = 1e-7
max_range_impurity_decrease = 1

min_range_class_weight = 0
max_range_class_weight = 1

#individuals_results = pd.DataFrame()

problem = Problem(num_of_variables = 7, objectives = [accuracy_inv, dem_fpr], variables_range=[(min_range_criterion, max_range_criterion), (min_range_max_depth, max_range_max_depth), (min_range_samples_split, max_range_samples_split), (min_range_samples_leaf, max_range_samples_leaf), (min_range_leaf_nodes, max_range_leaf_nodes), (min_range_impurity_decrease, max_range_impurity_decrease), (min_range_class_weight, max_range_class_weight)], individuals_df = pd.DataFrame(), num_of_generations = generations, num_of_individuals = individuals, dataset_name = dataset, variable_name = variable)
evo = Evolution(problem, evolutions_df = pd.DataFrame(), dataset_name = dataset, protected_variable = variable, num_of_generations = generations, num_of_individuals = individuals)
func = [i.objectives for i in evo.evolve()]
print(func)

x = [i[0] for i in func]
y = [i[1] for i in func]

plt.xlabel('Error', fontsize=15)
plt.ylabel('Demography FPR', fontsize=15)
plt.grid(True)
plt.scatter(x, y)
plt.savefig('./results/figures/best_sol_' + dataset + '_' + variable + '_' + str(generations) + '_' + str(individuals) + '.png')
plt.show()
