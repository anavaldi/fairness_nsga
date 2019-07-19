from nsga2.problem import Problem
from nsga2.evolution import Evolution
import matplotlib.pyplot as plt
from nsga2.ml import *
import pandas as pd
from math import ceil
from nsga2.plot import *
import random

with open('nsga2/config_file.yaml', 'r') as f:
    config = yaml.load(f)


# problem parameters
generations = 200
individuals = 50
dataset = 'ricci'
variable = 'Race'
set_seed = random.randrange(1000)


# write datasets
X_tr, X_v, X_tst, y_tr, y_v, y_tst = get_matrices(dataset, set_seed)
write_train_val_test(dataset, set_seed, X_tr, X_v, X_tst, y_tr, y_v, y_tst)


# number of rows in train
num_rows_train = get_matrices(dataset, set_seed)[0].shape[0]

# range of hyperparameters
min_range_criterion = 0
max_range_criterion = 1

min_range_max_depth = 3
max_range_max_depth = None

min_range_samples_split = 2
#max_range_samples_split = num_rows_train
max_range_samples_split = 40

min_range_samples_leaf = 1
#max_range_samples_leaf = ceil(0.5*num_rows_train)
max_range_samples_leaf = 60

min_range_leaf_nodes = 2
max_range_leaf_nodes = None

min_range_impurity_decrease = 0.00001
max_range_impurity_decrease = 0.1

min_range_class_weight = 1
max_range_class_weight = 9

#individuals_results = pd.DataFrame()

#problem = Problem(num_of_variables = 7, objectives = [accuracy_inv, dem_fpr], variables_range=[(min_range_criterion, max_range_criterion), (min_range_max_depth, max_range_max_depth), (min_range_samples_split, max_range_samples_split), (min_range_samples_leaf, max_range_samples_leaf), (min_range_leaf_nodes, max_range_leaf_nodes), (min_range_impurity_decrease, max_range_impurity_decrease), (min_range_class_weight, max_range_class_weight)], individuals_df = pd.DataFrame(), num_of_generations = generations, num_of_individuals = individuals, dataset_name = dataset, variable_name = variable)
problem = Problem(num_of_variables = 5, objectives = [gmean_inv, dem_fpr], variables_range = [(min_range_criterion, max_range_criterion), (min_range_max_depth, max_range_max_depth), (min_range_samples_split, max_range_samples_split), (min_range_leaf_nodes, max_range_leaf_nodes), (min_range_class_weight, max_range_class_weight)], individuals_df = pd.DataFrame(), num_of_generations = generations, num_of_individuals = individuals, dataset_name = dataset, variable_name = variable, seed = set_seed)
#problem = Problem(num_of_variables = 5, objectives = [accuracy_inv, dem_fpr], variables_range = [(min_range_criterion, max_range_criterion), (min_range_max_depth, max_range_max_depth), (min_range_samples_leaf, max_range_samples_leaf), (min_range_impurity_decrease, max_range_impurity_decrease), (min_range_class_weight, max_range_class_weight)], individuals_df = pd.DataFrame(), num_of_generations = generations, num_of_individuals = individuals, dataset_name = dataset, variable_name = variable)
evo = Evolution(problem, evolutions_df = pd.DataFrame(), dataset_name = dataset, protected_variable = variable, num_of_generations = generations, num_of_individuals = individuals)
func = [i.objectives for i in evo.evolve()]
print(func)

df = pd.read_csv(config['ROOT_PATH'] + '/results/population/evolution_' + dataset + '_' + variable + '_seed_' + str(set_seed) + '_gen_' + str(generations) + '_indiv_' + str(individuals) + '.csv')
df = df.loc[(df['rank'] == 0) & ((df['generation'] == 1) | (df['generation'] == 20) | (df['generation'] == 40) | (df['generation'] == 60) | (df['generation'] == 80) | (df['generation'] == 100))]

scatterplot(df, set_seed, 'dem_fp', 'error', 'generation', dataset, variable, generations, individuals)


#x = [i[1] for i in func]
#y = [i[0] for i in func]

#plt.xlabel('Demography FPR', fontsize=15)
#plt.ylabel('Error', fontsize=15)
#plt.grid(True)
#plt.scatter(x, y)
#plt.savefig('./results/figures/best_sol_' + dataset + '_' + variable + '_' + str(generations) + '_' + str(individuals) + '.png')
#plt.show()
