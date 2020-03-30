from nsga2.problem import Problem
from nsga2.evolution import Evolution
import matplotlib.pyplot as plt
from nsga2.ml import *
import pandas as pd
from math import ceil
from nsga2.plot import *
import random

import warnings
warnings.filterwarnings("ignore")

with open('nsga2/config_file.yaml','r') as f:
    config = yaml.load(f)
    
# problem parameters
generations = 300
individuals = 50
dataset = 'adult'; variable = 'race'
#dataset = 'german'; variable = 'age'
#dataset = 'propublica_recidivism'; variable = 'race'
#dataset = 'propublica_violent_recidivism'; variable = 'race'
#dataset = 'ricci'; variable = 'Race'

set_seed_base = 100
n_runs = 10

for run in range(n_runs):
    set_seed = set_seed_base + run

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
    
    problem = Problem(num_of_variables = 5, objectives = [gmean_inv, dem_fpr], variables_range = [(min_range_criterion, max_range_criterion), (min_range_max_depth, max_range_max_depth), (min_range_samples_split, max_range_samples_split), (min_range_leaf_nodes, max_range_leaf_nodes), (min_range_class_weight, max_range_class_weight)], individuals_df = pd.DataFrame(), num_of_generations = generations, num_of_individuals = individuals, dataset_name = dataset, variable_name = variable, seed = set_seed)
    
    print("------------RUN:",run)
    
    evo = Evolution(problem, evolutions_df = pd.DataFrame(), dataset_name = dataset, protected_variable = variable, num_of_generations = generations, num_of_individuals = individuals)
    pareto = evo.evolve()
    
    first = True
    for p in pareto:    
        problem.test_and_save(p,first,problem.seed)
        first = False