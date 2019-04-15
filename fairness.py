from nsga2.problem import Problem
from nsga2.evolution import Evolution
import matplotlib.pyplot as plt
from nsga2.ml import *


min_range = 0
max_range = 1
max_range_depth = 15
min_range_samples_leaf = 0.0000000000000001
max_range_samples_leaf = 0.5
min_range_leaf_nodes = 1.0000000000000001
max_range_leaf_nodes = 1000

results = pd.DataFrame()

problem = Problem(num_of_variables = 7, objectives = [accuracy_inv, dem_fpr], variables_range=[(min_range, max_range), (min_range, max_range_depth), (min_range, max_range), (min_range_samples_leaf, max_range_samples_leaf), (min_range_leaf_nodes, max_range_leaf_nodes), (min_range, max_range), (min_range, max_range)], results_df = results)
evo = Evolution(problem, mutation_param=20)
func = [i.objectives for i in evo.evolve()]
print(func)

#results = pd.read_csv('results.csv')
#x = results['error']
#y = results['dem_fp']

#plt.xlabel('Error', fontsize=15)
#plt.ylabel('Demography FPR', fontsize=15)
#plt.grid(True)
#plt.scatter(x, y)
#plt.show()
