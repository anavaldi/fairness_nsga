from nsga2.problem import Problem
from nsga2.evolution import Evolution
import matplotlib.pyplot as plt
from nsga2.ml import *


min_range = 0
max_range = 1
max_range_depth = 15
min_range_samples_leaf = 0.0000000000000001
max_range_samples_leaf = 0.5
max_leaf_nodes = 1000


problem = Problem(num_of_variables = 7, objectives = [accuracy_diff, dem_fpr], variables_range=[(min_range, max_range), (min_range, max_range_depth), (min_range, max_range), (min_range_samples_leaf, max_range_samples_leaf), (min_range, max_leaf_nodes), (min_range, max_range), (min_range, max_range)])
evo = Evolution(problem, mutation_param=20)
func = [i.objective for i in evo.evolve()]

function1 = [i[0] for i in func]
function2 = [i[1] for i in func]

plt.xlabel('Accuracy Difference', fontsize=15)
plt.ylabel('Demography FPR', fontsize=15)
plst.scatter(function1, function2)
plt.show()
