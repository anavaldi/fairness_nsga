from nsga2.utils import NSGA2Utils
from nsga2.population import Population
import pandas as pd
from collections import OrderedDict

class Evolution:

    def __init__(self, problem, evolutions_df, dataset_name, protected_variable, num_of_generations=5, num_of_individuals=10, num_of_tour_particips=2, tournament_prob=0.9, crossover_param=2, mutation_param=5, mutation_prob=0.1, beta_method="uniform"):
        self.utils = NSGA2Utils(problem, num_of_individuals, num_of_tour_particips, tournament_prob, crossover_param, mutation_param, mutation_prob, beta_method)
        self.population = None
        self.evolutions_df = evolutions_df
        self.dataset_name = dataset_name
        self.protected_variable = protected_variable
        self.num_of_generations = num_of_generations
        self.on_generation_finished = []
        self.num_of_individuals = num_of_individuals
        self.mutation_prob = mutation_prob
        self.beta_method = beta_method

    def evolve(self):
        self.population = self.utils.create_initial_population()
        self.utils.fast_nondominated_sort(self.population)
        for front in self.population.fronts:
            self.utils.calculate_crowding_distance(front)
        children = self.utils.create_children(self.population)
        returned_population = None
        for i in range(self.num_of_generations):
            for indiv in self.population.population:
                indiv_list = list(indiv.features.items())
                criterion, max_depth, min_samples_split, min_samples_leaf, max_leaf_nodes, min_impurity_decrease, class_weight = [item[1] for item in indiv_list]
                evolutions_aux = pd.DataFrame({'criterion': [criterion], 'max_depth': [max_depth], 'min_samples_split': [min_samples_split], 'min_samples_leaf': [min_samples_leaf], 'max_leaf_nodes': [max_leaf_nodes], 'min_impurity_decrease': [min_impurity_decrease], 'class_weight': [class_weight],  'error': indiv.objectives[0], 'dem_fp': indiv.objectives[1], 'generation': i})
                self.evolutions_df = pd.concat([self.evolutions_df, evolutions_aux])
            if i == (self.num_of_generations-1):
                self.evolutions_df.to_csv("./results/population/evolution_" + self.dataset_name + "_" + self.protected_variable + "_" + str(self.num_of_generations) + "_" + str(self.num_of_individuals) +  ".csv", index = False, header = True, columns = ['error', 'dem_fp', 'generation', 'criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_leaf_nodes', 'min_impurity_decrease', 'class_weight'])
            print("GENERATION:")
            print(i)
            self.population.extend(children)
            self.utils.fast_nondominated_sort(self.population)
            new_population = Population()
            front_num = 0
            while len(new_population) + len(self.population.fronts[front_num]) <= self.num_of_individuals:
                self.utils.calculate_crowding_distance(self.population.fronts[front_num])
                new_population.extend(self.population.fronts[front_num])
                front_num += 1
            self.utils.calculate_crowding_distance(self.population.fronts[front_num])
            [print(ind.features) for ind in new_population.population]
            self.population.fronts[front_num].sort(key=lambda individual: individual.crowding_distance, reverse=True)
            new_population.extend(self.population.fronts[front_num][0:self.num_of_individuals-len(new_population)])
            returned_population = self.population
            self.population = new_population
            #self.utils.fast_nondominated_sort(self.population)
            #for front in self.population.fronts:
               # self.utils.calculate_crowding_distance(front)
            children = self.utils.create_children(self.population)
        return returned_population.fronts[0]
