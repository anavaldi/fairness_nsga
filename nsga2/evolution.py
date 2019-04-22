from nsga2.utils import NSGA2Utils
from nsga2.population import Population

class Evolution:

    def __init__(self, problem, num_of_generations=5, num_of_individuals=10, num_of_tour_particips=2, tournament_prob=0.9, crossover_param=2, mutation_param=5, mutation_prob=0.1, beta_method="uniform"):
        self.utils = NSGA2Utils(problem, num_of_individuals, num_of_tour_particips, tournament_prob, crossover_param, mutation_param, mutation_prob, beta_method)
        self.population = None
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
            print("GENERATION:")
            print(i)
            print("FEATURES:")
            [print(ind.features) for ind in self.population.population]
            print("OBJECTIVES:")
            [print(ind.objectives) for ind in self.population.population]
            self.population.extend(children)
            self.utils.fast_nondominated_sort(self.population)
            new_population = Population()
            print("NEW POPULATION:")
            [print(ind.features) for ind in new_population.population]
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
            print("FRONT 0:")
            print([i.features for i in returned_population.fronts[0]])
            print([i.objectives for i in returned_population.fronts[0]])
            print("FRONT 1:")
            print([i.features for i in returned_population.fronts[1]])
            print([i.objectives for i in returned_population.fronts[1]])
            #self.utils.fast_nondominated_sort(self.population)
            #for front in self.population.fronts:
               # self.utils.calculate_crowding_distance(front)
            children = self.utils.create_children(self.population)
        return returned_population.fronts[0]
