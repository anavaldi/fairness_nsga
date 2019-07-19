from nsga2.individual import Individual
from nsga2.ml import *
import random
import string
from collections import OrderedDict as od
from collections import Counter

class Problem:

    def __init__(self, objectives, num_of_variables, variables_range, individuals_df, num_of_generations, num_of_individuals, dataset_name, variable_name, seed, expand=True, same_range=False):
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
        self.seed = seed
        if same_range:
            for _ in range(num_of_variables):
                self.variables_range.append(variables_range[0])
        else:
            self.variables_range = variables_range

    def generate_default_individual_gini(self):
        individual = Individual()
        individual.id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))              
        individual.features = [0, None, 2, None, None]
        hyperparameters = ['criterion','max_depth', 'min_samples_split', 'max_leaf_nodes', 'class_weight']
        #individual.features = [0, None, 1, 0.00001, None]
        #hyperparameters = ['criterion','max_depth', 'min_samples_leaf', 'min_impurity_decrease', 'class_weight']
        individual.features = od(zip(hyperparameters, individual.features))
        individual.features = decode(self.variables_range, **individual.features)
        individual.creation_mode = "inicialization"
        return individual

    def generate_default_individual_entropy(self):
        individual = Individual()
        individual.id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))        
        individual.features = [1, None, 2, None, None]
        hyperparameters = ['criterion','max_depth', 'min_samples_split', 'max_leaf_nodes', 'class_weight']
        #individual.features = [1, None, 1, 0.00001, None]
        #hyperparameters = ['criterion','max_depth', 'min_samples_leaf', 'min_impurity_decrease', 'class_weight']
        individual.features = od(zip(hyperparameters, individual.features))
        individual.features = decode(self.variables_range, **individual.features)
        individual.creation_mode = "inicialization"
        return individual

    def generate_individual(self):
        individual = Individual()
        individual.id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        individual.features = [random.uniform(*x) for x in self.variables_range]
        hyperparameters = ['criterion', 'max_depth', 'min_samples_split', 'max_leaf_nodes', 'class_weight']
        #hyperparameters = ['criterion', 'max_depth', 'min_samples_leaf', 'min_impurity_decrease', 'class_weight'] 
        individual.features = od(zip(hyperparameters, individual.features))
        individual.features = decode(self.variables_range, **individual.features)
        individual.creation_mode = "inicialization"
        return individual

    def calculate_objectives(self, individual, first_individual, seed):
        if self.expand:
            #hyperparameters = decode(self.variables_range, **individual.features)
            hyperparameters = individual.features
            learner = train_model(self.dataset_name, seed, **hyperparameters)
            X, y, pred = val_model(self.dataset_name, learner, seed)
            y_fair = evaluate_fairness(X, y, pred, self.variable_name)
            #error = accuracy_inv(y, pred)
            error = gmean_inv(y, pred)
            dem_fp = dem_fpr(y_fair[0], y_fair[1], y_fair[2], y_fair[3])
            individual.objectives = [error, dem_fp]
            indiv_list = list(individual.features.items())
            criterion, max_depth, min_samples_split, max_leaf_nodes, class_weight = [item[1] for item in indiv_list]
            #criterion, max_depth, min_samples_leaf, min_impurity_decrease, class_weight = [item[1] for item in indiv_list]
            depth, leaves = print_properties_tree(learner)
            individual.actual_depth = depth
            individual.actual_leaves = leaves
            if(first_individual):
                var_range_list = list(self.variables_range)
                var_range_list[1] = (self.variables_range[1][0], depth)
                var_range_list[3] = (self.variables_range[3][0], leaves) #
                self.variable_range = []
                self.variables_range = tuple(var_range_list)
            individuals_aux = pd.DataFrame({'id': individual.id, 'creation_mode':individual.creation_mode, 'criterion': [criterion], 'max_depth': [max_depth], 'min_samples_split': [min_samples_split], 'max_leaf_nodes': [max_leaf_nodes], 'class_weight': [class_weight], 'error': error, 'dem_fp': dem_fp, 'actual_depth': depth, 'actual_leaves': leaves})
            self.individuals_df = pd.concat([self.individuals_df, individuals_aux])
            self.individuals_df.to_csv('./results/individuals/individuals_' + self.dataset_name + '_seed_' + str(seed) + '_gen_' + self.variable_name + '_indiv_' + str(self.num_of_generations) + '_' + str(self.num_of_individuals) + '.csv', index = False, header = True, columns = ['id', 'creation_mode', 'error', 'dem_fp', 'criterion', 'max_depth', 'min_samples_split', 'max_leaf_nodes', 'class_weight', 'actual_depth', 'actual_leaves'])
            #individuals_aux = pd.DataFrame({'id': individual.id, 'creation_mode': individual.creation_mode, 'criterion': [criterion], 'max_depth': [max_depth], 'min_samples_leaf': [min_samples_leaf], 'min_impurity_decrease': [min_impurity_decrease], 'class_weight': [class_weight], 'error': error, 'dem_fp': dem_fp, 'actual_depth': depth, 'actual_leaves': leaves})
            #self.individuals_df = pd.concat([self.individuals_df, individuals_aux])
            #self.individuals_df.to_csv('./results/individuals/individuals_' + self.dataset_name + '_' + self.variable_name + '_' + str(self.num_of_generations) + '_' + str(self.num_of_individuals) + '.csv', index = False, header = True, columns = ['id', 'creation_mode', 'error', 'dem_fp', 'criterion', 'max_depth', 'min_samples_leaf', 'min_impurity_decrease', 'class_weight', 'actual_depth', 'actual_leaves'])
