import pandas as pd
import math
import glob
import yaml
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from nsga2.ml import train_model, test_model, evaluate_fairness, gmean_inv, dem_fpr
from collections import OrderedDict as od
from collections import Counter

with open('nsga2/config_file.yaml', 'r') as f:
    config = yaml.load(f)


# pareto function

def identify_pareto(scores):
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] <= scores[i]) and any(scores[j] < scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]



df_names = ['adult', 'german', 'propublica_recidivism', 'propublica_violent_recidivism', 'ricci']

adult_seeds = [832, 58, 616, 551, 300, 707, 326, 509, 474, 765]
german_seeds = [537, 620, 197, 860, 132, 604, 542, 517, 345, 278]
propublica_recidivism_seeds = [958, 923, 840, 653, 522, 382, 317, 263, 149, 15]
propublica_violent_recidivism_seeds = [894, 366, 440, 24, 436, 941, 71, 838, 893, 738]
ricci_seeds = [263, 945, 601, 315, 902, 216, 374, 205, 274, 607]

check = True

for df_name in df_names:
    print(df_name)
    if check:
    	df = pd.concat(map(pd.read_csv, glob.glob(config['ROOT_PATH'] + '/results/population/' + df_name + '/' +  "*.csv")))
    	df_last_generation = df.loc[df['generation'] == 200]
    	df_last_generation = df_last_generation.reset_index()
    	rows_pareto = identify_pareto(df_last_generation[['error', 'dem_fp']].as_matrix())
    	df_pareto = df_last_generation.iloc[rows_pareto]
    	#print(df_pareto.head())
    	seeds = globals()[df_name + '_seeds']
    	if(df_name == 'german'): sens_atr = 'age'
    	elif(df_name == 'ricci'): sens_atr = 'Race'
    	else: sens_atr = 'race'
    	results = pd.DataFrame()
    	for seed in seeds:
            print(seed)
            for index, row in df_pareto.iterrows():
                #print('class_weght')
                #print(row['class_weight'])
                #print('max_leaf_nodes')
                #print(row['max_leaf_nodes'])
                if(math.isnan(row['class_weight'])): 
                    row['class_weight'] = None
                    feature_values = [row['criterion'], row['max_depth'], row['min_samples_split'], int(row['max_leaf_nodes']), row['class_weight']]
                else:
                    feature_values = [row['criterion'], row['max_depth'], row['min_samples_split'], int(row['max_leaf_nodes']), int(row['class_weight'])]
                hyperparameters = ['criterion', 'max_depth', 'min_samples_split', 'max_leaf_nodes', 'class_weight']
                features = od(zip(hyperparameters, feature_values))
                #print(feature_values)
                learner = train_model(df_name, seed, **features)
                X, y, pred = test_model(df_name, learner, seed)
                y_fair = evaluate_fairness(X, y, pred, sens_atr)
                error = gmean_inv(y, pred)
                dem_fp = dem_fpr(y_fair[0], y_fair[1], y_fair[2], y_fair[3])
                #print(error)
                #print(dem_fp)
                results_aux = pd.DataFrame({'id': [row['id']], 'seed': [seed], 'criterion': row['criterion'], 'max_depth': row['max_depth'], 'min_samples_split': row['min_samples_split'], 'max_leaf_nodes': row['max_leaf_nodes'], 'class_weight': row['class_weight'], 'error_val': row['error'], 'dem_fp_val': row['dem_fp'], 'error_test': error, 'dem_fp_test': dem_fp, 'actual_depth': row['actual_depth'], 'actual_leaves': row['actual_leaves']})
                results = pd.concat([results, results_aux])
                results.to_csv('results/test/' + df_name + '_' + sens_atr + '.csv',  header = True, columns = ['id', 'seed', 'criterion', 'max_depth', 'min_samples_split', 'max_leaf_nodes', 'class_weight', 'error_val', 'dem_fp_val', 'error_test', 'dem_fp_test', 'actual_depth', 'actual_leaves'
])
