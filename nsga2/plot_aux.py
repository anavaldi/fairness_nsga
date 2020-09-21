import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import yaml
import numpy as np
import seaborn as sns
with open('nsga2/config_file.yaml', 'r') as f:
    config = yaml.load(f)

sns.set()

dataset = 'propublica_violent_recidivism'
variable = 'race'
generations = 60
individuals = 20

df = pd.read_csv(config['ROOT_PATH'] + '/results/population/evolution_' + dataset + '_' + variable + '_' + str(generations) + '_' + str(individuals) + '.csv')

df = df.loc[(df['rank'] == 0) & ((df['generation'] == 1) | (df['generation'] == 19) | (df['generation'] == 39) | (df['generation'] == 59) | (df['generation'] == 79) | (df['generation'] == 99))]

def rand_jitter(arr):
    stdev = .005*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def scatterplot(df, x_dim, y_dim, category, dataset, variable, generations, individuals):
   #x = rand_jitter(df[x_dim])
   #y = rand_jitter(df[y_dim])
   x = df[x_dim]
   y = df[y_dim]
   categories = df[category]
   sns.pairplot(x_vars=[x_dim], y_vars=[y_dim], data=df, hue='generation', size=5)
   #fig, ax = plt.subplots(figsize=(10, 5))

   #assigns a color to each data point
   #colors = ['#22D3FF', '#2EA865', '#ECB921', '#E33233', '#E670B6', '#4F7CBA']
   #colors = ['pink', 'green', 'yellow', 'blue', 'red', 'orange']
   #colors = ['crimson', 'purple', 'limegreen', 'gold', 'fuchsia', 'c']
   #colors = ['crimson', 'limegreen', 'c', 'fuchsia']
   #iterates through the dataset plotting each data point and assigning it its corresponding color and label
   #for i in range(len(df)):
   #  ax.scatter(x.iloc[i], y.iloc[i], alpha=0.7, color = colors[i%len(colors)], label=categories.iloc[i])


   #ax.plot(x, y,'-o', color = colors, label = list(set(categories)))
   #adds title and axes labels
   #ax.set_xlabel('Demography FPR')
   #ax.set_ylabel('1-Fmeasure')
   #ax.set_title(dataset + ' ' + variable + ' ' + str(generations) + '_' + str(individuals))

   #removing top and right borders
   #ax.spines['top'].set_visible(False)
   #ax.spines['right'].set_visible(False)

   #adds major gridlines
   #ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.7)
   #adds legend
   #ax.legend(categories.unique())
   plt.savefig('./results/figures/non_dominated_sol_' + dataset + '_' + variable + '_' + str(generations) + '_' + str(individuals) + '.png')
   plt.show()

scatterplot(df, 'dem_fp', 'error', 'generation', 'german', 'age', 100, 50)

#fig, ax = plt.subplots()
#fig.set_size_inches(16, 4)
#sns.lineplot(x='dem_fp', y='error', data=df, hue='generation',style="generation", markers='o')
#sns.lmplot('dem_fp', 'error', data=df, hue='generation', ci=None, order=1, truncate=True)
#plt.show()
#sns.lmplot('dem_fp', 'error', data=df, hue='generation', ci=None, order=2, truncate=True)

#df.plot(kind='scatter', x='dem_fp',y='error',c='generation')
#plt.savefig('./results/figures/non_dominated_sol_' + dataset + '_' + variable + '_' + str(generations) + '_' + str(individuals) + '_' + '.png')
#plt.show()


#g = sns.FacetGrid(df, hue="generation", size=8)
#g.map(plt.scatter, "dem_fp", "error")
#g.map(plt.plot, "dem_fp", "error")
#plt.show()


#for key,grp in df.groupby('generation'):
#    plt.plot(grp.dem_fp,grp.error,'o-',label = key, alpha = 0.5)
#plt.legend(loc = 'best')
#plt.show()


