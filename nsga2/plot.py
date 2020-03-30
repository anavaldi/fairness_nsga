import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import seaborn as sns

sns.set()

def rand_jitter(arr):
    stdev = .005*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def scatterplot(df, seed, x_dim, y_dim, category, dataset, variable, generations, individuals):
   x = df[x_dim]
   y = df[y_dim]

   sns.pairplot(x_vars=[x_dim], y_vars=[y_dim], data=df, hue='generation', size=5)
   plt.savefig('./results/figures/non_dominated_sol_' + dataset + '_' + variable + '_seed_' + str(seed) + '_gen_' + str(generations) + '_indiv_' + str(individuals) + '.png')
   plt.show()


