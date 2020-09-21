import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import numpy as np
import seaborn as sns


import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D

sns.set()

def rand_jitter(arr):
    stdev = .005*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

def scatterplot(df, seed, x_dim, y_dim, category, dataset, variable, generations, individuals):
   #x = rand_jitter(df[x_dim])
   #y = rand_jitter(df[y_dim])
   x = df[x_dim]
   y = df[y_dim]

   sns.pairplot(x_vars=[x_dim], y_vars=[y_dim], data=df, hue='generation', size=5)
   plt.savefig('./results/figures/non_dominated_sol_' + dataset + '_' + variable + '_seed_' + str(seed) + '_gen_' + str(generations) + '_indiv_' + str(individuals) + '.png')
   plt.show()


def scatterplot_3d(df, seed, x_dim, y_dim, z_dim, category, dataset, variable, generations, individuals):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x_1 = df.loc[df['generation']==1, x_dim]
    y_1 = df.loc[df['generation']==1, y_dim]
    z_1 = df.loc[df['generation']==1, z_dim]

    x_20 = df.loc[df['generation']==20, x_dim]
    y_20 = df.loc[df['generation']==20, y_dim]
    z_20 = df.loc[df['generation']==20, z_dim]

    x_40 = df.loc[df['generation']==40, x_dim]
    y_40 = df.loc[df['generation']==40, y_dim]
    z_40 = df.loc[df['generation']==40, z_dim]
    
    x_60 = df.loc[df['generation']==60, x_dim]
    y_60 = df.loc[df['generation']==60, y_dim]
    z_60 = df.loc[df['generation']==60, z_dim]

    x_80 = df.loc[df['generation']==80, x_dim]
    y_80 = df.loc[df['generation']==80, y_dim]
    z_80 = df.loc[df['generation']==80, z_dim]

    x_100 = df.loc[df['generation']==100, x_dim]
    y_100 = df.loc[df['generation']==100, y_dim]
    z_100 = df.loc[df['generation']==100, z_dim]


    ax.scatter(x_1, y_1, z_1, marker='o', color='blue')
    ax.scatter(x_20, y_20, z_20, marker='o', color='orange')
    ax.scatter(x_40, y_40, z_40, marker='o', color='green')
    ax.scatter(x_60, y_60, z_60, marker='o', color='red')
    ax.scatter(x_80, y_80, z_80, marker='o', color='purple')
    ax.scatter(x_100, y_100, z_100, marker='o', color='brown')

    ax.set_xlabel('dem_fp')
    ax.set_ylabel('error')
    ax.set_zlabel('tree_depth')

    plt.savefig('./results/figures/non_dominated_sol_' + dataset + '_' + variable + '_seed_' + str(seed) + '_gen_' + str(generations) + '_indiv_' + str(individuals))
    plt.show()
