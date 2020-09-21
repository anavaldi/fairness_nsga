import pandas as pd
import glob
import yaml


dataset_name = 'propublica_recidivism'
seed = 15 

# read dataset
with open('config_file.yaml', 'r') as f:
    config = yaml.load(f)
print(config)

# merge all
df_all = pd.concat(map(pd.read_csv, glob.glob(config['ROOT_PATH'] + '/results/population/' + dataset_name + '/' +  "*.csv")))
print(df.shape)

