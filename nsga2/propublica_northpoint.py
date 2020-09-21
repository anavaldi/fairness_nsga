from ml import *
import pandas as pd

dataset = 'propublica_recidivism'
variable = 'race'
seed = 958

#df = read_data(dataset)
df = get_matrices(dataset, seed)[2]
print(df['score_text'])
df['y_northpoint'] = np.where(df['score_text'] == 1, 0.0, 1.0)

#df[variable] = np.where(df[variable] == 'Caucasian', 0, 1)
y_val = get_matrices(dataset, seed)[5]
y_pred = df['y_northpoint']

df = df.iloc[:, :-1]
df = df.iloc[:, :-1]
print(df.columns)
y_val_p, y_val_u, y_pred_p, y_pred_u = evaluate_fairness(df, y_val, y_pred, variable)

print('GMEAN:')
print(gmean_inv(y_val, y_pred))

print('DEM_FPR:')
print(dem_fpr(y_val_p, y_val_u, y_pred_p, y_pred_u))
