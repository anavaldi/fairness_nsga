import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import yaml
import math


with open('nsga2/config_file.yaml', 'r') as f:
    config = yaml.load(f)


def decode(**features):
    """
    Decoding hyperaparameters.
    """
 
    if(features['criterion'] <= 0.5):
       features['criterion'] = "gini"
    else:
       features['criterion'] = "entropy"

    features['max_depth'] = math.ceil(features['max_depth'])
    features['max_leaf_nodes'] = math.ceil(features['max_leaf_nodes'])
    features['min_samples_split'] = float(features['min_samples_split'])

    return features

def read_data(df_name):
    """
    Reads the dataset to work with.
    """

    df = pd.read_csv(config['ROOT_PATH'] + '/data/' + df_name + '.csv', sep = ',')
    return df

def get_matrices(df_name):
    """
    Split dataframe into train and test.
    """

    df = read_data(df_name)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]


    #if(df_name == 'propublica_recidivism' or df_name == 'propublica_violent_recidivism'):
       #drop_columns = ['id', 'name', 'first', 'last', 'age_cat']
       #X.drop(drop_columns, inplace = True, axis = 1)

    le = preprocessing.LabelEncoder()
    for column_name in X.columns:
        if X[column_name].dtype == object:
            X[column_name] = X[column_name].astype(str)
            if(column_name == 'race' and df_name == 'adult'):
                X[column_name] = np.where(X[column_name] == 'White', 0, 1)
            elif(column_name == 'sex'):
                X[column_name] = np.where(X[column_name] == 'Male', 0, 1)
            elif(column_name == 'age' and df_name == 'german'):
                X[column_name] == np.where(X[column_name] > 25, 0, 1)
            elif(column_name == 'race' and (df_name == 'propublica_recidivism' or df_name == 'propublica_violent_recidivism')): 
                X[column_name] = np.where(X[column_name] == 'white', 0, 1)
            elif(column_name == 'compas_screening_date' or column_name == 'screening_date' or column_name == 'dob'):
                X[column_name] = pd.to_datetime(X[column_name])
                X['year'] = X[column_name].dt.year
                X['month'] = X[column_name].dt.month
                X['day'] = X[column_name].dt.day
                X.drop(column_name, inplace = True, axis = 1)
            elif(column_name == 'Race'):
                X[column_name] = np.where(X[column_name] == 'W', 0, 1)
            else:
                X[column_name] = le.fit_transform(X[column_name])
        else:
            pass


    if(df_name == 'adult'):
        y = np.where(y == '>50K', 0, 1)
    elif(df_name == 'german'):
        y = np.where(y == 1, 0, 1)
    elif(df_name == 'propublica_recidivism' or df_name == 'propublica_violent_recidivism'):
        X = X.apply(pd.to_numeric, errors = 'coerce')
        X = X.fillna('ffill', inplace = True)
        y = np.where(y == 0, 0, 1)
    elif(df_name == 'ricci'):
        y =  np.where(y >=  70.000, 0, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 15)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train) 
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_val_model(df_name, **features):
    """
    Train and test the classifier.
    """
    print(features)
    X_train, X_val, X_test, y_train, y_val, y_test = get_matrices(df_name)
    clf = DecisionTreeClassifier(random_state = 15, criterion = features['criterion'], max_depth = features['max_depth'], min_samples_split = features['min_samples_split'], min_samples_leaf = features['min_samples_leaf'], max_leaf_nodes = features['max_leaf_nodes'], min_impurity_decrease = features['min_impurity_decrease'], class_weight = {0:features['class_weight'], 1:(1-features['class_weight'])})
    y_pred = clf.fit(X_train, y_train).predict(X_val)
    return X_val, y_val, y_pred


def split_protected(X, y, pred, protected_variable, protected_value = 1):
    """
    Split datasets into (white, black), (male, female), etc.
    """
 
    df = pd.DataFrame({protected_variable: X[protected_variable], 'y_val': y, 'y_pred': pred})
    df_p = df.loc[df[protected_variable] == protected_value]
    df_u = df.loc[df[protected_variable] != protected_value]
    y_val_p = df_p['y_val']
    y_val_u = df_u['y_val']
    y_pred_p = df_p['y_pred']
    y_pred_u = df_u['y_pred']
    return y_val_p, y_val_u, y_pred_p, y_pred_u

def evaluate(df_name, protected_variable, **features):
    """
    Evaluate the model with fairness data.
    """

    X_val, y_val, y_pred = train_val_model(df_name, **features)

    return X_val, y_val, y_pred

def evaluate_fairness(X_val, y_val, y_pred):
    y_val_p, y_val_u, y_pred_p, y_pred_u = split_protected(X_val, y_val, y_pred, 'sex', 1)

    return y_val_p, y_val_u, y_pred_p, y_pred_u

def accuracy_inv(y_val, y_pred):
    err = 1 - accuracy_score(y_val, y_pred)
    return err

def accuracy_diff(y_val_p, y_val_u, y_pred_p, y_pred_u):
    """
    Compute difference of accuracies.
    """
    acc_p  = accuracy_score(y_val_p, y_pred_p)
    acc_u = accuracy_score(y_val_u, y_pred_u)
    acc_fair = abs(acc_u - acc_p)
    return acc_fair

def dem_fpr(y_val_p, y_val_u, y_pred_p, y_pred_u):
    """
    Compute demography metric.
    """
    tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_val_p, y_pred_p).ravel() 
    tn_u, fp_u, fn_u, tp_u = confusion_matrix(y_val_u, y_pred_u).ravel()
    tpr_p = tp_p/(tp_p + fn_p)
    tpr_u = tp_u/(tp_u + fn_u)
    dem = abs(tpr_p - tpr_u)
    return dem

def dem_tnr(y_val_p, y_val_u, y_pred_p, y_pred_u):
    tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_val_p, y_pred_p).ravel()
    tn_u, fp_u, fn_u, tp_u = confusion_matrix(y_val_u, y_pred_u).ravel()
    tnr_p = tn_p/(tn_p + fp_p)
    tnr_u = tn_u/(tn_u + fp_u)
    dem = abs(tnr_p - tnr_u)
    return dem


