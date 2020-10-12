import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import yaml
from math import ceil
import collections
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
from imblearn.metrics import geometric_mean_score

import pickle

with open('nsga2/config_file.yaml', 'r') as f:
    config = yaml.load(f)

def decode_dt(var_range, **features):
    """
    Decoding hyperaparameters of decision tree.
    """

    features['criterion'] = round(features['criterion'], 0)

    if features['max_depth'] is not None:
        features['max_depth'] = int(round(features['max_depth']))
    else:
        features['max_depth'] = var_range[1][1]

    features['min_samples_split'] = int(round(features['min_samples_split']))

    #features['min_samples_leaf'] = int(round(features['min_samples_leaf']))

    if features['max_leaf_nodes'] is not None:
        features['max_leaf_nodes'] = int(round(features['max_leaf_nodes']))
    else:
        features['max_leaf_nodes'] = var_range[3][1]

    if features['class_weight'] is not None:
       features['class_weight'] = int(round(features['class_weight']))

    hyperparameters = ['criterion', 'max_depth', 'min_samples_split', 'max_leaf_nodes', 'class_weight']
    list_of_hyperparameters = [(hyperparameter, features[hyperparameter]) for hyperparameter in hyperparameters]
    features = collections.OrderedDict(list_of_hyperparameters)
    return features

def decode_log(var_range, **features):
    """
    Decoding hyperaparameters of logistic regression.
    """

    if features['class_weight'] is not None:
       features['class_weight'] = int(round(features['class_weight']))

    hyperparameters = ['max_iter', 'tol', 'C', 'l1_ratiofloat', 'class_weight']
    list_of_hyperparameters = [(hyperparameter, features[hyperparameter]) for hyperparameter in hyperparameters]
    features = collections.OrderedDict(list_of_hyperparameters)
    return features


def read_data(df_name):
    """
    Reads the dataset to work with.
    """

    df = pd.read_csv(config['ROOT_PATH'] + '/data/' + df_name + '.csv', sep = ',')
    return df

def score_text(v):
    if v == 'Low':
        return 0
    elif v == 'Medium':
        return 1
    else:
        return 2

def get_matrices(df_name, seed):
    """
    Split dataframe into train and test.
    """

    df = read_data(df_name)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    if(df_name == 'propublica_violent_recidivism'):
        X = X[['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_charge_degree', 'c_charge_desc', 'decile_score', 'score_text']]
    if(df_name == 'propublica_recidivism'):
        X = X[['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_charge_degree', 'c_charge_desc', 'decile_score', 'score_text']]

    le = preprocessing.LabelEncoder()
    for column_name in X.columns:
        if X[column_name].dtype == object:
            X[column_name] = X[column_name].astype(str)
            if(column_name == 'race' and df_name == 'adult'):
                X[column_name] = np.where(X[column_name] == 'White', 0, 1)
            elif(column_name == 'sex'):
                X[column_name] = np.where(X[column_name] == 'Male', 0, 1)
            elif(column_name == 'race' and (df_name == 'propublica_recidivism' or df_name == 'propublica_violent_recidivism')):
                X[column_name] = np.where(X[column_name] == 'Caucasian', 0, 1)
            elif(column_name == 'compas_screening_date' or column_name == 'screening_date' or column_name == 'dob'):
                X[column_name] = pd.to_datetime(X[column_name])
                X['year'] = X[column_name].dt.year
                X['month'] = X[column_name].dt.month
                X['day'] = X[column_name].dt.day
                X.drop(column_name, inplace = True, axis = 1)
            elif(column_name == 'Race'):
                X[column_name] = np.where(X[column_name] == 'W', 0, 1)
            elif(column_name == 'score_text'):
                X[column_name] = X[column_name].map(score_text)
            else:
                X[column_name] = le.fit_transform(X[column_name])
        elif(column_name == 'age' and df_name == 'german'):
             X[column_name] = np.where(X[column_name] > 25, 0, 1)
        else:
            pass

    # POSITIVE = 1
    if(df_name == 'adult'):
        y = np.where(y == '>50K', 1, 0)
    elif(df_name == 'german'):
        y = np.where(y == 1, 0, 1)
    elif(df_name == 'propublica_recidivism' or df_name == 'propublica_violent_recidivism'):
        c = X.select_dtypes(np.number).columns
        X[c] = X[c].fillna(0)
        X = X.fillna("")
        y = np.where(y == 0, 0, 1)
    elif(df_name == 'ricci'):
        y =  np.where(y >=  70.000, 0, 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state = seed)
    return X_train, X_val, X_test, y_train, y_val, y_test

def write_train_val_test(df_name, seed, X_train, X_val, X_test, y_train, y_val, y_test):
    train = X_train
    train['y'] = y_train.tolist()
    train.to_csv('./data/train_val_test/' + df_name + '_train_seed_' + str(seed) + '.csv', index = False)
    val = X_val
    val['y'] = y_val.tolist()
    val.to_csv('./data/train_val_test/' + df_name + '_val_seed_' + str(seed) + '.csv', index = False)
    test = X_test
    test['y'] = y_test.tolist()
    test.to_csv('./data/train_val_test/' + df_name + '_test_seed_' + str(seed) + '.csv', index = False)

def print_tree(classifier, features):
    dot_data = StringIO()
    export_graphviz(classifier, out_file = dot_data, feature_names = features)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("./results/trees/tree.png")

def print_properties_tree(learner):
    depth = learner.get_depth()
    leaves = learner.get_n_leaves()
    return depth, leaves

def train_model_dt(df_name, seed, **features):
    """
    Train classifier (decision tree).
    """
    train = pd.read_csv('./data/train_val_test/' + df_name + '_train_seed_' + str(seed) + '.csv')
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]

    if features['class_weight'] is not None:
       if(features['criterion'] <= 0.5):
          clf = DecisionTreeClassifier(criterion = 'gini', max_depth = features['max_depth'], min_samples_split = features['min_samples_split'], max_leaf_nodes = features['max_leaf_nodes'], class_weight = {0:features['class_weight'], 1:(10-features['class_weight'])}, presort = True)
       else:
          clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = features['max_depth'], min_samples_split = features['min_samples_split'], max_leaf_nodes = features['max_leaf_nodes'], class_weight = {0:features['class_weight'], 1:(10-features['class_weight'])}, presort = True)
    else:
       if features['criterion'] <= 0.5:
          clf = DecisionTreeClassifier(criterion = 'gini', max_depth = features['max_depth'], min_samples_split = features['min_samples_split'], max_leaf_nodes = features['max_leaf_nodes'], class_weight = features['class_weight'], presort = True)
       else:
          clf = DecisionTreeClassifier(criterion = 'entropy', max_depth = features['max_depth'], min_samples_split = features['min_samples_split'], max_leaf_nodes = features['max_leaf_nodes'], class_weight = features['class_weight'], presort = True)

    learner = clf.fit(X_train, y_train)
    return learner

def train_model_log(df_name, seed, **features):
    """
    Train classifier (decision tree).
    """
    train = pd.read_csv('./data/train_val_test/' + df_name + '_train_seed_' + str(seed) + '.csv')
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]

    #invert lambda to C
    C = 1 / features['C']

#    if features['class_weight'] is not None:
#    clf = LogisticRegression(penalty = 'elasticnet', max_iter = features['max_iter'], tol = features['tol'], C = features['C'], l1_ratio = features['l1_ratiofloat'], class_weight = {0:features['class_weight'], 1:(10-features['class_weight'])}, solver = 'saga')
    clf = LogisticRegression(penalty = 'elasticnet', max_iter = features['max_iter'], tol = features['tol'], C = C, l1_ratio = features['l1_ratiofloat'], class_weight = {0:features['class_weight'], 1:(10-features['class_weight'])}, solver = 'saga')
#    else:
#        clf = LogisticRegression(penalty = 'elasticnet', max_iter = features['max_iter'], tol = features['tol'], C = features['C'], l1_ratio = features['l1_ratiofloat'], class_weight = features['class_weight'], solver = 'saga')

    learner = clf.fit(X_train, y_train)
    return learner


def save_model(learner, dataset_name, seed, variable_name, num_of_generations, num_of_individuals, individual_id):
    # save the model to disk
    path = './results/models/' + dataset_name + '/'
    filename = 'model_' + dataset_name + '_seed_' + str(seed) + '_gen_' + variable_name + '_indiv_' + str(num_of_generations) + '_' + str(num_of_individuals) + '_id_' + individual_id + '.sav'
    pickle.dump(learner, open(path + filename, 'wb'))
    return

def val_model(df_name, learner, seed):
    """
    Test classifier.
    """
    val = pd.read_csv('./data/train_val_test/' + df_name + '_val_seed_' + str(seed) + '.csv')
    X_val = val.iloc[:, :-1]
    y_val = val.iloc[:, -1]
    y_pred = learner.predict(X_val)
    return X_val, y_val, y_pred

def test_model(df_name, learner, seed):
    test = pd.read_csv('./data/train_val_test/' + df_name + '_test_seed_' + str(seed) + '.csv')
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]
    y_pred = learner.predict(X_test)
    return X_test, y_test, y_pred

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

def evaluate_fairness(X_val, y_val, y_pred, protected_variable):
    y_val_p, y_val_u, y_pred_p, y_pred_u = split_protected(X_val, y_val, y_pred, protected_variable, 1)
    return y_val_p, y_val_u, y_pred_p, y_pred_u

def accuracy_inv(y_val, y_pred):
    err = 1 - f1_score(y_val, y_pred)
    return err

def gmean_inv(y_val, y_pred):
    gmean_error = 1 - geometric_mean_score(y_val, y_pred)
    return gmean_error

def accuracy_diff(y_val_p, y_val_u, y_pred_p, y_pred_u):
    """
    Compute difference of accuracies.
    """
    acc_p  = accuracy_score(y_val_p, y_pred_p)
    acc_u = accuracy_score(y_val_u, y_pred_u)
    acc_fair = abs(acc_u - acc_p)
    return acc_fair

def dem_tpr(y_val_p, y_val_u, y_pred_p, y_pred_u):
    """
    Compute demography metric.
    """
    tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_val_p, y_pred_p).ravel()
    tn_u, fp_u, fn_u, tp_u = confusion_matrix(y_val_u, y_pred_u).ravel()
    tpr_p = tp_p/(tp_p + fn_p)
    tpr_u = tp_u/(tp_u + fn_u)
    dem = abs(tpr_p - tpr_u)
    if(tpr_p == 0 or tpr_u == 0):
        dem = 1
    return dem

def dem_fpr(y_val_p, y_val_u, y_pred_p, y_pred_u):
    """
    Compute false positive rate parity.
    """
    tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_val_p, y_pred_p).ravel()
    tn_u, fp_u, fn_u, tp_u = confusion_matrix(y_val_u, y_pred_u).ravel()
    fpr_p = fp_p/(fp_p + tn_p)
    fpr_u = fp_u/(fp_u + tn_u)
    dem = abs(fpr_p - fpr_u)
    if(fpr_p == 0 or fpr_u == 0):
        dem = 1
    return dem

def dem_tnr(y_val_p, y_val_u, y_pred_p, y_pred_u):
    tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_val_p, y_pred_p).ravel()
    tn_u, fp_u, fn_u, tp_u = confusion_matrix(y_val_u, y_pred_u).ravel()
    tnr_p = tn_p/(tn_p + fp_p)
    tnr_u = tn_u/(tn_u + fp_u)
    dem = abs(tnr_p - tnr_u)
    return dem

def complexity(learner):
    complex = learner.get_n_leaves()
    return complex
