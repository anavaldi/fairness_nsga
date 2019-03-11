import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import yaml



with open('config_file.yaml', 'r') as f:
    config = yaml.load(f)

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


    if(df_name == 'propublica_recidivism' or df_name == 'propublica_violent_recidivism'):
       drop_columns = ['id', 'name', 'first', 'last', 'age_cat']
       X.drop(drop_columns, inplace = True, axis = 1)

    le = preprocessing.LabelEncoder()
    for column_name in X.columns:
        if X[column_name].dtype == object:
            X[column_name] = X[column_name].astype(str)
            #if(column_name == 'race' and df_name == 'adult'):
            #    X[column_name] = np.where(X[column_name] == 'White', 0, 1)
            #elif(column_name == 'sex'):
            #    X[column_name] = np.where(X[column_name] == 'Male', 0, 1)
            #elif(column_name == 'age' and df_name == 'german'):
            #    X[column_name] == np.where(X[column_name] > 25, 0, 1)
            #elif(column_name == 'race' and (df_name == 'propublica_recidivism' or df_name == 'propublica_violent_recidivism')): 
            #    X[column_name] = np.where(X[column_name] == 'white', 0, 1)
            # elif(column_name == 'compas_screening_date' or column_name == 'screening_date' or column_name == 'dob'):
            #    X[column_name] = pd.to_datetime(X[column_name])
            #    X['year'] = X[column_name].dt.year
            #    X['month'] = X[column_name].dt.month
            #    X['day'] = X[column_name].dt.day
            #    X.drop(column_name, inplace = True, axis = 1)
            #elif(column_name == 'Race'):
            #    X[column_name] = np.where(X[column_name] == 'W', 0, 1)
            #else:
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
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state = 15) 
    return X_train, X_val, X_test, y_train, y_val, y_test

def train_val_model(df_name):
    """
    Train and test the classifier
    """
    X_train, X_val, X_test, y_train, y_val, y_test = get_matrices(df_name)
    clf = DecisionTreeClassifier(random_state = 15)
    y_pred = clf.fit(X_train, y_train).predict(X_val)
    return X_val, y_val, y_pred

def evaluate(X_val, y_val, y_pred, protected_variable):
    """
    Evaluate the model with fairness data.
    """

    X_val, y_val, y_pred = train_test_model(df_name)
    df_val = pd.DataFrame({protected_variable : X_val[protected_variable], 'y_val' : y_val, 'y_pred' : y_pred}) 
    y_val_p = df_val[df_val.protected_variable == 1].y_val
    y_val_u = df_val[df_val.protected_variable == 0].y_val
    y_pred_p = df_val[df_val.protected_variable == 1].y_pred
    y_pred_u = df_val[df_val.protected_variable == 0].y_pred

    acc_p = accuracy(y_val_p, y_pred_p)
    acc_u = accuracy(y_val_u, y_pred_u)
    tpr_p = tpr(y_val_u, y_pred_p)
    tpr_u = tpr(y_val_u, y_pred_u)
    tnr_p = tnr(y_val_p, y_pred_p)
    tnr_u = tnr(y_val_u, y_pred_u)
    bcr_p = bcr(y_val_p, y_pred_p)
    bcr_u = bcr(y_val_u, y_pred_u)



