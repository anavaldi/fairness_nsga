import matplotlib.pyplot as plt

from nsga2.problem import Problem
from nsga2.evolution import Evolution
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 15)

classifier = DecisionTreeClassifier(random_state = 15, criterion = 'gini', max_depth = None, min_samples_split = 2).fit(X_train, y_train)
y_pred = classifier.fit(X_train, y_train).predict(X_test)

cm = confusion_matrix(y_test, y_pred)[0,0]

print(cm)
