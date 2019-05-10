from nbconvert.exporters import notebook
from sklearn import svm
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import style
from matplotlib import pyplot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

base = pd.read_csv('base/data.csv')
def remove_features(lista_features):
    for i in lista_features:
        base.drop(i, axis=1, inplace=True)
    return 0

remove_features(['song_title'])

enc = LabelEncoder()
inteiros = enc.fit_transform(base['artist'])
base['artist_inteiros'] = inteiros

remove_features(['artist'])

param_grid = {
    'hidden_layer_sizes': [(13,13), (8,8), (8,)],
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'adam'],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'learning_rate': ['constant','adaptive'],
    'max_iter': [500, 1000, 5000],
    'momentum': [0.5, 0.8, 0.9],
}

X_train, X_test, y_train, y_test = train_test_split(base, base['target'], test_size=0.33)

grid = GridSearchCV(MLPClassifier(), param_grid, verbose=3)
grid.fit(X_train, y_train)

print('Melhores parâmetros', grid.best_params_)
print('Melhor acurácia', grid.best_score_)
