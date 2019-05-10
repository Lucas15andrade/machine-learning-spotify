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

X_train, X_test, y_train, y_test = train_test_split(base, base['target'], test_size=0.33)

param_grid = {
    'n_neighbors': range(1,40,2),
    'metric': ['euclidean', 'minkowski', 'chebyshev'],
}

grid = GridSearchCV(KNeighborsClassifier, param_grid, verbose=1, cv=3)
grid.fit(X_train, y_train)

print('Melhores parâmetros', grid.best_params_)
print('Melhor acurácia', grid.best_score_)