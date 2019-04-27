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


def remove_features(lista_features):
    for i in lista_features:
        dataset.drop(i, axis=1, inplace=True)
    return 0

dataset = pd.read_csv('base/data.csv')
print(dataset.head())

#Verificando se existem valores faltosos
print(dataset.isnull().sum())

#print(dataset.describe())

style.use('seaborn-colorblind')
dataset.plot(x='acousticness', y= 'danceability', c='target', kind='scatter', colormap='Accent_r')
#plt.show()


remove_features(['song_title'])

print(dataset.head())

enc = LabelEncoder()
inteiros = enc.fit_transform(dataset['artist'])
dataset['artist_inteiros'] = inteiros

remove_features(['artist'])

print(dataset.head())

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10,100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}

grid = GridSearchCV(SVC(), param_grid, verbose=1, cv=3)

X_train, X_test, y_train, y_test = train_test_split(dataset, dataset['target'], test_size=0.33)
grid.fit(X_train, y_train)

print('Melhores parametros: ', grid.best_params_)
print('Melhor resultado: ', grid.best_score_)

predicao = grid.predict(X_test)

acuracia = metrics.accuracy_score(predicao, y_test)

print('Acurácia: ', acuracia)