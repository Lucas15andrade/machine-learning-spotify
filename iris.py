from sklearn import svm
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from matplotlib import style
from matplotlib import pyplot

iris = datasets.load_iris()

treino = iris.data
classes = iris.target

treino[:-30]
classes[:-30]

#clf = svm.SVC.fit(treino[:-30],classes[:-30])

X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], test_size=0.33)
print(iris.keys())
modelo = SVC(gamma='auto')
modelo.fit(X_train, y_train)
predicao = modelo.predict(X_test)

acuracia = metrics.accuracy_score(predicao, y_test)
print('Acurácia: ',acuracia)

style.use('ggplot')
pyplot.xlabel('Petal length')
pyplot.ylabel('Petal width')
pyplot.title('Petal Width vs Length')
pyplot.scatter(treino[-30:,2], treino[-30:,3], c=clf.predict(treino[-30:]))