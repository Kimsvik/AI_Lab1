import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


"""
1. id
2. dignosis - M - malignant (злокачественная) и B - benign (доброкачественная)
3. radius_mean (средний радиус) - означает расстояние от центра до точек на периметре
4. texture_mean (средняя текстура) - шкала серости
5. perimeter_mean (средний периметр) - средний размер ядра опухоли
6. area_mean (средняя площадь)
7. smoothness_mean (сглаженность) - среднее значение локального изменения длин радисов
8. compactness_mean (компактность) - (средний периметр)^2/площадь - чем меньше, тем более округлый объект
9. concavity (вогнутость) - среднее значение выраженности вогнутых участков контура
10. concave points (вогнутые точки) - среднее число вогнутых точек в контуре
"""

raw_data = pd.read_csv('input/breast_cancer.csv')
dataset = pd.get_dummies(raw_data, columns=['diagnosis'], drop_first=True)
print(dataset)

y = dataset['diagnosis_M']
x = dataset.drop('diagnosis_M', axis=1)
x = x.drop('id', axis=1)
x = x.drop(x.columns[-1], axis=1)

scaler = StandardScaler()
scaler.fit(x)
scaled_x = scaler.transform(x)

scaled_data = pd.DataFrame(scaled_x, columns=x.columns)


def KNC(x, y):
    # x_train, x_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.2, random_state=1234)
    x_train, x_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.2)
    model = KNeighborsClassifier(n_neighbors=5)
    # print(model)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    print(accuracy)

    accuracy = []
    number_of_neighbours = 50

    max_accuracy = 0
    number_of_neighbour = 0
    for i in np.arange(1, number_of_neighbours+1):
        new_model = KNeighborsClassifier(n_neighbors=i, weights='distance')
        new_model.fit(x_train, y_train)
        new_predictions = new_model.predict(x_test)
        accuracy_i = accuracy_score(y_test, new_predictions)
        accuracy.append(accuracy_i)
        if accuracy_i > max_accuracy:
            max_accuracy = accuracy_i
            number_of_neighbour = i
    print(f'Max accuracy = {max_accuracy}')
    print(f'Best number of neighbours = {number_of_neighbour}')


    plt.plot(accuracy)
    plt.grid(True)
    plt.xlabel('Number of neighbours')
    plt.ylabel('Accuracy')
    plt.show()


def CV(x, y):
    x = np.array(x)
    y = np.array(y)

    # Задаем количество блоков (в данном случае - 5)
    n_splits = 5

    # Создаем генератор разбиений
    kf = KFold(n_splits=n_splits)

    # Генерируем итератор, возвращающий индексы обучающих и тестовых наборов данных для каждого разбиения
    for train_index, test_index in kf.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # обучаем модель и оцениваем на тестовом наборе данных

    model = KNeighborsClassifier(n_neighbors=5)
    array = cross_val_score(model, x, y, cv=kf, scoring='accuracy')
    print(array)


def LR(x, y):
    clf = LogisticRegression(random_state=0).fit(x, y)
    clf.predict(x[:2, :])
    clf.predict_proba(x[:2, :])
    clf.score(x, y)

LR(x, y)