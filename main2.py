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

# x_train, x_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.2, random_state=1234)
x_train, x_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.2)

# print(x_test)

# y.replace(('M', 'B'), (1, 0), inplace=True)


# for i in range(1, 51, 1):


model = KNeighborsClassifier(n_neighbors=5)
# print(model)
model.fit(x_train, y_train)

predictions = model.predict(x_test)

# print(classification_report(y_test, predictions))
accuracy = accuracy_score(y_test, predictions)
print(accuracy)

accuracy = []
number_of_neighbours = 50
# neighbours = np.arange(1, number_of_neighbours+1, 1)

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


