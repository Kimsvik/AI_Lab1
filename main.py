import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

dataset = pd.read_csv('input/breast_cancer.csv')
ds = pd.get_dummies(dataset, columns=['diagnosis'], drop_first=True)
print(ds)
y = ds['diagnosis_M']
x = ds.drop('diagnosis_M', axis=1)
x = x.drop('id', axis=1)
i = len(x.columns)
x = x.drop(x.columns[i-1], axis=1)
# y.replace(('M', 'B'), (1, 0), inplace=True)
sc = StandardScaler()
sc.fit(x)
x_ans = sc.transform(x)

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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1234)
print(x_test)

best_model = KNeighborsClassifier(
    n_neighbors=10,
    weights='distance',
    algorithm='auto',
    leaf_size=30,
    metric='euclidean',
    metric_params=None,
    n_jobs=4
)
print(best_model)

model_params = best_model.get_params()
print(model_params)
tuned_params = {}
for k, v in model_params.items():
    tuned_params[k] = [v]
tuned_params['n_neighbors'] = range(1, 30)
clf = GridSearchCV(KNeighborsClassifier(), tuned_params, cv=10, n_jobs=-1)
clf.fit(x_train, y_train)
best_params = clf.best_params_


best_model = KNeighborsClassifier(**best_params)
best_model.fit(x_train, y_train)
predicted = best_model.predict(x_test)

print('Used params:', best_params)
print('Evaluation:\n', metrics.classification_report(y_test, predicted))
