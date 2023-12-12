import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from sklearn import feature_extraction
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier


"ПУНКТ 1"
def p1_read(path):
    data_file = pd.read_csv(path)
    return data_file


"ПУНКТ 2"
def p2_sum(data_n):
    data_n.isnull().sum()
    data_n['workclass'].hist()
    f, ax = plt.subplots(figsize=(10, 8))
    ax = sns.countplot(x="income", hue="sex", data=data_n, palette="Set1")
    ax.set_title("Frequency distribution of income variable wrt sex")
    plt.show()

    f, ax = plt.subplots(figsize=(10, 8))
    x = data_n['age']
    ax = sns.distplot(x, bins=10, color='blue')
    ax.set_title("Distribution of age variable")
    plt.show()

    f, ax = plt.subplots(figsize=(10, 8))
    x = data_n['age']
    ax = sns.boxplot(x)
    ax.set_title("Visualize outliers in age variable")
    plt.show()

    f, ax = plt.subplots(figsize=(10, 8))
    ax = sns.boxplot(x="income", y="age", data=data_n)
    ax.set_title("Visualize income wrt age variable")
    plt.show()

    f, ax = plt.subplots(figsize=(10, 8))
    ax = sns.boxplot(x="income", y="age", hue="sex", data=data_n)
    ax.set_title("Visualize income wrt age and sex variable")
    ax.legend(loc='upper right')
    plt.show()

    data_n.replace('?', np.NaN, inplace=True)

    categorical = [var for var in data_n.columns if data[var].dtype=='O']
    data_n[categorical].head()


"ПУНКТ 3"
def p3_split(data_file):
    data_file['income'] = data_file['income'].map({'>50K': 1, '<=50K': 0})
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, data_file['income'], test_size=0.33)
    return x_train, x_test, y_train, y_test

"ПУНКТ 4"



data = p1_read('C:/Users/Сергей/PycharmProjects/AI_Lab1/input/income.csv')
p2_sum(data)



