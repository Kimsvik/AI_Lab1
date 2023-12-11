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

"ПУНКТ 1"
def p1_read(path):
    data_file = pd.read_csv(path)
    return data_file


"ПУНКТ 2"
def p2_sum(data_n):
    data_n.isnull().sum()
    data_n['workclass'].hist()
    f, ax = plt.subplots(figsize=(10, 8))
    ax = sns.countplot(x="income", hue="sex", data=data, palette="Set1")
    ax.set_title("Frequency distribution of income variable wrt sex")
    plt.show


data = p1_read('input/income.csv')
p2_sum(data)
