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

"ПУНКТ 1"
data = pd.read_csv('input/spam.csv', encoding="ISO-8859-1")


"ПУНКТ 2"
target = data['v1'].value_counts()
#target = pd.value_counts(data)
print(target)
target.plot(kind='pie')
plt.title('pie chart')
plt.ylabel('')
plt.show()


"ПУНКТ 3"
ham_words = Counter(" ".join(data[data['v1']=='ham']['v2']).split()).most_common(20)
df_ham_words = pd.DataFrame.from_dict(ham_words)
df_ham_words = df_ham_words.rename(columns={0: 'words in non-spam', 1:'count'})

df_ham_words.plot.bar(legend=False)
y_pos = np.arange(len(df_ham_words['words in non-spam']))
plt.xticks(y_pos, df_ham_words['words in non-spam'])
plt.title('more frequent words in non-spam messages')
plt.xlabel('words')
plt.ylabel('number')
plt.show()


"ПУНКТ 4"
tokenizer = feature_extraction.text.CountVectorizer(stop_words='english')
X = tokenizer.fit_transform(data['v2'])


"ПУНКТ 5"
data['v1'] = data['v1'].map({'spam':1, 'ham':0})
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['v1'], test_size=0.33)

alpha_range = np.arange(0.1, 20, 0.1)

y_predict = []
metrics.accuracy_score(y_test, y_predict)
metrics.recall_score(y_test, y_predict)
metrics.precision_score(y_test, y_predict)

matrix = np.matrix(np.c_[alpha_range, train_score, test_score, test_recall, test_precision])
models = pd.DataFrame(data=matrix, columns=['alpha', 'train accuracy', 'test accuracy', 'test recall', 'test precision'])
