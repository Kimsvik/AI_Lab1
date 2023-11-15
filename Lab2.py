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
print(ham_words)

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


def metrics_func(alpha):
    clf = MultinomialNB(alpha=alpha)
    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)

    test_score = metrics.accuracy_score(y_test, y_predict)
    test_recall = metrics.recall_score(y_test, y_predict)
    test_precision = metrics.precision_score(y_test, y_predict)

    return test_score, test_recall, test_precision

test_score = np.empty(len(alpha_range))
test_recall = np.empty(len(alpha_range))
test_precision = np.empty(len(alpha_range))

for i in range(len(alpha_range)):
    test_score[i], test_recall[i], test_precision[i] = metrics_func(alpha_range[i])

# print(test_score)

matrix = np.matrix(np.c_[alpha_range, test_score, test_recall, test_precision])
models = pd.DataFrame(data=matrix, columns=['alpha', 'test accuracy', 'test recall', 'test precision'])

# print(models)

best_index = models['test precision'].idxmax()
print(f'best index = {best_index}')

model = MultinomialNB(alpha=alpha_range[best_index])

print(f'best alpha = {alpha_range[best_index]}')

model = MultinomialNB(alpha=alpha_range[best_index])
model.fit(X_train, y_train)


"ПУНКТ 6"
confusion_matrix = confusion_matrix(y_test, model.predict(X_test))
conf_frame = pd.DataFrame(data = confusion_matrix, columns=['predicted ham', 'predicted spam'],
             index=['actual ham', 'actual spam'])
print(conf_frame)


"ПУНКТ 7"
y_pred_pr = model.predict_proba(X_test)[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_pr)

roc_auc = metrics.auc(fpr, tpr)

plt.title('Reciever Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.grid('on')
plt.show()


"ПУНКТ 8"
