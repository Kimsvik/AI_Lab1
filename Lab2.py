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
def p1_read(path, encoding):
    data_file = pd.read_csv(path, encoding=encoding)
    return data_file


"ПУНКТ 2"
def p2_pie_chart(data_file):
    target = data_file['v1'].value_counts()
    print(target)
    target.plot(kind='pie')
    plt.title('pie chart')
    plt.ylabel('')
    plt.show()


"ПУНКТ 3"
def p3_mf_words(data_file):
    ham_words = Counter(" ".join(data[data['v1'] == 'ham']['v2']).split()).most_common(20)
    # print(ham_words)

    df_ham_words = pd.DataFrame.from_dict(ham_words)
    df_ham_words = df_ham_words.rename(columns={0: 'words in non-spam', 1: 'count'})

    df_ham_words.plot.bar(legend=False)
    y_pos = np.arange(len(df_ham_words['words in non-spam']))
    plt.xticks(y_pos, df_ham_words['words in non-spam'])
    plt.title('more frequent words in non-spam messages')
    plt.xlabel('words')
    plt.ylabel('number')
    plt.show()


"ПУНКТ 4"
def p4_token(data_file):
    tokenizer = feature_extraction.text.CountVectorizer(stop_words='english')
    X_token = tokenizer.fit_transform(data_file['v2'])
    return X_token


"ПУНКТ 5"
def p5_1_split(data_file):
    data_file['v1'] = data_file['v1'].map({'spam': 1, 'ham': 0})
    x_train, x_test, y_train, y_test = model_selection.train_test_split(X, data_file['v1'], test_size=0.33)
    return x_train, x_test, y_train, y_test


def p5_2_MultinominalNB(x_train, x_test, y_train, y_test):
    alpha_range = np.arange(0.1, 20, 0.1)

    def metrics_func(alpha):
        clf = MultinomialNB(alpha=alpha)
        clf.fit(x_train, y_train)

        y_predict = clf.predict(x_test)

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
    print(f'\nbest index = {best_index}')
    print(f'best alpha = {alpha_range[best_index]}')
    print(f'best accuracy = {"%.3f" % test_score[best_index]}')

    model = MultinomialNB(alpha=alpha_range[best_index])
    model.fit(x_train, y_train)

    plt.plot(alpha_range, test_score)
    plt.grid(True)
    plt.ylabel('Accuracy')
    plt.xlabel('alpha')
    plt.show()

    # plt.plot(alpha_range, test_precision)
    # plt.grid(True)
    # plt.ylabel('Precision')
    # plt.xlabel('alpha')
    # plt.show()

    return model


"ПУНКТ 6"
def p6_confusion(model, y_test, x_test):
    confusion_matr = confusion_matrix(y_test, model.predict(x_test))
    conf_frame = pd.DataFrame(data=confusion_matr, columns=['predicted ham', 'predicted spam'],
                              index=['actual ham', 'actual spam'])
    print(conf_frame)




"ПУНКТ 7"
def p7_roc(model, x_test, y_test):
    y_pred_pr = model.predict_proba(x_test)[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_pr)

    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Reciever Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')

    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(True)
    plt.show()


"ПУНКТ 8"
def p8_svc(x_train, x_test, y_train, y_test):
    c_range = np.arange(0.1, 3, 0.1)

    def metrics_func(c):
        clf = SVC(C=c)
        clf.fit(x_train, y_train)

        y_predict = clf.predict(x_test)

        test_score = metrics.accuracy_score(y_test, y_predict)
        test_recall = metrics.recall_score(y_test, y_predict, zero_division=1.0)
        test_precision = metrics.precision_score(y_test, y_predict, zero_division=1.0)

        return test_score, test_recall, test_precision

    test_score = np.empty(len(c_range))
    test_recall = np.empty(len(c_range))
    test_precision = np.empty(len(c_range))

    for i in range(len(c_range)):
        test_score[i], test_recall[i], test_precision[i] = metrics_func(c_range[i])

    # print(test_score)

    matrix = np.matrix(np.c_[c_range, test_score, test_recall, test_precision])
    models = pd.DataFrame(data=matrix, columns=['c', 'test accuracy', 'test recall', 'test precision'])

    # print(models)

    best_index = models['test precision'].idxmax()
    print(f'best index = {best_index}')
    print(f'best c = {c_range[best_index]}')
    print(f'best accuracy = {test_score[best_index]}')

    model = SVC(C=c_range[best_index], probability=True)
    model.fit(x_train, y_train)

    plt.plot(c_range, test_score)
    plt.show()

    return model


data = p1_read('input/spam.csv', "ISO-8859-1")
p2_pie_chart(data)
p3_mf_words(data)
X = p4_token(data)
x_train, x_test, y_train, y_test = p5_1_split(data)
model = p5_2_MultinominalNB(x_train, x_test, y_train, y_test)
# model = p8_svc(x_train, x_test, y_train, y_test)
p6_confusion(model, y_test, x_test)
p7_roc(model, x_test, y_test)
