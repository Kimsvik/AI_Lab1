import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import sklearn.tree
from sklearn import feature_extraction
from sklearn import model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout


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
def p3_token(data_file):
    tokenizer = feature_extraction.text.CountVectorizer()
    X_token = tokenizer.fit_transform(data_file[:'native_country'])
    return X_token

def p3_split(df):
    #df['income'] = df['income'].map({'>50K': 1, '<=50K': 0})

    def zip_clmn(clmn):
        list_of_uniq = list(set(clmn))

        N = len(list_of_uniq)
        spam = list(range(0, N))

        dict_clmn = dict(zip(list_of_uniq, spam))
        return dict_clmn

    df['workclass'] = df['workclass'].map(zip_clmn(df['workclass']))
    df['education'] = df['education'].map(zip_clmn(df['education']))
    df['marital_status'] = df['marital_status'].map(zip_clmn(df['marital_status']))
    df['occupation'] = df['occupation'].map(zip_clmn(df['occupation']))
    df['relationship'] = df['relationship'].map(zip_clmn(df['relationship']))
    df['race'] = df['race'].map(zip_clmn(df['race']))
    df['sex'] = df['sex'].map(zip_clmn(df['sex']))
    df['native_country'] = df['native_country'].map(zip_clmn(df['native_country']))
    df['income'] = df['income'].map(zip_clmn(df['income']))

    new_ydf = df['income']
    new_df = df.drop('income', axis=1)

    x_train, x_test, y_train, y_test = model_selection.train_test_split(new_df, new_ydf, test_size=0.33)
    return x_train, x_test, y_train, y_test


"ПУНКТ 4"
def p4_Tree(x_train, x_test, y_train, y_test):
    depth_range = np.arange(1, 100, 1)
    train_score = []
    test_score = []
    optimal_depth = 0
    useless_var = 0

    for i in range(len(depth_range)):
        model = DecisionTreeClassifier(max_depth=depth_range[i])
        model.fit(x_train, y_train)
        train_score.append(f1_score(y_train, model.predict(x_train)))
        test_score.append(f1_score(y_test, model.predict(x_test)))
        if (test_score[i] > useless_var):
            useless_var = test_score[i]
            optimal_depth = depth_range[i]

    model = DecisionTreeClassifier(max_depth=optimal_depth)
    model.fit(x_train, y_train)

    print(optimal_depth)

    f = plt.figure()
    s1 = f.add_subplot(1, 1, 1)
    s1.grid(True)
    f.clf()

    plt.title("Зависимость F-меры от глубины (depth_range)")
    plt.plot(depth_range, train_score, depth_range, test_score)
    plt.ylabel("F-мера")
    plt.xlabel("Глубина")
    plt.grid(True)

    plt.show()


"ПУНКТ 5"
def p5_Forest(x_train, x_test, y_train, y_test):
    estimators_range = np.arange(10, 30, 1)
    train_score = []
    test_score = []
    optimal_estimator = 0
    useless_var = 0

    for i in range(len(estimators_range)):
        model = RandomForestClassifier(n_estimators=estimators_range[i])
        model.fit(x_train, y_train)
        train_score.append(f1_score(y_train, model.predict(x_train)))
        test_score.append(f1_score(y_test, model.predict(x_test)))
        if (test_score[i] > useless_var):
            useless_var = test_score[i]
            optimal_estimator = estimators_range[i]

    model = RandomForestClassifier(n_estimators=optimal_estimator)
    model.fit(x_train, y_train)

    print(optimal_estimator)

    f = plt.figure()
    s1 = f.add_subplot(1, 1, 1)
    s1.grid(True)
    f.clf()

    plt.title("Зависимость F-меры от глубины (estimators range)")
    plt.plot(estimators_range, train_score, estimators_range, test_score)
    plt.ylabel("F-мера")
    plt.xlabel("Количество деревьев")
    plt.grid(True)

    plt.show()


"ПУНКТ 6"
def p6_Boosting(x_train, x_test, y_train, y_test):
    boost_estimators_range = np.arange(10, 200, 10)
    train_score = []
    test_score = []
    optimal_estimator = 0
    useless_var = 0

    for i in range(len(boost_estimators_range)):
        model = CatBoostClassifier(n_estimators=boost_estimators_range[i], task_type="GPU", devices='0:1')
        model.fit(x_train, y_train)
        train_score.append(f1_score(y_train, model.predict(x_train)))
        test_score.append(f1_score(y_test, model.predict(x_test)))
        if (test_score[i] > useless_var):
            useless_var = test_score[i]
            optimal_estimator = boost_estimators_range[i]

    model = CatBoostClassifier(n_estimators=optimal_estimator)
    model.fit(x_train, y_train)

    print(optimal_estimator)

    f = plt.figure()
    f.set_size_inches(16, 5)
    s1 = f.add_subplot(1, 1, 1)
    s1.grid(True)
    f.clf()

    plt.title("Зависимость F-меры от глубины (boost estimators range)")
    plt.plot(boost_estimators_range, train_score, boost_estimators_range, test_score)
    plt.ylabel("F-мера")
    plt.xlabel("Количество деревьев")
    plt.grid(True)

    plt.show()


"ПУНКТ 7"
def p7_Precept(x_train, x_test, y_train, y_test):

    def
    x_train = x_train.to_numpy(dtype=('float32'))
    x_test = x_test.to_numpy(dtype=('float32'))
    y_train = np_utils.to_categorical(y_train, 2)
    y_test = np_utils.to_categorical(y_test, 2)

    NB_CLASSES = y_train.shape[1]
    INPUT_SHAPE = (x_train.shape[1],)
    model = Sequential()
    model.add((32, input_shape=INPUT_SHAPE))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add((16))
    model.add(Activation('relu'))
    model.add((8))
    model.add(Activation('relu'))
    model.add((NB_CLASSES))
    model.add(Activation('softmax'))
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['Precision', 'Recall'])
    EPOCHS = 50
    history = model.fit(x_train, y_train, batch_size=32, epochs=EPOCHS, verbose=1,
                        validation_data=(x_test, y_test))
    f1_score_list_train = []
    f1_score_list_test = []
    history_dict = history.history
    print(history_dict.keys())
    for i in range(EPOCHS):
        f1_score_list_train.append((2 * history.history['precision'][i] * history.history['recall'][i]) / (
                    history.history['precision'][i] + history.history['recall'][i]))
        f1_score_list_test.append((2 * history.history['val_precision'][i] * history.history['val_recall'][i]) / (
                    history.history['val_precision'][i] + history.history['val_recall'][i]))

    y_pred = model.predict(x_test)

    f = plt.figure()
    s1 = f.add_subplot(1, 1, 1)
    s1.grid(True)
    f.clf()

    epochs_range = np.arange(1, 51, 1)

    plt.title("Зависимость F-меры от эпохи (epochs range)")
    plt.plot(epochs_range, f1_score_list_train, epochs_range, f1_score_list_test)
    plt.ylabel("F-мера")
    plt.xlabel("Эпоха")
    plt.grid(True)

    plt.show()


data = p1_read('C:/Users/Сергей/PycharmProjects/AI_Lab1/input/income.csv')
#p2_sum(data)
#X = p3_token(data)
x_train, x_test, y_train, y_test = p3_split(data)
#p4_Tree(x_train, x_test, y_train, y_test)
#p5_Forest(x_train, x_test, y_train, y_test)
#p6_Boosting(x_train, x_test, y_train, y_test)



