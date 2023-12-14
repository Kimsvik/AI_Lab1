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
from tensorflow.keras.datasets import mnist
from dython import nominal

from sklearn.metrics import f1_score

"ПУНКТ 1"
def p1_read(path):
    data_file = pd.read_csv(path)
    return data_file


"ПУНКТ 2"
def p2_1_null_counter(df):

    for col in df.columns:
        print(col)
    print('\n')

    columns = list(df)
    for i in columns:
        print(df[df[i] == ' ?'][i].count())
    print('\n')









    # df.replace('?', np.NaN, inplace=True)
    #
    # categorical = [var for var in df.columns if data[var].dtype == 'O']
    # df[categorical].head()

def p2_2_workclass(df):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax = df['workclass'].hist(edgecolor='black', color='lightsteelblue')
    plt.grid(False)
    plt.xticks(rotation=15)
    plt.ylabel("count")

    for label in ax.containers:
        ax.bar_label(label)

    plt.show()

def p2_3_workclass_hist(df):
    ax = plt.subplots(figsize=(7, 5))
    ax = sns.countplot(x="income", hue="sex", data=df, palette="Set1")
    ax.set_title("Frequency distribution of income variable sex")

    for label in ax.containers:
        ax.bar_label(label)

    plt.show()

def p2_4_race(df):
    ax = plt.subplots(figsize=(7, 5))
    ax = sns.countplot(x="income", hue="race", data=df, palette="Set1")
    ax.set_title("Frequency distribution of income variable race")

    for label in ax.containers:
        ax.bar_label(label)

    plt.show()

def p2_5_income_by_workclass(df):
    ax = plt.subplots(figsize=(10, 7))
    ax = sns.countplot(x="workclass", hue="income", data=df, palette="Set1")
    ax.set_title("Frequency distribution of income variable race")

    for label in ax.containers:
        ax.bar_label(label)

    plt.show()

def p2_6_workclass_by_sex(df):
    ax = plt.subplots(figsize=(10, 7))
    ax = sns.countplot(x="workclass", hue="sex", data=df, palette="Set1")
    ax.set_title("Frequency distribution of workclass variable sex")

    for label in ax.containers:
        ax.bar_label(label)

    plt.show()

def p2_7_age(df):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax = df['age'].hist(edgecolor='black', color='lightsteelblue')
    plt.grid(False)
    plt.xticks(rotation=0)
    plt.ylabel("count")
    plt.xlabel("age")

    for label in ax.containers:
        ax.bar_label(label)

    plt.show()

def p2_8_age_box_plot(df):
    f, ax = plt.subplots(figsize=(5, 7))
    x = df['age']
    ax = sns.boxplot(x)
    ax.set_title("Visualize outliers in age variable")
    plt.show()

def p2_9_income_age(df):
    f, ax = plt.subplots(figsize=(10, 8))
    ax = sns.boxplot(x="income", y="age", data=df)
    ax.set_title("Visualize income wrt age variable")
    plt.show()

def p2_10(df):
    f, ax = plt.subplots(figsize=(10, 8))
    ax = sns.boxplot(x="income", y="age", hue="sex", data=df)
    ax.set_title("Visualize income wrt age and sex variable")
    ax.legend(loc='upper right')
    plt.show()

def p2_11(df):
    f, ax = plt.subplots(figsize=(10, 8))
    ax = sns.boxplot(x="race", y="age", data=df)
    ax.set_title("Visualize income wrt age and sex variable")
    ax.legend(loc='upper right')
    plt.show()

def p2_12_heat_map(df):
    numeric_data = df.select_dtypes(include=[np.number]).columns
    categorical_data = df.select_dtypes(exclude=[np.number]).columns

    # print(numeric_data)
    # print(categorical_data)
    # new_df = df[numeric_data].copy()

    nominal.associations(df, nominal_columns='all', annot=False, cmap='coolwarm')
    # nominal.associations(df[numeric_data])
    # sns.heatmap(df.corr(), cmap='coolwarm')
    # plt.show()

    # new_df.corr().style.format("{:.4}").background_gradient(cmap=plt.get_cmap('coolwarm'), axis=1)
    # sns.heatmap(new_df.corr())
    # plt.show()

def p2_13(df):
    df.replace(' ?', np.NaN, inplace=True)

def p2_14(df):
    numeric_data = df.select_dtypes(include=[np.number]).columns
    categorical_data = df.select_dtypes(exclude=[np.number]).columns

    print('\nnumeric data:\n')
    for name in numeric_data:
        print(name)

    print('\ncategorical data:\n')
    for name in categorical_data:
        print(name)

    # print(numeric_data)
    # print(categorical_data)


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
"ПУНКТ 4"
def p4_Tree(x_train, x_test, y_train, y_test):
    depth_range = np.arange(1, 20, 1)
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
    plt.legend(['Тренировочная выборка', 'Тестовая выборка'])
    plt.ylabel("F-мера")
    plt.xlabel("Глубина")
    plt.grid(True)

    """матрица ошибок решающего дерева"""
    confusion_matr = confusion_matrix(y_test, model.predict(x_test))
    conf_frame = pd.DataFrame(data=confusion_matr, columns=['predicted <=50k', 'predicted >50k'],
                              index=['actual <=50k', 'actual >50k'])
    print('\nconfusion frame\n')
    print(conf_frame)

    plt.show()


"ПУНКТ 5"
def p5_Forest(x_train, x_test, y_train, y_test):
    max_tree = 20
    estimators_range = np.arange(1, max_tree, 1)
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
    plt.legend(['Тренировочная выборка', 'Тестовая выборка'])
    plt.grid(True)
    plt.xlim(1, max_tree-1)

    """матрица ошибок случайный лес"""
    confusion_matr = confusion_matrix(y_test, model.predict(x_test))
    conf_frame = pd.DataFrame(data=confusion_matr, columns=['predicted <=50k', 'predicted >50k'],
                              index=['actual <=50k', 'actual >50k'])
    print('\nconfusion frame\n')
    print(conf_frame)

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
    x_train = x_train.to_numpy(dtype=('float32'))
    x_test = x_test.to_numpy(dtype=('float32'))
    y_train = np_utils.to_categorical(y_train, 2)
    y_test = np_utils.to_categorical(y_test, 2)

    NB_CLASSES = y_train.shape[1]
    INPUT_SHAPE = (x_train.shape[1],)
    model = Sequential()
    model.add(Dense(32, input_shape=INPUT_SHAPE))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(Dense(NB_CLASSES))
    model.add(Activation('softmax'))
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['Precision', 'Recall'])
    EPOCHS = 30
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

    epochs_range = np.arange(1, 31, 1)

    plt.title("Зависимость F-меры от эпохи (epochs range)")
    plt.plot(epochs_range, f1_score_list_train, epochs_range, f1_score_list_test)
    plt.ylabel("F-мера")
    plt.xlabel("Эпоха")
    plt.grid(True)

    plt.show()


"ПУНКТ 8"
def p8_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # pick a sample to plot
    sample = 1
    image = x_train[sample]
    # plot the sample
    fig = plt.figure
    plt.imshow(image, cmap='gray')
    plt.show()

    x_train = x_train.reshape(x_train.shape[0], 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)

    boost_estimators_range = np.arange(10, 40, 10)
    boost_train_score = []
    boost_test_score = []
    boost_optimal_estimator = 0
    boost_useless_var = 0

    for i in range(len(boost_estimators_range)):
        model = CatBoostClassifier(n_estimators=boost_estimators_range[i], task_type="GPU", devices='0:1')
        model.fit(x_train, y_train)
        boost_train_score.append(f1_score(y_train, model.predict(x_train), average='micro'))
        boost_test_score.append(f1_score(y_test, model.predict(x_test), average='micro'))
        if (boost_test_score[i] > boost_useless_var):
            boost_useless_var = boost_test_score[i]
            boost_optimal_estimator = boost_estimators_range[i]

    print("Оптимальное количество - ", boost_optimal_estimator)

    f = plt.figure()
    s1 = f.add_subplot(1, 1, 1)
    s1.grid(True)
    f.clf()

    plt.title("Зависимость F-меры от глубины (boost estimators range)")
    plt.plot(boost_estimators_range, boost_train_score, boost_estimators_range, boost_test_score)
    plt.ylabel("F-мера")
    plt.xlabel("Количество деревьев")
    plt.grid(True)

    plt.show()

#data = p1_read('input/income.csv')
data = p1_read('C:/Users/Сергей/Desktop/МЭИ/М3/ИИ/lab/input/income.csv')

# p2_1_null_counter(data)
# p2_2_workclass(data)
# p2_3_workclass_hist(data)
# p2_4_race(data)
# p2_5_income_by_workclass(data)
# p2_6_workclass_by_sex(data)
# p2_7_age(data)
# p2_8_age_box_plot(data)
# p2_9_income_age(data)
# p2_10(data)
# p2_11(data)
# p2_12_heat_map(data)
# p2_13(data)
# p2_14(data)

#X = p3_token(data)
x_train, x_test, y_train, y_test = p3_split(data)
# print(y_test)
# p4_Tree(x_train, x_test, y_train, y_test)
p5_Forest(x_train, x_test, y_train, y_test)
#p6_Boosting(x_train, x_test, y_train, y_test)
#p7_Precept(x_train, x_test, y_train, y_test)
p8_mnist()
