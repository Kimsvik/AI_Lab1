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
from dython import nominal


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
    new_df = df[numeric_data].copy()

    nominal.associations(df[categorical_data], nominal_columns='all')
    nominal.associations(df[numeric_data])
    sns.heatmap(df.corr())
    plt.show()

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
        print(dict_clmn)
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
    f.set_size_inches(16, 5)
    s1 = f.add_subplot(1, 1, 1)
    s1.grid(True)
    f.clf()

    plt.title("Зависимость F-меры от глубины")
    plt.plot(depth_range, train_score, depth_range, test_score)
    plt.ylabel("F-мера")
    plt.xlabel("Глубина")
    plt.grid(True)

    plt.show()


data = p1_read('input/income.csv')

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
p2_14(data)

#X = p3_token(data)
# x_train, x_test, y_train, y_test = p3_split(data)
# print(y_test)
# p4_Tree(x_train, x_test, y_train, y_test)




