import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import random
import time
import joblib


# Разбиение датафрейма на обучающую и тестовую выборки
# Работает за 370 секунд, когда функция из sklearn за 0.01
def split_train_test(dataframe, test_size):
    train = pd.DataFrame(columns=dataframe.columns)
    test = pd.DataFrame(columns=dataframe.columns)
    size = int(test_size * len(dataframe.index))
    test_index = random.sample(range(len(dataframe.index)), size)

    for index, row in dataframe.iterrows():
        if index in test_index:
            test = test.append(row, ignore_index=True)
            continue
        train = train.append(row, ignore_index=True)

    return train, test


# Возвращает обучающую выборку и классы для алгоритма + список симптомов
def data_target(dataframe):
    temp_data = []
    # Последняя колонка в датасете — название болезни
    temp_target = dataframe[dataframe.columns[-1]]

    for index, row in dataframe.iterrows():
        row_values = row[:20].tolist()
        temp_data.append(row_values)

    data = np.array(temp_data)
    target = np.array(temp_target)

    return data, target


# Проверка точности модели на тестовой выборке
def calculate_accuracy(classifier, test_data, test_target):
    correct = len(test_data)
    for i in range(len(test_data)):
        result = classifier.predict([test_data[i]])
        if result != test_target[i]:
            correct -= 1
    accuracy = correct * 100.0 / len(test_data)

    return accuracy


if __name__ == "__main__":
    df = pd.read_csv('new_data/large_data.csv')

    # Список симптомов
    symptoms = list(df[df.columns[:20]].columns.values)

    # train, test = split_train_test(df, test_size=0.2)
    train, test = train_test_split(df, test_size=0.2, random_state=0)

    train_data, train_target = data_target(train)

    test_data, test_target = data_target(test)

    # Лучшими оказались параметры criterion='entropy' и max_depth=12 (точность 92.70048%)
    def check_max_depth_criterion():
        for j in range(2):
            crit = 'gini'
            if j == 1:
                crit = 'entropy'

            for i in range(1, 20):
                tree_classifier = tree.DecisionTreeClassifier(random_state=0, max_depth=i, criterion=crit)
                tree_classifier = tree_classifier.fit(train_data, train_target)
                accuracy = calculate_accuracy(tree_classifier, test_data, test_target)
                print('max_depth=%s\t\tcriterion:%s\t\tAccuracy: %.5f%%' % (i, tree_classifier.criterion, accuracy))
            print()

    # При max_leaf_nodes от 25 до 31 точность максимальная (93.07165%), возьмем значение 28
    def check_max_leaf_nodes():
        for i in range(24, 33):
            start = time.time()
            tree_classifier = tree.DecisionTreeClassifier(random_state=0, max_depth=12, criterion='entropy',
                                                          max_leaf_nodes=i)
            tree_classifier = tree_classifier.fit(train_data, train_target)
            accuracy = calculate_accuracy(tree_classifier, test_data, test_target)
            accuracy2 = calculate_accuracy(tree_classifier, train_data, train_target)
            end = time.time()
            print(
                'max_leaf_nodes=%s\t\tAccuracy обуч.: %.50f%%\t\tTime elapsed: %.25fs' % (i, accuracy2, (end - start)))
            print('max_leaf_nodes=%s\t\tAccuracy тест.: %.50f%%\t\tTime elapsed: %.25fs' % (i, accuracy, (end - start)))


    # check_max_depth_criterion()
    # check_max_leaf_nodes()

    tree_classifier = tree.DecisionTreeClassifier(random_state=0, max_depth=12, criterion='entropy', max_leaf_nodes=28)
    tree_classifier = tree_classifier.fit(train_data, train_target)

    random_forest_classifier = ensemble.RandomForestClassifier(random_state=0)
    random_forest_classifier = random_forest_classifier.fit(train_data, train_target)

    accuracy = calculate_accuracy(random_forest_classifier, test_data, test_target)
    print(accuracy)

    # Экспорт красивого графика (https://dreampuf.github.io/GraphvizOnline)
    # tree.export_graphviz(tree_classifier,
    #                      feature_names=list(df[df.columns[:20]].columns.values),
    #                      class_names=list(df[df.columns[20]].unique()),
    #                      out_file='misc/new_test_tree.dot')

    # Сохранение модели в файл для быстрой загрузки
    # joblib.dump(tree_classifier, 'new_train_tree_classifier.pkl')
