import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

from new_accuracy_test import calculate_accuracy
from new_accuracy_test import data_target
from new_input_symptoms import create_object

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Загрузка датасета
df = pd.read_csv('new_data/large_data.csv')

# Список симптомов
symptoms = list(df[df.columns[:20]].columns.values)
diseases = list(df[df.columns[20]].unique())

# Разделение датасета на обучающую и тестовую выборки
train, test = train_test_split(df, test_size=0.2, random_state=0)

# Разделение обучающей выборки на N частей, которые будут использоваться для обучения разных алгоритмов
NUMBER_OF_CLASSIFIERS = 4
SAMPLE_SIZE = int(len(train) / NUMBER_OF_CLASSIFIERS)

duplicates = True
if duplicates:
    # С повторениями
    train_parts = []
    for i in range(NUMBER_OF_CLASSIFIERS):
        train_parts.append(train.sample(SAMPLE_SIZE))
else:
    # Без повторений
    shuffled_train = train.sample(frac=1, random_state=0)
    train_parts = np.array_split(shuffled_train, NUMBER_OF_CLASSIFIERS)

# Разделение частей на обучающие примеры и метки класса
classifiers_data_target = []
for i in range(NUMBER_OF_CLASSIFIERS):
    classifiers_data_target.append(data_target(train_parts[i]))

test_data, test_target = data_target(test)

# Инициализация классификаторов
random_forest_classifier = RandomForestClassifier(random_state=0, max_depth=12, criterion='entropy', max_leaf_nodes=28)
dtree_classifier = DecisionTreeClassifier(random_state=0, max_depth=12, criterion='entropy', max_leaf_nodes=28)
bayes_classifier = GaussianNB()
neighbors_classifier = KNeighborsClassifier(n_neighbors=7, metric='minkowski')

# Массив с моделями
classifiers = [dtree_classifier,
               random_forest_classifier,
               bayes_classifier,
               neighbors_classifier]

classifier_labels = ['Дерево решений',
                     'Случайный лес',
                     'Наивный байес',
                     'K ближайших соседей']

# Обучение моделей
for index, classifier in enumerate(classifiers):
    classifiers[index] = classifier.fit(classifiers_data_target[index][0], classifiers_data_target[index][1])

# Объект с введенными симптомами
con = ''
while con != 'n':
    print(symptoms)
    obj = create_object(out=False)
    res = [[],
           [],
           [],
           []]

    for index, classifier in enumerate(classifiers):
        result = classifier.predict(obj)
        result_probability = classifier.predict_proba(obj)

        res[index].append(result[0])
        print('%s: %s' % (classifier_labels[index], result[0]))
        for i in range(4):
            res[index].append(result_probability[0][i] * 100.0)

    results = np.array(res)

    print("{:<25} {:<10} {:<10} {:<10} {:<10}".format('Классификатор', diseases[0], diseases[1], diseases[2],
                                                      diseases[3]))
    for i in range(len(classifiers)):
        print("{:<25} {:<10.1f} {:<10.1f} {:<10.1f} {:<10.1f}".format(classifier_labels[i], float(results[i][1]),
                                                                      float(results[i][2]), float(results[i][3]),
                                                                      float(results[i][4])))

    print('\nФинальное решение:')
    proba_temp = results[:, 1:].astype(float)
    result_temp = results[:, 0]

    # Самая популярная метка класса и усредненные вероятности
    results_mean = proba_temp.mean(axis=0)
    result_disease = stats.mode(result_temp)

    print("{:<25} {:<10.1f} {:<10.1f} {:<10.1f} {:<10.1f}".format(result_disease[0][0], results_mean[0],
                                                                  results_mean[1], results_mean[2], results_mean[3]))

    con = input('Продолжить? (y/n)')

# Точность
print('Точность на тестовой выборке:')
for index, classifier in enumerate(classifiers):
    accuracy = calculate_accuracy(classifier, test_data, test_target)
    print('%s: %.1f%%' % (classifier_labels[index], accuracy))
