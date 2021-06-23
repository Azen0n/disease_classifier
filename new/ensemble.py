import time

import numpy as np
import pandas as pd
from scipy import stats
import joblib
from sklearn.model_selection import train_test_split

from new_accuracy_test import calculate_accuracy, calculate_accuracy_ensemble, my_ensemble
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

duplicates = False
if duplicates:
    # С повторениями: N выборок с одного и того же датасета
    train_parts = []
    for i in range(NUMBER_OF_CLASSIFIERS):
        train_parts.append(train.sample(SAMPLE_SIZE))
else:
    # Без повторений: весь датасет делится на N частей
    shuffled_train = train.sample(frac=1, random_state=0)
    train_parts = np.array_split(shuffled_train, NUMBER_OF_CLASSIFIERS)


# Модели обучаются на N - 1 из N частей, последняя часть для проверки точности
# Что-то вроде кроссвалидации, наверное?
# Этот некрасивый код соединяет части, полученные в прошлом пункте
temp_array = [[], [], [], []]
test_for_classifiers = []
for j in range(3, -1, -1):
    temp_df = pd.DataFrame()
    for i in range(4):
        if i == j:
            # Добавляем пропущенную часть в массив с датафреймами для проверки точности
            test_for_classifiers.append(data_target(train_parts[j]))
            continue
        temp_df = temp_df.append(train_parts[i])
    temp_array[-j - 1].append(temp_df)

# Разделение частей на обучающие примеры и метки класса
classifiers_data_target = []
for i in range(NUMBER_OF_CLASSIFIERS):
    classifiers_data_target.append(data_target(temp_array[i][0]))

test_data, test_target = data_target(test)

# # Инициализация классификаторов
# random_forest_classifier = RandomForestClassifier(random_state=0, max_depth=12, criterion='entropy', max_leaf_nodes=28)
# dtree_classifier = DecisionTreeClassifier(random_state=0, max_depth=12, criterion='entropy', max_leaf_nodes=28)
# bayes_classifier = GaussianNB()
# neighbors_classifier = KNeighborsClassifier(n_neighbors=7, metric='minkowski')

# Загрузка моделей
dtree_classifier = joblib.load('model1.pkl')
random_forest_classifier = joblib.load('model2.pkl')
bayes_classifier = joblib.load('model3.pkl')
neighbors_classifier = joblib.load('model4.pkl')

# Массив с моделями
classifiers = [dtree_classifier,
               random_forest_classifier,
               bayes_classifier,
               neighbors_classifier]

classifier_labels = ['Дерево решений',
                     'Случайный лес',
                     'Наивный байес',
                     'K ближайших соседей']

# # Обучение моделей
# for index, classifier in enumerate(classifiers):
#     classifiers[index] = classifier.fit(classifiers_data_target[index][0], classifiers_data_target[index][1])
#     # Сохранение модели
#     joblib.dump(classifiers[index], 'model' + str(index + 1) + '.pkl')


# Объект с введенными симптомами
con = ''
while con != 'n':
    print(symptoms)
    obj = create_object(out=False)

    results, results_mean, result_disease = my_ensemble(obj, classifiers)

    print("{:<25} {:<10} {:<10} {:<10} {:<10}".format('Классификатор', diseases[0], diseases[1], diseases[2],
                                                      diseases[3]))
    for i in range(len(classifiers)):
        print("{:<25} {:<10.1f} {:<10.1f} {:<10.1f} {:<10.1f}".format(classifier_labels[i], float(results[i][1]),
                                                                      float(results[i][2]), float(results[i][3]),
                                                                      float(results[i][4])))

    print("{:<25} {:<10.1f} {:<10.1f} {:<10.1f} {:<10.1f}".format(result_disease, results_mean[0],
                                                                  results_mean[1], results_mean[2], results_mean[3]))

    con = input('Продолжить? (y/n)')


# Точность алгоритмов
print('Точность моделей на тестовой выборке:')
for index, classifier in enumerate(classifiers):
    start = time.time()
    accuracy = calculate_accuracy(classifier, test_for_classifiers[index][0], test_for_classifiers[index][1])
    end = time.time()
    print('%s: %.1f%% (%.2f сек.)' % (classifier_labels[index], accuracy, end - start))

# Точность ансамбля (на тестовой выборке, которую оставили в самом начале)
print('\nТочность ансамбля на тестовой выборке:')
start = time.time()
accuracy = calculate_accuracy_ensemble(classifiers, test_data, test_target)
end = time.time()
print('Ансамбль алгоритмов: %.1f%% (%.2f сек.)' % (accuracy, end - start))
