import numpy as np
import pandas as pd
from sklearn import tree
from main import diseases
from input_symptoms import create_object
import time
import joblib

# tree_classifier = tree.DecisionTreeClassifier()  # random_state=0
#
# # Классы
# target = diseases
#
# # Признаки
# # 55.811129 сек
# start = time.time()
# df_data = pd.read_csv('syditraige_data\\new_sd.csv', header=None, index_col=False)
# end = time.time()
# print('read_csv: %6f сек' % (end - start))
#
# data = df_data.to_numpy()
#
# # Создание дерева
# # 451.672804 сек
# start = time.time()
# tree_classifier = tree_classifier.fit(data, target)
# end = time.time()
# print('tree_classifier.fit: %6f сек' % (end - start))
#
# joblib.dump(tree_classifier, 'tree_classifier.pkl')

tree_classifier = joblib.load('tree_classifier.pkl')

ans = input('Ввести симптомы или использовать объекты из обучающей выборки? (1/0)')
# Объект с введенными симптомами
if ans == '1':
    con = ''
    while con != 'exit':
        obj = create_object(out=False)
        result = tree_classifier.predict(obj)
        print('Предсказано: %s' % result[0])
        con = input('Продолжить? (exit)')

# Объект из датасета со всеми симптотами
if ans == '0':
    i = input('Введите индекс болезни: ')
    while i != 'exit':
        obj = [data[int(i)]]
        result = tree_classifier.predict(obj)
        print('Предсказано: %s\tОжидалось: %s' % (result[0], target[int(i)]))
        i = input('Введите индекс болезни: ')

# Выдает ошибку, потому что такое дерево (которое выглядит как линейный список, по крайней мере в начале)
# даже в 10 экранов не влезет. Содержимое сгенерированного файла скопировать в http://webgraphviz.com/
tree.export_graphviz(tree_classifier, out_file='../misc/syditriage_tree.dot')
