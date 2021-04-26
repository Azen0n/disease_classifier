import numpy as np
import pandas as pd
from sklearn import tree
from main import diseases

tree_classifier = tree.DecisionTreeClassifier()

# Классы
target = diseases

# Признаки
df_data = pd.read_csv('new_sd.csv', header=None, index_col=False)
data = df_data.to_numpy()

tree_classifier = tree_classifier.fit(data, target)

i = input('Введите что-нибудь: ')
while i != 'exit':
    obj = [data[int(i)]]
    result = tree_classifier.predict(obj)
    print('Предсказано: %s\tОжидалось: %s' % (result[0], target[int(i)]))
    i = input('Введите что-нибудь: ')

# Выдает ошибку, потому что такое дерево (которое выглядит как линейный список, по крайней мере в начале)
# даже в 10 экранов не влезет. Содержимое сгенерированного файла скопировать в http://webgraphviz.com/
tree.export_graphviz(tree_classifier, out_file='tree123.dot')
