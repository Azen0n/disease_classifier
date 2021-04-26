import numpy as np
from sklearn import tree
from sklearn import datasets
import pydot

tree_classifier = tree.DecisionTreeClassifier()

names = ['Nausea', 'General feeling of unwellness', 'Anxiety', 'Eye pain', 'Tiredness']

data = [[1, 1, 1, 0, 0],
        [1, 0, 0, 1, 0],
        [0, 0, 1, 0, 1]]

target = ['Hypertension',
          'Migraine',
          'Insomnia']

tree_classifier = tree_classifier.fit(data, target)

result1 = tree_classifier.predict([[1, 0, 0, 0, 0]])
result2 = tree_classifier.predict([[1, 0, 0, 1, 0]])
result3 = tree_classifier.predict([[0, 0, 1, 0, 1]])

print(result1)
print(result2)
print(result3)

tree.export_graphviz(tree_classifier, out_file='tree.dot')
# Скопировать текст из файла в http://webgraphviz.com/
