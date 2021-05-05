import numpy as np
import pandas as pd
from sklearn import tree
from new_accuracy_test import data_target
import random
import joblib

df = pd.read_csv('new_data/large_data.csv')

# Список симптомов
symptoms = list(df[df.columns[:20]].columns.values)

data, target = data_target(df)

tree_classifier = tree.DecisionTreeClassifier(random_state=0, max_depth=12, criterion='entropy', max_leaf_nodes=12)
tree_classifier = tree_classifier.fit(data, target)

# Сохранение модели в файл для быстрой загрузки
joblib.dump(tree_classifier, 'new_tree_classifier.pkl')
