import numpy as np
import pandas as pd
from sklearn import tree
import joblib

flu_df = pd.read_csv('../new_data/large_data.csv')

temp_data = []
# Последняя колонка в датасете — название болезни
temp_target = flu_df[flu_df.columns[-1]]
# Список симптомов
symptoms = list(flu_df[flu_df.columns[:20]].columns.values)

for index, row in flu_df.iterrows():
    row_values = row[:20].tolist()
    temp_data.append(row_values)

data = np.array(temp_data)
target = np.array(temp_target)

tree_classifier = tree.DecisionTreeClassifier(random_state=0, max_depth=10)
tree_classifier = tree_classifier.fit(data, target)

# Сохранение модели в файл для быстрой загрузки
joblib.dump(tree_classifier, 'new_tree_classifier.pkl')
