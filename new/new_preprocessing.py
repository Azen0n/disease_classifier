import numpy as np
import pandas as pd
from sklearn import tree
from new_input_symptoms import create_object
import joblib

# загрузка уже обученного дерева из файла
tree_classifier = joblib.load('new_train_tree_classifier.pkl')

# Список симптомов и болезней
df = pd.read_csv('new_data/large_data.csv')
symptoms = list(df[df.columns[:20]].columns.values)
diseases = list(df[df.columns[20]].unique())

# Объект с введенными симптомами
con = ''
while con != 'нет':
    obj = create_object(out=False)
    result = tree_classifier.predict(obj)
    result_probability = tree_classifier.predict_proba(obj)

    print('Предсказано: %s' % result[0])
    print('Вероятность:')
    for i in range(4):
        print('%s: %.1f%%' % (diseases[i], (result_probability[0][i] * 100.0)))

    con = input('Продолжить? (да/нет)')
