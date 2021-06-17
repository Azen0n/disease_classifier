import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from new_accuracy_test import data_target, calculate_accuracy
from new_input_symptoms import create_object
from sklearn.model_selection import train_test_split
import random
import time
import joblib

forest_classifier = RandomForestClassifier(random_state=0)

df = pd.read_csv('new_data/large_data.csv')
data, target = data_target(df)

train, test = train_test_split(df, test_size=0.2, random_state=0)

train_data, train_target = data_target(train)
test_data, test_target = data_target(test)

forest_classifier = forest_classifier.fit(train_data, train_target)

accuracy = calculate_accuracy(forest_classifier, test_data, test_target)
print(accuracy)

symptoms = list(df[df.columns[:20]].columns.values)
diseases = list(df[df.columns[20]].unique())

print(symptoms[:10])
print(symptoms[10:20])

# Объект с введенными симптомами
con = ''
while con != 'нет':
    obj = create_object(out=False)
    result = forest_classifier.predict(obj)
    result_probability = forest_classifier.predict_proba(obj)

    print('Предсказано: %s' % result[0])
    print('Вероятность:')
    for i in range(4):
        print('%s: %.1f%%' % (diseases[i], (result_probability[0][i] * 100.0)))

    con = input('Продолжить? (да/нет)')
