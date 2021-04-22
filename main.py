import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Загрузка датасета
df = pd.read_csv('syditriage.csv')

# Создание словаря
disease_dictionary = dict()

for index, row in df.iterrows():
    if row['disease'] in disease_dictionary:
        disease_dictionary[row['disease']].append(row['symptom'])
    else:
        disease_dictionary[row['disease']] = [row['symptom']]

print('Количество ключей: %s' % len(disease_dictionary.keys()))
print('Количество симптомов: %s' % df['symptom'].nunique())

disease = input('Введите название болезни: ')
while disease != '0':
    print('Симптомы %s : %s' % (disease, disease_dictionary[disease]))
    disease = input('Введите название болезни: ')
