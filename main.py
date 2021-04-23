import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deep_translator import GoogleTranslator

# Загрузка датасета
df = pd.read_csv('syditriage.csv')

# Словарь {'Болезнь' : ['Симптомы']}
disease_dictionary = dict()

for index, row in df.iterrows():
    if row['disease'] in disease_dictionary:
        disease_dictionary[row['disease']].append(row['symptom'])
    else:
        disease_dictionary[row['disease']] = [row['symptom']]

print('Количество болезней: %s' % len(disease_dictionary.keys()))
print('Количество симптомов: %s' % df['symptom'].nunique())

disease = input('Введите название болезни: ')
while disease != '0':
    print('Симптомы %s : %s' % (disease, disease_dictionary[disease]))
    disease = input('Введите название болезни: ')

symptoms = df['symptom'].unique()
diseases = df['disease'].unique()
