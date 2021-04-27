import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deep_translator import GoogleTranslator


# Создание словаря {'Болезнь' : ['Симптомы']}
def create_dictionary(dataframe, sort):
    disease_dict = dict()

    for index, row in dataframe.iterrows():
        if row['disease'] in disease_dict:
            disease_dict[row['disease']].append(row['symptom'])
        else:
            disease_dict[row['disease']] = [row['symptom']]

    if sort:
        sorted_dictionary = dict(sorted(disease_dict.items(), key=lambda x: len(x[1])))
        return sorted_dictionary
    else:
        return disease_dict


# Вывод симптомов по названию болезни
def print_symptoms(dictionary):
    disease = input('Введите название болезни: ')
    while disease != '0':
        print('Симптомы %s : %s' % (disease, dictionary[disease]))
        disease = input('Введите название болезни: ')


# Удаление болезней с количеством симптомов меньше чем min_number_of_symptoms из словаря
def remove_diseases(min_number_of_symptoms, dictionary):
    new_array = []
    for dis in dictionary.items():
        if len(dis[1]) < min_number_of_symptoms:
            continue
        for item in dis[1]:
            new_array.append([dis[0], item])

    disease_dataset = np.array(new_array)
    another_df = pd.DataFrame(data=disease_dataset)
    another_df.columns = ['disease', 'symptom']
    # Запись в csv файл
    another_df.to_csv('new_syditriage.csv', index=False)
    return another_df


# TODO: все болезни с одинаковыми симптотами объединить в одну, разделив '; '
def unite_diseases_with_same_symptoms(dictionary):
    # Менять и удалять ключи в цикле по словарю нельзя, поэтому создать новый словарь и вернуть его
    dict_sorted = dict(dictionary)
    for sym in dict_sorted.keys():
        dict_sorted[sym].sort()

    # Выводит болезни с одинаковыми симптомами
    for disease in dict_sorted:
        for another_disease in dict_sorted:
            if disease == another_disease:
                continue
            elif dict_sorted[disease] == dict_sorted[another_disease]:
                print('%s = %s' % (disease, another_disease))
                # Ошибка, если здесь изменить/удалить ключ:
                # RuntimeError: dictionary keys changed during iteration

                # Здесь что-то делать с новым словарем

    # Возвращает неупорядоченный словарь с упорядоченными симптомами
    return dict_sorted


# Загрузка датасета
df = pd.read_csv('new_syditriage.csv')

disease_dictionary = create_dictionary(df, sort=False)

print('Количество болезней: %s' % len(disease_dictionary.keys()))
print('Количество симптомов: %s' % df['symptom'].nunique())

symptoms = df['symptom'].unique()
diseases = df['disease'].unique()

# new_df = remove_diseases(3, disease_dictionary)
# new_dict = create_dictionary(new_df, sort=True)
#
# print('Количество болезней: %s' % len(new_dict.keys()))
# print('Количество симптомов: %s' % new_df['symptom'].nunique())
#
# symptoms2 = new_df['symptom'].unique()
# diseases2 = new_df['disease'].unique()
