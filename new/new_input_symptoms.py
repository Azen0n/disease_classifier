import numpy as np
import pandas as pd
from new_train_tree import symptoms


# Флаг out для вывода симптомов созданного объекта
def create_object(out):
    obj = np.zeros(shape=(1, len(symptoms)))

    i = input('Введите симптом: ')
    while i != '0':
        for index, symptom in enumerate(symptoms):
            if symptom == i:
                obj[0][index] = 1
                break
        i = input('Введите симптом: ')

    if out:
        print('Симптомы созданного объекта:')
        count = 1
        for index, symptom in enumerate(obj[0]):
            if symptom == 1:
                print('%s. %s' % (count, symptoms[index]))
                count += 1

    return obj


if __name__ == "__main__":
    obj1 = create_object(out=True)
