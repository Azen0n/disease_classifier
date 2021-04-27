from main import diseases, symptoms, disease_dictionary
import numpy as np
import pandas as pd
import time

# Генерирует огромную матрицу размером [болезни × симптомы]. 1, если симптом есть, иначе 0
# Максимум симптомов на болезнь ~100, поэтому в матрице получается 18К+ нулей.

# Создание матрицы: 138.139293 сек.
start = time.time()
sd = np.zeros(shape=(len(diseases), len(symptoms)))

j = 0
for i, dis in enumerate(diseases):
    for sym in disease_dictionary[dis]:
        for s in symptoms:
            if sym == s:
                sd[i][j] = 1
            j += 1
        j = 0
end = time.time()
print('Создание матрицы: %6f сек.' % (end - start))

df = pd.DataFrame(data=sd)

# Запись в csv: 69.296059 сек.
# Файл весит 450 Мб
start = time.time()
df.to_csv('new_sd.csv', header=False, index=False)
end = time.time()
print('Запись в файл csv: %6f сек.' % (end - start))


# TODO: Добавить запись в txt файл без разделителей
def write_txt():
    pass


# TODO: Добавить чтение txt файла без разделителей
def read_txt():
    pass
