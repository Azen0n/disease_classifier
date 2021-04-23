from main import diseases, symptoms, disease_dictionary
import numpy as np

# За 5-10 минут генерирует огромную матрицу
# Матрица размером [7944 × 18755] ([болезни × симптомы]). 1, если симптом есть, иначе 0
# Максимум симптомов на болезнь ~20, поэтому в матрице получается 18К+ нулей.
# Можно сравнивать строки, но это будет долго и мучительно

sd = np.zeros(shape=(len(diseases), len(symptoms)))

j = 0
for i, dis in enumerate(diseases):
    for sym in disease_dictionary[dis]:
        for s in symptoms:
            if sym == s:
                sd[i][j] = 1
            j += 1
        j = 0

# Можно тоже добавить запись в файл, но я его уже эскпортировал с помощью среды
