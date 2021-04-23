import time
# Библиотека для перевода слов. Неофициальная googletrans работала с перебоями, а google.cloud слишком страшная
from deep_translator import GoogleTranslator
from main import symptoms
import numpy as np

# Работает за 253.854773 минуты

split1 = int(len(symptoms) / 4)
split2 = int(len(symptoms) / 2)
split3 = int(len(symptoms) / 2 + len(symptoms) / 4)

start = time.time()
symptoms_part_1 = symptoms[:split1]
symptoms_part_2 = symptoms[split1:split2]
symptoms_part_3 = symptoms[split2:split3]
symptoms_part_4 = symptoms[split3:]

# В какой-то момент вылетела ошибка, поэтому разбил массив на 4 части <5000 элементов.
# Возможно, ошибка была связана с другим

translated = list()
for symptom in symptoms_part_1:
    translated.append(GoogleTranslator(source='en', target='ru').translate(symptom))
for symptom in symptoms_part_2:
    translated.append(GoogleTranslator(source='en', target='ru').translate(symptom))
for symptom in symptoms_part_3:
    translated.append(GoogleTranslator(source='en', target='ru').translate(symptom))
for symptom in symptoms_part_4:
    translated.append(GoogleTranslator(source='en', target='ru').translate(symptom))

# Добавить запись в файл.csv

symptoms_ru = np.array(translated)
end = time.time()
print('%s - %s' % (len(symptoms), len(symptoms_ru)))
print('%1f min' % ((end - start) / 60.0))
