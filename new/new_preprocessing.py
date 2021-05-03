import numpy as np
import pandas as pd
from sklearn import tree
from new_input_symptoms import create_object
import joblib

# загрузка уже обученного дерева из файла
tree_classifier = joblib.load('new_tree_classifier.pkl')

print('COUGH', 'MUSCLE_ACHES', 'TIREDNESS', 'SORE_THROAT', 'RUNNY_NOSE', 'STUFFY_NOSE', 'FEVER', 'NAUSEA', 'VOMITING',
      'DIARRHEA', 'SHORTNESS_OF_BREATH', 'DIFFICULTY_BREATHING', 'LOSS_OF_TASTE', 'LOSS_OF_SMELL', 'ITCHY_NOSE',
      'ITCHY_EYES', 'ITCHY_MOUTH', 'ITCHY_INNER_EAR', 'SNEEZING', 'PINK_EYE')
# Объект с введенными симптомами
con = ''
while con != 'нет':
    obj = create_object(out=False)
    result = tree_classifier.predict(obj)
    result2 = tree_classifier.predict_proba(obj)
    print('Предсказано: %s' % result[0])
    print('Вероятность:')
    print('ALLERGY: %.0f%%' % (result2[0][0] * 100.0))
    print('COLD: %.0f%%' % (result2[0][1] * 100.0))
    print('COVID: %.0f%%' % (result2[0][2] * 100.0))
    print('FLU: %.0f%%' % (result2[0][3] * 100.0))
    con = input('Продолжить? (да/нет)')

# Выдает ошибку, потому что такое дерево (которое выглядит как линейный список, по крайней мере в начале)
# даже в 10 экранов не влезет. Содержимое сгенерированного файла скопировать в http://webgraphviz.com/
tree.export_graphviz(tree_classifier, out_file='../misc/new_tree.dot')
