from sklearn.preprocessing import OneHotEncoder

from src.data_processing.data_generator import data_generator
from src.features.build_features import build_features
from src.features.fit_models import fit_models
from src.predict.prediction import make_predict

# дает доступ ко всем функциям кода

i = input('Введите команду (help для справки): ')
if i == 'help':
    print('генератор данных - генерация датасета, по умолчанию используется - https://docs.google.com/spreadsheets/d/18oQZQ9asaFbT5TTO8qXF4ZIA_qB4SiIMCHJ2UcnSVWo/edit#gid=0\n',
          'обработка данных - OheHot-кодировка категориальных переменных, разделение датасета на тренировочную и обучающую выборки\n',
          'обучение моделей - обучение моделей\n',
          'прогноз - сделать прогноз')
    
elif i == 'генератор данных':
    # получаем ссылку на датасет
    k = input('Введите ссылку на датасет, либо оставьте поле пустым для использования данных по умолчанию: ')
    if k == '':
        link = r'https://docs.google.com/spreadsheets/d/18oQZQ9asaFbT5TTO8qXF4ZIA_qB4SiIMCHJ2UcnSVWo/export?format=csv'
    else: link = k
    data_generator(link)

elif i == 'обработка данных':
    # получаем список категориальных переменных
    k = input('Введите категориальные переменные через запятую, либо оставьте поле пустым для использования данных по умолчанию: ')
    if k == '':
        categorical_columns = ['Пол', 'Образование', 'Семейный статус', 'Уровень дохода', 'Категория карты']
    else: categorical_columns = list(k)
    
    # получаем целевую параметр
    l = input('Введите название целевой параметр, либо оставьте поле пустым для использования данных по умолчанию: ')
    if l == '':
        target = 'Статус клиента'
    else: target = l
    
    # создаем энкодер
    encoder = OneHotEncoder(sparse_output = False, drop = 'first')
    build_features(categorical_columns, target, encoder)

elif i == 'обучение моделей':
    fit_models()

elif i == 'прогноз':
    make_predict()

else: print('Ошибка ввода, обратитесь к справке')