import pandas as pd

def data_generator(link):
    """
    Функция получает ссылку на датасет,
    после чего генерирует scv-файл в /data/raw/
    """
    pd.read_csv(link).to_csv('data/raw/data_raw.csv', index = False)
    print('Генерация данных выполнена успешно')