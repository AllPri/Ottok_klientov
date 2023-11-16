import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def build_features(categorical_columns, target, encoder):
    
    """
    Функция принимает категориальные переменные, целевой признак и энкодер.
    По завершению работы в data/processed/ создаются тестовые и тренеровочные данные.
    Промежуточные данные сохраняются в data/interim/ 
    """

    encoder = OneHotEncoder(sparse_output = False, drop = 'first')

    # преобразуем категориальные данные в формат 0/1
    df = pd.read_csv('data/raw/data_raw.csv')

    # создаем и применяем кодировщик
    encoded_data = encoder.fit_transform(df[categorical_columns])

    # получаем имена столбцов после кодирования
    encoded_columns = encoder.get_feature_names_out(input_features = categorical_columns)

    # новый DataFrame с закодированными признаками
    encoded_df = pd.DataFrame(encoded_data, columns = encoded_columns).astype(bool)

    # удаляем оригинальные категориальные признаки из исходного DataFrame
    df.drop(categorical_columns, axis = 1, inplace = True)

    # объединяем исходный DataFrame с DataFrame закодированных признаков
    df_OHV = pd.concat([df, encoded_df], axis = 1)

    df_OHV.to_csv('data/interim/df_OHV.csv')

    # выделяем целевую переменную и предикторы
    y = df_OHV[target].map({'Покинул': 1,'Остался': 0})
    X = df_OHV.drop([target], axis = 1)

    # разделяем данные
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.25,
                                                        random_state = 0,
                                                        stratify = y)

    # сохраняем данные
    X_train.to_csv('data/processed/X_train.csv', index = False)
    X_test.to_csv('data/processed/X_test.csv', index = False)
    y_train.to_csv('data/processed/y_train.csv', index = False)
    y_test.to_csv('data/processed/y_test.csv', index = False)

    print('Разбивка датасета выполнены успешно')
    