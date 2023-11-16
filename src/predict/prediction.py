import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import joblib
import time

def make_predict():
    encoder = OneHotEncoder(sparse_output = False, drop = 'first')
    categorical_columns = ['Пол', 'Образование', 'Семейный статус', 'Уровень дохода', 'Категория карты']
    encoder.fit_transform(pd.read_csv('data/raw/data_raw.csv')[categorical_columns])


    def prepare_new_data(new_data, categorical_columns, encoder):
        # преобразуем категориальные признаки в формат one-hot-encoding
        new_data_encoded = encoder.transform(new_data[categorical_columns])

        # получаем имена столбцов после кодирования
        encoded_columns = encoder.get_feature_names_out(input_features = categorical_columns)

        # создаем DataFrame с закодированными признаками
        new_data_encoded_df = pd.DataFrame(new_data_encoded, columns = encoded_columns).astype(bool)

        # удаляем оригинальные категориальных признаки из новых данных
        new_data = new_data.drop(categorical_columns, axis=1)

        # объединяем новые данные с DataFrame закодированными признаками
        new_data = pd.concat([new_data, new_data_encoded_df], axis=1)

        return new_data

    # ввод данных
    f = input('Введите "свои данные" для прогноза на своих данных и "пример", для использования примера: ')

    if f == 'свои данные':
        model = input('Введите название модели для прогноза: ')
        age = float(input('Возраст заёмщика (в годах): '))
        gender = input('Пол (Мужчина / Женщина): ')
        dependents = float(input('Количество иждивенцев: '))
        education = input('Образование (Нет образования / Колледж / Выпускник / Старшая школа / Колледж / Аспирант / Докторская степень): ')
        family_status = input('Семейный статус (Женат/Замужем / Холост/Холостая / Неизвестно / В разводе): ')
        income_level = input('Уровень дохода (до $40K / от $40K до $60K / от $60K до $80K / от $80K до $120K / больше $120K / неизвестно)): ')
        card_categories = input('Категория карты (Blue / Silver / Gold / Platinum): ')
        duration_relationship_bank = float(input('Длительность отношения с банком (месяцев): '))
        number_bank_products = float(input('Количество продуктов банка у клиента: '))
        inactive_months = float(input('Количество неактивных месяцев за последний год: '))
        number_contacts = float(input('Количество контактов за последний год: '))
        credit_limit = float(input('Кредитный лимит ($): '))
        total_revolving_balance = float(input('Общий возобновляемый остаток средств ($): '))
        credit_line = float(input('Открытая кредитная линия ($): '))
        changing_transaction_amount = float(input('Индекс изменения суммы транзация с 4го квартала до первого (от 0 до 1): '))
        transaction_amount = float(input('Общая сумма транзакции за год ($): '))
        number_transactions = float(input('Общее количество транзакций за год: '))
        changing_number_transactions = float(input('Индекс изменения количества транзаций с 4го квартала до первого (от 0 до 1): '))
        average_card_usage_rate = float(input('Средний коэффициент использования карты: '))
    
    elif f == 'пример':
        model = 'LGBMClassifier'
        age = 26
        gender = 'Мужчина'
        dependents = 0
        education = 'Колледж'
        family_status = 'Холост/Холостая'
        income_level = 'до $40K'
        card_categories = 'Blue'
        duration_relationship_bank = 16
        number_bank_products = 1
        inactive_months = 0
        number_contacts = 0
        credit_limit = 4200
        total_revolving_balance = 500
        credit_line = 8000
        changing_transaction_amount = 0.3
        transaction_amount = 1500
        number_transactions = 50
        changing_number_transactions = 0.3
        average_card_usage_rate = 0.1
    
    else: print('Ошибка ввода!')

    # прогноз с расчетом длительности принятия решения
    start_time = time.time()
    data = {'Возраст': [age],
            'Пол': [gender],
            'Количество иждивенцев': [dependents],
            'Образование': [education],
            'Семейный статус': [family_status],
            'Уровень дохода': [income_level],
            'Категория карты': [card_categories],
            'Длительность взаимоотношения с банком': [duration_relationship_bank],
            'Количество продуктов банка у клиента': [number_bank_products],
            'Количество неактивных месяцев за последний год': [inactive_months],
            'Количество контактов за последний год': [number_contacts],
            'Кредитный лимит': [credit_limit],
            'Общий возобновляемый остаток средств': [total_revolving_balance],
            'Открытая кредитная линия': [credit_line],
            'Изменение суммы транзакции(4 к 1 кварталу)': [changing_transaction_amount],
            'Общая сумма транзакции за год': [transaction_amount],
            'Общее количество транзакций за год': [number_transactions],
            'Изменение количества транзакций(4 к 1 кварталу)': [changing_number_transactions],
            'Средний коэффициент использования карты': [average_card_usage_rate]}

    # загрузка предобученной модели
    if model == 'GaussianNB': model = joblib.load('models/GaussianNB_model.pkl')
    elif model == 'LogisticRegression': model = joblib.load('models/LogisticRegression_model.pkl')
    elif model == 'DecisionTreeClassifier': model = joblib.load('models/DecisionTreeClassifier_model.pkl')
    elif model == 'RandomForestClassifier': model = joblib.load('models/RandomForestClassifier_model.pkl')
    elif model == 'SVC': model = joblib.load('models/SVC_model.pkl')
    elif model == 'KNeighborsClassifier': model = joblib.load('models/KNeighborsClassifier_model.pkl')
    elif model == 'GradientBoostingClassifier': model = joblib.load('models/GradientBoostingClassifier_model.pkl')
    elif model == 'XGBClassifier': model = joblib.load('models/XGBClassifier_model.pkl')
    elif model == 'CatBoostClassifier': model = joblib.load('models/CatBoostClassifier_model.pkl')
    elif model == 'LGBMClassifier': model = joblib.load('models/LGBMClassifier_model.pkl')
    elif model == 'StackingClassifier_cl': model = joblib.load('models/StackingClassifier_cl_model.pkl')
    elif model == 'StackingClassifier': model = joblib.load('models/StackingClassifier_model.pkl')

    new_data = pd.DataFrame(data)
    new_data = prepare_new_data(new_data, categorical_columns, encoder)

    y_pred_new = model.predict_proba(new_data)
    end_time = time.time()

    # получим вероятность ухода
    print(f'Вероятность ухода: {round(y_pred_new[0][1] * 100, 2)}%, решение принято за {round(end_time - start_time, 5)} секунд')
