import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

import functools

# рассчитаем, сколько зарабатывает банк без применения машинного обучения
BASIC_INCOME = 8500 * 500  # 8500 оставшихся клиентов принесут нам по 500 рублей каждый

# 1627 клиентам, которые собирались покинуть наш банк 
# мы предложили скидку и 50% согласились на нее
MODEL_INCOME = (8500 * 500 + 1627 * (500 - 100) / 2) - BASIC_INCOME

# Функции для оптимизации моделей и вывода результатов обучения

def income_function(y_test, y_pred, REVENUE = 500, BONUS = 100, PROBABILITTY = 0.5):
    """
    Функция оптимизация модели.
    Принимает тестовые и спрогнозированные данные,
    а также прибыль с одного клиента, размер бонуса и вероятность ухода.
    Возвращает долю от теоретически возможной прибыли
    """
    # извлекаем значения из матрицы ошибок
    TN, FP = confusion_matrix(y_test, y_pred)[0]
    FN, TP = confusion_matrix(y_test, y_pred)[1]
    
    # расчитываем прибыль для текущих параметров
    income = (TN * REVENUE + 
              FP * (REVENUE - BONUS) +
              TP * PROBABILITTY * (REVENUE - BONUS) + 
              FN * 0) * 4 - BASIC_INCOME
    
    # определяем сколько процентов от максимальной прибыли мы смогли бы получить
    result = income / MODEL_INCOME

    return result

def gridsearch(function, param, cv_ = 5, metrik = income_function):
    """
    Функция поиска по сетке параметров, 
    вместе с -income_function()- оптимизируют параметры моделей.
    Принимает функцию и список параметров для оптимизации.
    На выход подается модель с оптимальными параметрами.
    """
    # считываем данные
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')

    # определяем функцию оптимизации
    custom_scorer = functools.partial(metrik)
    
    # определяем параметры поиска оптимальной модели
    grid_search = GridSearchCV(function,
                               param_grid = param,
                               cv = cv_,
                               scoring = make_scorer(custom_scorer),
                               verbose = 10)
    
    # запуск обучения
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_

def conclusion(optim_model, start, end):
    """
    Функция вывода заключения по результатам обучения.
    На вход получает модель с оптимальными параметрами.
    В результате сообщает:
    - модель с оптимальными параметрами,
    - доля от теоретически возможной максимальной прибыли,
    которую можно получить с помощью этой модели,
    - оптимальный порог вероятности ухода, 
    при котором необходимо активировать бонусную программу,
    - точность обучения модели,
    - точность прогнозов модели,
    - длительность обучения модели,
    - точность прогноза, что клиент остается,
    - точность прогноза, что клиент уйдет,
    - матрица ошибок.
    """
    # считываем данные
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')

    # оптимальные настройки модели
    print(f'Оптимальная модель: {optim_model}\n')
    
    # расчитываем экономический эффект модели,
    # подобрав оптимальный порог активации бонусной программы
    THERSHOLDS = np.arange(1, 101)
    results = []
    above_threshold_percentage = []

    for threshold in THERSHOLDS:
        y_pred = (optim_model.predict_proba(X_test)[:, 1] >= threshold / 100)
        result = income_function(y_test, y_pred)
        results.append(result)
        above_threshold_percentage.append(np.mean(y_pred))
    
    # фиксируем лучшие результаты
    max_profit = max(results)
    optim_threshold = results.index(max(results)) + 1
    print(f'Модель обеспечивает {(max_profit * 100):,.2f}% возможной добавочной прибыли, при пороге активации бонусов при {optim_threshold}% \n')
    
    # на основании оптимальной модели делаем предсказания
    y_pred = (optim_model.predict_proba(X_test)[:, 1] >= optim_threshold / 100).reshape(-1, 1)
    y_training = (optim_model.predict_proba(X_train)[:, 1] >= optim_threshold / 100).reshape(-1, 1)
    
    # точность обучения
    print(f'Точность обучения: {round((y_training == y_train).mean() * 100, 2)}%\n' +
          f'Точность предсказания: {round((y_pred == y_test).mean() * 100, 2)}%\n' +
          f'Время: {round((end - start), 3)} сек.\n')
    
    # создаем матрицу ошибок и извлекаем из нее значения
    cm = confusion_matrix(y_test, y_pred)
    TN, FP = cm[0]
    FN, TP = cm[1]
    
    # точность прогнозов
    print(f'Точность прогноза "клиент останется": {round(TN / (TN + FP) * 100, 2)}%')
    print(f'Точность прогноза "клиент уйдет": {round(TP / (TP + FN) * 100, 2)}%')

    # строим матрицу ошибок
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Остались', 'Ушли'],
                yticklabels=['Остались', 'Ушли'])

    # добавляем подписи
    plt.xlabel('Предсказанные классы')
    plt.ylabel('Истинные классы')
    plt.title('Матрица ошибок')

    plt.show()

