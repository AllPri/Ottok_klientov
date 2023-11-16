import numpy as np
import pandas as pd
import joblib
import time

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegressionCV

from src.features.optimization_functions import gridsearch
from src.features.optimization_functions import conclusion

def fit_GaussianNB_model():
    # старт таймера
    start = time.time()
    # подбор оптимальных параметров
    GaussianNB_model = gridsearch(GaussianNB(),
                                param = {'var_smoothing': np.logspace(0, -9, num = 100)})  # уровень сглаживания
    # остановка таймера
    end = time.time()
    # сохранение модели
    joblib.dump(GaussianNB_model, 'models/GaussianNB_model.pkl')
    # вывод результата обучения
    conclusion(GaussianNB_model, start, end)

def fit_LogisticRegression_model():
    start = time.time()
    LogisticRegression_model = gridsearch(LogisticRegression(class_weight = {0: 1, 1: 1.75}, random_state = 42, max_iter = 1500),
                                         param = {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # алгоритм решения
                                                  'penalty':[None, 'elasticnet', 'l1', 'l2'],                    # метод регуляризации
                                                  'C':[0.001, 0.01, 0.1, 1, 10, 100]})                           # обратный коэффициент регуляризации
    end = time.time()
    joblib.dump(LogisticRegression, 'models/LogisticRegression_model.pkl')

    conclusion(LogisticRegression_model, start, end)

def fit_DecisionTreeClassifier_model():
    start = time.time()
    DecisionTreeClassifier_model = gridsearch(DecisionTreeClassifier(class_weight = {0: 1, 1: 1.75}, random_state = 42),
                                              param = {'max_features': ['sqrt', 'log2'],  # максимальное количество признаков для деления
                                                       'criterion': ['gini', 'entropy'],  # критерий для деления
                                                       'ccp_alpha': [0.1, .01, .001],     # минимальное значение alpha для ограничения дерева
                                                       'max_depth': range(3, 21)})        # максимальная глубина дерева
    end = time.time()
    joblib.dump(DecisionTreeClassifier_model, 'models/DecisionTreeClassifier_model.pkl')

    conclusion(DecisionTreeClassifier_model, start, end)


def fit_RandomForestClassifier_model():
    start = time.time()
    RandomForestClassifier_model = gridsearch(RandomForestClassifier(class_weight = {0: 1, 1: 1.75}, random_state = 42),
                                            param = {'criterion': ['gini', 'entropy'],  # критерий для разделения
                                                     'max_features': ['sqrt', 'log2'],  # максимальное количество признаков для разделения
                                                     'n_estimators': [100, 250, 500],   # количество деревьев
                                                     'ccp_alpha': [0.1, .01, .001],     # минимальная стоимость среза
                                                     'max_depth': range(3, 21)})        # максимальная глубина деревьев
    end = time.time()
    joblib.dump(RandomForestClassifier_model, 'models/RandomForestClassifier_model.pkl')

    conclusion(RandomForestClassifier_model, start, end)

def fit_SVC_model():
    start = time.time()
    SVC_model = gridsearch(SVC(class_weight = {0: 1, 1: 1.75}, probability = True, random_state = 42),
                           param = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']})  # тип ядра 
    end = time.time()
    joblib.dump(SVC_model, 'models/SVC_model.pkl')

    conclusion(SVC_model, start, end)


def fit_KNeighborsClassifier_model():
    start = time.time()
    KNeighborsClassifier_model = gridsearch(KNeighborsClassifier(),
                                            param = {'n_neighbors': range(2, 21),                      # количество соседей для анализа
                                                     'weights': ['uniform', 'distance'],               # метод определения веса соседей
                                                     'algorithm': ['ball_tree', 'kd_tree', 'brute']})  # алгоритм поиска ближайших соседей
    end = time.time()    
    joblib.dump(KNeighborsClassifier_model, 'models/KNeighborsClassifier_model.pkl')

    conclusion(KNeighborsClassifier_model, start, end)

def fit_GradientBoostingClassifier_model():
    start = time.time()
    GradientBoostingClassifier_model = gridsearch(GradientBoostingClassifier(),
                                                  param = {'loss':['log_loss', 'exponential'],                           # функция потерь
                                                           'learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],  # размер шага обучения
                                                           'max_depth': range(3, 15),                                    # максимальная глубина дерева
                                                           'max_features': ['log2', 'sqrt'],                             # функции для поиска наилучшего решения
                                                           'subsample': [0.5, 0.6, 0.7, 0.8, 0.9 ,1]})                   # доля образцов для построения каждого дерева
    end = time.time()    
    joblib.dump(GradientBoostingClassifier_model, 'models/GradientBoostingClassifier_model.pkl')

    conclusion(GradientBoostingClassifier_model, start, end)

def fit_XGBClassifier_model():
    start = time.time()
    XGBClassifier_model = gridsearch(XGBClassifier(class_weight = {0: 1, 1: 1.75}, random_state = 42, tree_method = 'gpu_hist'),
                                    param = {'n_estimators': range(100, 350, 50),            # количество деревьев
                                             'colsample_bytree': np.arange(0.1, 1.1, 0.1)})  # доля признаков, которая будут использоваться при обучении каждого дерева
    end = time.time()    
    joblib.dump(XGBClassifier_model, 'models/XGBClassifier_model.pkl')

    conclusion(XGBClassifier_model, start, end)

def fit_CatBoostClassifier_model():
    start = time.time()
    CatBoostClassifier_model = gridsearch(CatBoostClassifier(random_state = 42, task_type = 'GPU', verbose = 100),
                                          param = {'depth': range(3, 6),                               # глубина дерева
                                                   'learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1]})  # размер шага обучения
                                                
    end = time.time()
    joblib.dump(CatBoostClassifier_model, 'models/CatBoostClassifier_model.pkl')

    conclusion(CatBoostClassifier_model, start, end)

def fit_LGBMClassifier_model():
    start = time.time()
    LGBMClassifier_model = gridsearch(LGBMClassifier(class_weight = {0: 1, 1: 1.75}, random_state = 42, device = "gpu", verbose = 0),
                                    param = {'feature_fraction': np.arange(0.1, 1.1, 0.1),       # Доля признаков для обучения отдельного дерева
                                            'learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1]})  # Скорость обучения 

    end = time.time()
    joblib.dump(LGBMClassifier_model, 'models/fit_LGBMClassifier_model.pkl')

    conclusion(LGBMClassifier_model, start, end)

def fit_StackingClassifier_cl_model():

    # загрузка предобученных моделей
    GaussianNB_model = joblib.load('models/GaussianNB_model.pkl')
    LogisticRegression_model = joblib.load('models/LogisticRegression_model.pkl')
    DecisionTreeClassifier_model = joblib.load('models/DecisionTreeClassifier_model.pkl')
    RandomForestClassifier_model = joblib.load('models/RandomForestClassifier_model.pkl')
    SVC_model = joblib.load('models/SVC_model.pkl')
    KNeighborsClassifier_model = joblib.load('models/KNeighborsClassifier_model.pkl')
    GradientBoostingClassifier_model = joblib.load('models/GradientBoostingClassifier_model.pkl')

    # считываем данные
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')

    # обозначаем используемые модели и метамодель, которая будет подводить общий итог
    estimators = [('GaussianNB', GaussianNB_model),
                ('LogisticRegression', LogisticRegression_model),
                ('DecisionTreeClassifier', DecisionTreeClassifier_model),
                ('RandomForestClassifier', RandomForestClassifier_model),
                ('SVC', SVC_model),
                ('KNeighborsClassifier', KNeighborsClassifier_model),
                ('GradientBoostingClassifier', GradientBoostingClassifier_model)]

    meta_model = LogisticRegressionCV(random_state = 42, max_iter = 1000)

    # запускаем обучение
    start = time.time()
    StackingClassifier_cl_model = StackingClassifier(estimators = estimators,
                                                    final_estimator = meta_model,
                                                    passthrough = True,
                                                    cv = 5,
                                                    verbose = 2).fit(X_train.to_numpy(), 
                                                                    y_train.to_numpy())
    end = time.time()    
    joblib.dump(StackingClassifier_cl_model, 'models/StackingClassifier_cl_model.pkl')

    conclusion(StackingClassifier_cl_model, start, end)

def fit_StackingClassifier_model():

    GaussianNB_model = joblib.load('models/GaussianNB_model.pkl')
    LogisticRegression_model = joblib.load('models/LogisticRegression_model.pkl')
    DecisionTreeClassifier_model = joblib.load('models/DecisionTreeClassifier_model.pkl')
    RandomForestClassifier_model = joblib.load('models/RandomForestClassifier_model.pkl')
    SVC_model = joblib.load('models/SVC_model.pkl')
    KNeighborsClassifier_model = joblib.load('models/KNeighborsClassifier_model.pkl')
    GradientBoostingClassifier_model = joblib.load('models/GradientBoostingClassifier_model.pkl')
    XGBClassifier_model = joblib.load('models/XGBClassifier_model.pkl')
    CatBoostClassifier_model = joblib.load('models/CatBoostClassifier_model.pkl')
    LGBMClassifier_model = joblib.load('models/LGBMClassifier_model.pkl')

    # считываем данные
    X_train = pd.read_csv('data/processed/X_train.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')

    estimators = [('GaussianNB', GaussianNB_model),
                  ('LogisticRegression', LogisticRegression_model),
                  ('DecisionTree', DecisionTreeClassifier_model),
                  ('RandomForest', RandomForestClassifier_model),
                  ('SVC', SVC_model),
                  ('KNeighbors', KNeighborsClassifier_model),
                  ('GradientBoosting', GradientBoostingClassifier_model),
                  ('XGB', XGBClassifier_model),
                  ('CatBoost', CatBoostClassifier_model),
                  ('LGBM', LGBMClassifier_model)]

    meta_model = LogisticRegressionCV(random_state = 42, max_iter = 1000)

    start = time.time()
    StackingClassifier_model = StackingClassifier(estimators = estimators,
                                                  final_estimator = meta_model,
                                                  passthrough = True,
                                                  cv = 5,
                                                  verbose = 2).fit(X_train.to_numpy(), 
                                                                   y_train.to_numpy())
    end = time.time()
    joblib.dump(StackingClassifier_model, 'models/StackingClassifier_model.pkl')

    conclusion(StackingClassifier_model, start, end)