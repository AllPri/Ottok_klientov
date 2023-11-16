from src.features.model_training import *

def fit_models():
    """
    Запуск обучения моделей
    """
    model = input('Ведите название модели ("help" вызов справки): ')
    if model == 'help':
        print('Поддерживаемые модели:\n',
            '"GaussianNB" - наивный Байесовский классификатор\n',
            '"LogisticRegression" - логистическая регрессия\n',
            '"DecisionTreeClassifier" - дерево решений\n',
            '"RandomForestClassifier" -  случайные леса\n',
            '"SVC" - метод опорных векторов\n',
            '"KNeighborsClassifier" - метод k-средних\n',
            '"GradientBoostingClassifier" - градиентный бустинг\n',
            '"XGBClassifier" - XGBoost\n',
            '"CatBoostClassifier" - CatBoost\n',
            '"LGBMClassifier" - LightGBM\n',
            '"StackingClassifier_cl" - стек классических методов ML\n',
            '"StackingClassifier" - стек всех моделей\n',
            )
    
    elif model == 'GaussianNB':
        fit_GaussianNB_model()

    elif model == 'LogisticRegression':
        fit_LogisticRegression_model()

    elif model == 'DecisionTreeClassifier':
        fit_DecisionTreeClassifier_model()

    elif model == 'RandomForestClassifier':
        fit_RandomForestClassifier_model()

    elif model == 'SVC':
        fit_SVC_model()

    elif model == 'KNeighborsClassifier':
        fit_KNeighborsClassifier_model()

    elif model == 'GradientBoostingClassifier':
        fit_GradientBoostingClassifier_model()

    elif model == 'XGBClassifier':
        fit_XGBClassifier_model()

    elif model == 'CatBoostClassifier':
        fit_CatBoostClassifier_model()

    elif model == 'LGBMClassifier':
        fit_LGBMClassifier_model()

    elif model == 'StackingClassifier_cl':
        fit_StackingClassifier_cl_model()

    elif model == 'StackingClassifier':
        fit_StackingClassifier_model()

    else: print('Ошибка ввода, обратитесь к справке')