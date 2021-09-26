import pandas as pd
from xgboost import XGBRFClassifier
from sklift.models import SoloModel, ClassTransformation, TwoModels
from sklift.metrics import uplift_at_k


def combine_models(X_train, y_train, treat_train, X_test, y_test, treat_test):
    
    models_results_dic = {'approach': [],
                      'model': [],
                      'uplift': []}
    
    uplift_algos = ['SoloModel', 'ClassTransformation']
    for uplift_alg in uplift_algos:
        uplift_model(models_results_dic, uplift_alg, 'XGBRFClassifier',
                     X_train, y_train, treat_train,
                     X_test, y_test, treat_test,
                     method=None)

    uplift_algos = ['TwoModels']
    methods = ['vanilla', 'ddr_control', 'ddr_treatment']

    for uplift_alg in uplift_algos:
        for method in methods:
            uplift_model(models_results_dic, uplift_alg, 'XGBRFClassifier',
                         X_train, y_train, treat_train,
                         X_test, y_test, treat_test,
                         method)

    return models_results_dic

def uplift_model(models_results_dic, uplift_algo, classification_algo,
                 X_train, y_train,
                 treat_train,
                 X_test, y_test,
                 treat_test, method=None):


    classification_algos = {'XGBRFClassifier': XGBRFClassifier}
    uplift_algos = {'SoloModel': SoloModel,
                    'ClassTransformation': ClassTransformation,
                    'TwoModels': TwoModels}

    if uplift_algo == 'TwoModels':
        model = TwoModels(
            estimator_trmnt=classification_algos[classification_algo](random_state=42),
            estimator_ctrl=classification_algos[classification_algo](random_state=42),
            method=method)
        model = model.fit(X_train, y_train, treat_train)
        uplift = model.predict(X_test)

        models_results_dic['approach'].append("TwoModels_" + method + "_" + classification_algo)
        models_results_dic['model'].append(model)
        models_results_dic['uplift'].append(uplift)

    else:
        model = uplift_algos[uplift_algo](classification_algos[classification_algo](random_state=42))
        model = model.fit(X_train, y_train, treat_train)
        uplift = model.predict(X_test)

        models_results_dic['approach'].append(uplift_algo + "_" + classification_algo)
        models_results_dic['model'].append(model)
        models_results_dic['uplift'].append(uplift)


def uplift_at_k_best(models_results, y_test, treat_test):
    models_results['uplift_at_30'] = []
    for i in range(len(models_results['uplift'])):
        models_results['uplift_at_30'].append(uplift_at_k(y_test,
                                                          models_results['uplift'][i],
                                                          treat_test,
                                                          strategy='overall',
                                                          k=0.3))
    return (pd.DataFrame({'approach': models_results['approach'],
                          'uplift_at_30': models_results['uplift_at_30']}
                        ).sort_values(by='uplift_at_30', ascending=False))