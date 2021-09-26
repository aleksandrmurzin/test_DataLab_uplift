import pandas as pd
import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBRFClassifier
from sklift.models import SoloModel, ClassTransformation, TwoModels
from sklift.metrics import uplift_at_k
from sklift.metrics import weighted_average_uplift
from sklift.metrics import uplift_by_percentile
from sklift.viz import plot_uplift_by_percentile
from sklift.viz import plot_qini_curve
from sklift.viz import plot_uplift_preds
from sklift.viz import plot_uplift_curve
from pandas_profiling import ProfileReport
import joblib


def prepare_treatment_target_columns(data):
    """
    Function that convertes target variable to 1 and 0.
    Also renames collumns to pylift model standart
    Params:
        data -- dataframe
    Returns:
        data -- dataframe with renamed and
        converten to int values treatnemt variable
    """

    try:
        data['segment'] = data['segment'].apply(
            lambda x: 1 if x == 'Womens E-Mail' else 0)
        data.rename(columns={'segment': 'treatment_flg',
                             'visit': 'target'}, inplace=True)
    except:
        pass
    return data
    

def str_shuf_train_test_split(data, stratification_column='recency'):
    """
    Function that does stratified shuffled train test split
    Parametrs:
        data -- dataframe to split
        stratification_column -- column based on which make a stratification
    """
    split = StratifiedShuffleSplit(n_splits=5, test_size=0.5,
                                   train_size=0.5, random_state=42)
    for train_index, test_index in split.split(data,
                                               data[stratification_column]):
        train = data.iloc[train_index]
        test = data.iloc[test_index]
    return train, test     
    

def build_preprocessing_pipline(df):
    """
    Function takes data frame and does transformation of categorical features
    through one hot encoding.
    Function takes data frame and does transformation of numerical features
    through standartization or normalization

    Params: data -- a dataframe to transform
    Returns: transformation dict
    """
    transformation = {}
    numeric_features = df.select_dtypes(include='float64').columns.to_list()
    numeric_pipeline = Pipeline(steps=[('scaler', StandardScaler())])

    categorical_features = df.select_dtypes('object').columns.to_list()
    categorical_pipeline = Pipeline(steps=[
        ('OHE', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)])

    transformation = {
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'preprocessor': preprocessor}
    return transformation
    
def create_processed_dataframe(data,
                               data_array_transformed,
                               transformation):
    """
    Function that takes dataframe, doues transformation and
    then wraps it all to a ready to use dataframe 
    Parametrs:
        data -- initial df needed to get column names
        data_array_transformed -- 2D array of transformed data
        transformation -- dictionary from build_preprocessing_pipline 
        to get names for columns
    """

    columns_init = data.columns
    columns = (set(columns_init) -
               set(transformation['numeric_features']) -
               set(transformation['categorical_features']))
    categorical_features_transformed = (transformation['preprocessor'].
                                        named_transformers_['cat']['OHE'].
                                        get_feature_names(
                                        transformation['categorical_features']))
    categorical_features_transformed = [i.split(")")[0] for i in categorical_features_transformed]
    data_ = pd.DataFrame(data_array_transformed,
                         columns=(transformation['numeric_features']
                                  + categorical_features_transformed),
                         index=data.index)
    data_transformed = data_.join(data[columns])
    data_transformed = data_transformed.drop(
        columns=[
            'history_segment_1',
            'zip_code_Surburban',
            'channel_Phone'], axis=1)
    return data_transformed

    
def full_preprocessing_pipline(train, test, df):
    """
    Make preprocessing pipline from trasformation build from
    build_preprocessing_pipline fucntion
    Parametrs:
        train -- train dataset
        test -- test dataset
    """
    transformation = build_preprocessing_pipline(df)
    train_array_transformed = transformation['preprocessor'].fit_transform(train)
    test_array_transformed = transformation['preprocessor'].transform(test)

    train_transformed = create_processed_dataframe(train,
                                                   train_array_transformed,
                                                   transformation)
    test_transformed = create_processed_dataframe(test,
                                                  test_array_transformed,
                                                  transformation)

    return train_transformed, test_transformed


def prepare_data_for_sklift(data):
    """
    Prepare data for sklift library
    Parametrs:
        data -- dataframe
    Returns:
        X -- features to train model
        y -- target feature
        treat -- treatment feature
    """
    X = data.drop(columns=['treatment_flg', 'target'], axis=1)
    y = data['target']
    treat = data['treatment_flg']
    return X, y, treat

   
    
    
    	
