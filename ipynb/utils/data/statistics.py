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


def read_dataframe(filepath_or_buffer='../data/minethatdata_womens_train.csv', index_col='index', sep=','):
    data = pd.read_csv(filepath_or_buffer=filepath_or_buffer, index_col=index_col, sep=sep)
    return data

    
def get_dataframe_statistics(data):
    for column in data.columns:
        print(str(column) + ' has {} unique variables'.format(data[column].nunique()))
        print('--------------------')
        

def get_dataframe_pearsoncorr(data):
    pearsoncorr = data.corr(method='pearson')
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(pearsoncorr,
            xticklabels=pearsoncorr.columns,
            yticklabels=pearsoncorr.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)
            
            
