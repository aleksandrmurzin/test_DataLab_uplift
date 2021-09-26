import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
            
            
