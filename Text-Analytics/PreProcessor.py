#Import relevant libraries
import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from fancyimpute import KNN
from pandas.io.stata import StataReader
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split

############################## LOADFILE-FUNCTIONS ##############################

#Load File(Stata File .dta) in Chunks Function
def chunkLoad(filename):
    reader = pd.stata_reader(filename, iterator=True)
    df = pd.DataFrame()

    try:
        chunk = reader.get_chunk(100*1000)
        while len(chunk) > 0:
            df = df.append(chunk, ignore_index=True)
            chunk = reader.get_chunk(100*1000)
            print(".")
            sys.stdout.flush()
    except (StopIteration, KeyboardInterrupt):
        pass

    print("\nLoaded {} rows".format(len(df)))
    return df

#Load File from Stata-Format
def stataLoad(dta_filename):
    reader = StataReader(dta_filename)
    data = reader.data()
    print("\nLoaded {} rows".format(len(data)))
    return data

def rename_cols(data):
    col_list = []
    for col_name in list(data.columns):
        col_name = col_name.replace(" ", "_")
        col_list.append(col_name)
    data.columns = col_list
    return data

############################ MANIPULATION-FUNCTIONS ############################

#Missing Summary Function
def missingSummary(data):
    total = data.isnull().sum()
    remaining = len(data) - data.isnull().sum()
    percent = round(data.isnull().sum() / data.isnull().count() * 100,2)
    missingdf = pd.concat([total, remaining, percent], axis=1, keys=['Total', 'Remaining', 'Percent'])
    missingdf = missingdf[missingdf['Percent'] > 0]
    return missingdf.sort_values('Percent', ascending=False)

#Clean Data Function
    #Allows for data cleaning based on minimum missing value cut-off req.
    #Returns Cleaned_DataFrame + Missing Summary of Cleaned_DataFrame
def cleanData(rawdata, missingdf, col_cutoff, row_cutoff):
    cols_to_drop = list(missingdf[missingdf['Percent'] >= col_cutoff].index)
    rows_to_drop = list(missingdf[missingdf['Percent'] <= row_cutoff].index)

    cleanedData = rawdata.copy()
    cleanedData.drop(cols_to_drop, axis=1, inplace=True)
    cleanedData.dropna(subset=rows_to_drop, how='any', inplace=True)
    cleanedData_missingdf = missingSummary(cleanedData)
    print('Dataset has been cleaned.')
    return cleanedData, cleanedData_missingdf

#Imputation Funtion
    #Allows the following Imputation methods: 'Backfill', 'Forwardfill', 'Valuefill', 'Mean', 'Median', 'Mode', 'KNN'
    #Returns Imputed_DataFrame + Missing Summary of Imputed_DataFrame
def imputeData(data, method):
    if method in ('bfill', 'ffill'):
        imputed_data = data.fillna(method=method, axis=1)
    elif method in ('value'):
        scalar_val = int(input('Enter imputation value: '))
        imputed_data = data.fillna(value=scalar_val, axis=1)
    elif method in ('mean', 'median', 'most_frequent'):
        imputer = Imputer(missing_values='NaN', strategy=method, axis=0)
        imputed_data = imputer.fit_transform(data)
    else:
        nn = int(input('Enter number of nearest neighbors: '))
        imputed_data = KNN(nn).complete(data)

    imputed_data_frame = pd.DataFrame(imputed_data)
    imputed_data_frame.columns = data.columns
    imputed_data_frame.index = data.index

    missing_imputed = missingSummary(imputed_data_frame)
    return round(imputed_data_frame,2), missing_imputed

############################## CORRELATION PLOTS ###############################

#Correlation Matrix Function
def heatmap(data):
    dfcorr = data.corr()
    mask = np.zeros_like(dfcorr)
    mask[np.triu_indices_from(mask)] = True
    figure, ax = plt.subplots(figsize=(12,9))
    with sns.axes_style('white'):
        ax = sns.heatmap(dfcorr, mask=mask, vmin=0, vmax=1, square=True, annot=True, cmap='Blues', center=0.5)
    return ax

def pairwise(data):
    sns.set(style="ticks", color_codes=True)
    g = sns.pairplot(data)
    return g

############################# CATEGORIZE-FUNCTIONS #############################

#Encode Score-to-NPS Category (Apply function)
def scoretoNPSCat(x):
    if x > 8:
        return "1.Promoter"
    elif x > 6:
        return "0.Passive"
    else:
        return "2.Detractor"

#Encode Score-to-NPS-Modified Category (Apply function)
def scoretoNPSCat_Modified(x):
    if x > 8:
        return "Promoter"
    else:
        return "Non-Promoter"

def pr_categorize(data, features_to_cat):
    for col in features_to_cat:
        data[col+'_CAT'] = data[col].apply(scoretoNPSCat)
    print('All features have been categorized')
    return data

################################## SPLIT DATA ##################################

def splitTargetFeatures(data, target_name, features_name):
    target = data[[target_name]]
    features = data.drop(target_name, axis=1)
    features = features[features_name]
    print('Target and features are now separated')
    return target, features

def train_test_split_fx(x, y, train_limit):
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=train_limit)
    print('Data has been split into training and test sets')
    return x_train, x_test, y_train, y_test
