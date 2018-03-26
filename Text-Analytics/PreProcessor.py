#Import relevant libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from fancyimpute import KNN
from pandas.io.stata import StataReader
from sklearn.preprocessing import Imputer

############################## LOADFILE-FUNCTIONS ##############################

#Load File in Chunks Function
def chunkLoad(dta_filename):
    ...
    return

#Load File from Stata-Format
def stataLoad(dta_filename):
    reader = StataReader(dta_filename)
    data = reader.data()
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

    return cleanedData, cleanedData_missingdf

#Imputation Funtion
    #Allows the following Imputation methods: 'Backfill', 'Forwardfill', 'Valuefill', 'Mean', 'Median', 'Mode', 'KNN'
    #Returns Imputed_DataFrame + Missing Summary of Imputed_DataFrame
def imputeData(data, method):
    if method in ('bfill', 'ffill'):
        imputedData = data.fillna(method=method, axis=1, inplace=False)
    elif method in ('value'):
        value = int(input('Enter scalar value: '))
        imputedData = data.fillna(value=value, axis=1, inplace=False)
    elif method in ('mean', 'median', 'most_frequent'):
        imp = Imputer(missing_values = np.NaN, strategy=method, axis=0, copy=True)
        imputedData = imp.fit_transform(data)
    else:
        nn = int(input('Enter number of nearest neighbors: '))
        imputedData = KNN(k=nn).complete(data)

    imputedData = pd.DataFrame(imputedData)
    imputedData.columns = data.columns

    # imputedData_missingdf = missingSummary(imputedData)
    return round(imputedData,0)

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
    ...
    return

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
