#Import relevant libraries
import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf

################################ Data to Excel #################################

def data_to_excel(data, filename):
    writer = pd.ExcelWriter(filename)
    data.to_excel(writer, 'results')
    writer.save()

############################### Key Driver - OLS ###############################

#OLS function with target and features as kwargs
def lin_reg_xy(target, features):
    features_add_constant = sm.add_constant(features)
    results = sm.OLS(target, features_add_constant).fit()
    results_frame = pd.concat([round(results.params,3), results.pvalues], axis=1, keys=['estimate', 'p_vals'])
    result_r2 = round(results.rsquared,2)
    print(results.summary())
    return results_frame, result_r2

#OLS function with R-style formula + data as kwargs
def lin_reg_formula(formula, data):
    mod = smf.ols(formula=formula, data=data)
    results = mod.fit()
    # y_hat = results.predict()
    results_frame = pd.concat([round(results.params,3), results.pvalues], axis=1, keys=['estimate', 'p_vals'])
    result_r2 = round(results.rsquared,2)

    print(results.summary())
    key_driver_summary(results_frame, result_r2)
    key_driver_plot(results_frame)
    key_driver_influence_plot(results)
    key_driver_partialreg_plot(results)

    return results_frame, result_r2

        #~~~~~~~~~~~~~~~~~~~~~~~~ Visuals ~~~~~~~~~~~~~~~~~~~~~~~#

#3 point summary of output (r2, non-sig drivers, most impactful drivers)
def key_driver_summary(data, r2):
    non_sig_drivers = list(data[data['p_vals'] >= 0.05].index)
    most_impactful_driver = data.nlargest(3, 'estimate')
    print('\n')
    print('#############################')
    print('1.R-squared: {}'.format(r2))

    print('\n2.Non-Significant drivers: ')
    for item in non_sig_drivers:
        print('\t{}'.format(item))

    print('\n3.3 Most Impactful Drivers: ')
    for item in most_impactful_driver.index:
        print('\t{}'.format(item))

    print('#############################')
    print('\n')
    return

#Sample Key Driver Plot
def key_driver_plot(data):
    data_plot = data[data['p_vals'] < 0.05].drop('Intercept').sort_values('estimate', ascending=False)
    ax = sns.barplot(x='estimate', y=data_plot.index, data=data_plot)
    plt.show()
    return

#Partial Regression Plot
def key_driver_partialreg_plot(lm_results):
    fig = plt.figure(figsize=(12,12))
    fig = sm.graphics.plot_partregress_grid(lm_results, fig=fig)
    plt.show()
    return

#Influence Plot
def key_driver_influence_plot(lm_results):
    fig, ax = plt.subplots(figsize=(12,12))
    ax.set_xlim(left=0, right=0.25)
    fig = sm.graphics.influence_plot(lm_results, ax=ax, criterion="cooks")
    plt.show()
    return

################################################################################
############################ Penalty Rewards - OLS #############################

def pr_plot():
    ...
    return


####################### Target Setting - Decision Trees ########################

def decision_tree_plot():
    ...
    return
