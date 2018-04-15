#Import relevant libraries
import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf

def lin_reg_xy(target, features):
    features_add_constant = sm.add_constant(features)
    results = sm.OLS(target, features_add_constant).fit()
    results_frame = pd.concat([round(results.params,3), results.pvalues], axis=1, keys=['estimate', 'p_vals'])
    result_r2 = round(results.rsquared,2)
    print(results.summary())
    return results_frame, result_r2

def lin_reg_formula(formula, data):
    mod = smf.ols(formula=formula, data=data)
    results = mod.fit()
    results_frame = pd.concat([round(results.params,3), results.pvalues], axis=1, keys=['estimate', 'p_vals'])
    result_r2 = round(results.rsquared,2)
    print(results.summary())
    return results_frame, result_r2

def key_driiver_plot():
    ...
    return

def pr_plot():
    ...
    return

def decision_tree_plot():
    ...
    return
    
