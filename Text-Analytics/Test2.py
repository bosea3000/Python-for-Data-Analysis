import pandas as pd
import matplotlib as plt
%matplotlib inline

import PreProcessor
import Modeler
import importlib

importlib.reload(PreProcessor)

#---------------------------------PreProcessor---------------------------------#
#Load data
data_raw = pd.read_excel('SampleData-2.xls')
data_raw = PreProcessor.rename_cols(data_raw)
missing_raw = PreProcessor.missingSummary(data_raw)
missing_raw

#Clean data
data_clean, missing_clean = PreProcessor.cleanData(data_raw, missing_raw, 75, 30)
missing_clean

#Impute data
data_impute, missing_impute = PreProcessor.imputeData(data_clean, 'KNN')
missing_impute

#Catergorize data
features_to_cat = list(['Internet_quality', 'Ease_of_making_reservation','Attitude_of_hotel_staff', 'Cleanliness_of_room', 'Quietness_of_room',
       'Breakfast_quality', 'Cleanliness_of_bathroom', 'Bar_ambiance','Accuracy_of_bill'])
data_impute_copy = data_impute.copy()
data_categorized = PreProcessor.pr_categorize(data_impute_copy, features_to_cat)
data_categorized.head()

#Split data
target, features = PreProcessor.splitTargetFeatures(data_impute, 'Overall_Experience', features_to_cat)

#split data into training and test
x_train, x_test, y_train, y_test = PreProcessor.train_test_split_fx(target, features, 0.3)

#------------------------------------------------------------------------------#
#-----------------------------------Modeling-----------------------------------#

#Key Driver Analysis
all_cols = ' + '.join(data_impute.columns.drop('Overall_Experience'))
formula = 'Overall_Experience ~ ' + all_cols
lm_results, r2 = Modeler.lin_reg_formula(formula, data_impute)
r2
lm_results


#Penalty Rewards Analysis
features_dummy = []
for feature in features_to_cat:
    feature_name = 'C(' + feature+'_CAT)'
    features_dummy.append(feature_name)
all_cols = ' + '.join(features_dummy)
formula = 'Overall_Experience ~ ' + all_cols
pr_results, pr_r2 = Modeler.lin_reg_formula(formula, data_categorized)
r2
pr_results
