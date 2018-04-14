import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
%matplotlib inline
from fancyimpute import KNN
from sklearn.model_selection import train_test_split
import PreProcessor

from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

#Load raw dataset
data_raw = pd.read_csv('cs_responses_export_2018-03-31_235959.csv', low_memory=False)
missing_raw = PreProcessor.missingSummary(data_raw)
missing_raw

#Clean dataset
data_cleaned, missing_cleaned = PreProcessor.cleanData(data_raw, missing_raw, 60, 20)
missing_cleaned

#Filter data - keep only relevant columns
cols_to_keep = list(['OSAT Experience', 'Client Effort','Likelihood to Recommend Bank','Professionalism', 'Knowledge','Communicating Clearly','Understanding Needs',
    'Issue Resolution','CC Segment','Asset Range','Age Range','Relationship Tenure','Client Tenure','State Code'])
data_subset = data_cleaned[cols_to_keep]

#Encode categorical / Object data
cols_cat = list(data_subset.select_dtypes('object').columns)
data_encoded_from_subset = pd.get_dummies(data_subset, columns=cols_cat)

#Split data into training and test
target = data_encoded_from_subset['OSAT Experience']
features = data_encoded_from_subset.drop('OSAT Experience', axis=1)
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.33)

#Model - Decision Tree
reg_tree_1 = tree.DecisionTreeRegressor(max_depth = 2)
reg_tree_2 = tree.DecisionTreeRegressor(max_depth = 3)
reg_tree_3 = tree.DecisionTreeRegressor(max_depth = 5)
reg_tree_4 = tree.DecisionTreeRegressor(max_depth = 7)

reg_tree_1 = reg_tree_1.fit(features_train, target_train)
reg_tree_2 = reg_tree_2.fit(features_train, target_train)
reg_tree_3 = reg_tree_3.fit(features_train, target_train)
reg_tree_4 = reg_tree_4.fit(features_train, target_train)

y_1 = reg_tree_1.predict(features_test)
y_2 = reg_tree_2.predict(features_test)
y_3 = reg_tree_3.predict(features_test)
y_4 = reg_tree_4.predict(features_test)

tree.export_graphviz(reg_tree_1, out_file='tree1.dot')
tree.export_graphviz(reg_tree_2, out_file='tree_2.dot')
tree.export_graphviz(reg_tree_3, out_file='tree_3.dot')
tree.export_graphviz(reg_tree_4, out_file='tree_4.dot')
!dot -Tpng tree_4.dot -o tree_4.png

scores = []
r2 = []
reg_tree_list = [reg_tree_1, reg_tree_2, reg_tree_3, reg_tree_4]
for item in reg_tree_list:
    r2 = item.score(features_test, target_test)
    scores.append(r2)
scores
features_test.columns

print(r2_score(target_test, reg_tree_4.predict(features_test)))
import pydotplus
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz

dot_data = StringIO()
export_graphviz(reg_tree_2, out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=features_train.columns)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
img = Image(graph.create_png())
img
