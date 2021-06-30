#Libraries
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

import pingouin as pg # LinearRegression

from sklearn.preprocessing import StandardScaler, RobustScaler

'''
In this step I will convert string into numbers. Especially, using OneHotEncoder and get_dummies 
from sklearn and pandas, respectively.Then, i will create for categorical[nominal] variables as 
many columns as the unique values they receive and i will I will delete one of them in order to 
eliminate the resulting multicolinearity.
'''
#display categorical variables
print(f'The categorical variable of our data set are {categorical}')
categorical_df.head()

#A copy of df as encoding df
df_enc = df.copy()
df_enc.head()

#Encoding categorical feature 'sex'. Drop first is acceptable due to the the fact that males are 676 instead of females' number which is 662
df_enc = pd.get_dummies(df_enc, prefix = ['sex'], columns = ['sex'], drop_first = True)
#Encoding categorical feature 'smoker'. I will drop smoker_yes due to the fact that smoker_no represents 1064 patients
df_enc = pd.get_dummies(df_enc, prefix = ['smoker'], columns = ['smoker'])
df_enc.drop('smoker_yes', axis = 1, inplace = True)
#Encoding categorical feature 'region'. The colum that dropped is 'northwest'
df_enc = pd.get_dummies(df_enc, prefix = ['region'], columns = ['region'])
df_enc.drop("region_northwest", axis = 1, inplace = True)

'''
In this section i will try to identify if feature scaling is important in this step in order to prepare
the model for training. I will implement different Scalers based on the description of sklearn such as
MinMaxScaler, StandardScaler, MinAbsScaler and RobustScaler(is less sensitive to outliers)
Scaling features is vital in this dataset due to the fact that because the scales of variable 
numbers vary widely and at the same time the units of measurement are different such as the bmi variable.
'''
#Firstly, we should split the dataset into X_features and y_predictor

#pretrain_model without scaling features using pinguin library
X_features = df_enc.drop('charges', axis = 1)
y_target_variable = df_enc['charges']

pretrain_model = pg.linear_regression(X_features, y_target_variable, add_intercept = True)
pretrain_model

'''
In this step i will create a pretrain model , assuming that Region is ordinal categorical variable in order to
examine if the results are identical
'''
df_enc2 = df.copy()
df_enc2 = pd.get_dummies(df_enc2, prefix = ['sex'], columns = ['sex'], drop_first = True)
#Encoding categorical feature 'smoker'. I will drop smoker_yes due to the fact that smoker_no represents 1064 patients
df_enc2 = pd.get_dummies(df_enc2, prefix = ['smoker'], columns = ['smoker'])
df_enc2.drop('smoker_yes', axis = 1, inplace = True)
#assuming that region is ordinal variable
df_enc2['region'] = df_enc2['region'].replace({'southwest' : 0, 'southeast' : 1,
                                              'northwest' : 2, 'northeast' : 3})

'''
Pretrain model2
'''
#pretrain_model2 without scaling features using pinguin library
X2_features = df_enc2.drop('charges', axis = 1)
y_target_variable2 = df_enc2['charges']

pretrain_mode2 = pg.linear_regression(X2_features, y_target_variable2, add_intercept = True)
pretrain_mode2

'''
Pretrain_model3 without using sex_male variable
'''

#pretrain_model3 without scaling features using pinguin library
df_enc3 = df_enc2.copy()
df_enc3.drop('sex_male', axis = 1, inplace = True) # non-statistically significant

#features and target
X3_features = df_enc3.drop('charges', axis = 1)
y_target_variable3 = df_enc3['charges']

pretrain_model3 = pg.linear_regression(X3_features, y_target_variable3, add_intercept = True)
pretrain_model3

'''
 ______________
|StandardScaler|
'''
scaler1 = StandardScaler()
#splitting features based on statistically significant features
features = df_enc3.drop("charges", axis = 1)
target = df_enc3['charges']

features = scaler1.fit_transform(features)
features = pd.DataFrame(features, columns = ['age', 'bmi', 'children', 'region', 'smoker_no'])
features.head()

'''
 ____________
|RobustScaler|
'''
scaler2 = RobustScaler()
#splitting features based on statistically significant features
features_robust = df_enc3.drop("charges", axis = 1)
target = df_enc3['charges']

features_robust = scaler3.fit_transform(features)
features_robust = pd.DataFrame(features_robust, columns = ['age', 'bmi', 'children', 'region', 'smoker_no'])
features_robust.head()