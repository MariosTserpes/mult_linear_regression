'''
Firstly, i will implement the linear regression algorithm, 
drawing on the conclusions of the above procedure. 
That is, in the in the total dataset that statistically significant variables 
are age, bmi, children, region, smoker_no. 
Hint : In this case, the "region" variable has been treated as an ordinal.
'''

features #features with StandardScaler
target # Charges dependent 

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size = 0.3, random_state = 42)

print(f"Features train size is {features_train.shape}. Number of obs = {len(features_train)}")
print(f"Features test size is {features_test.shape}. Number of obs = {len(features_test)}")
print(f"Target train size is {target_train.shape}. Number of obs = {len(target_train)}")
print(f"Target test size is {target_test.shape}. Number of obs = {len(target_test)}")

results_model1 = [] #saving results of model1

multiple_linear_model1 = LinearRegression()
multiple_linear_model1.fit(features_train, target_train)
prediction_model1 = multiple_linear_model1.predict(features_test)
score_model1 = r2_score(target_test, prediction_model1)
results_model1.append(score_model1)

print(f"R^2 in train set is : {r2_score(target_train, multiple_linear_model1.predict(features_train))}")
print(f"R^2 in test set is : {score_model1}")
print("-"* 50)
print(f"Mean Absolute Error in train set is : {mean_absolute_error(target_train, multiple_linear_model1.predict(features_train))}")
print(f"Mean Absolute Error in test set is : {mean_absolute_error(target_test, prediction_model1)}")
print("-" * 50)
print(f"Mean Squarred Error in train set is : {mean_squared_error(target_train, multiple_linear_model1.predict(features_train))}")
print(f"Mean Squarred Error in test set is : {mean_squared_error(target_test, prediction_model1)}")
print("-"*50)
print(f"Root Mean Squarred Error in train set is : {np.sqrt(mean_squared_error(target_train, multiple_linear_model1.predict(features_train)))}")
print(f"Root Mean Squarred Error in test set is : {np.sqrt(mean_squared_error(target_test, prediction_model1))}")

#Cross Validation for multiple Linear model 1
#5-foldCV  using statistically significant variable 
cross_validation_r2 = cross_val_score(LinearRegression(), features_train, target_train,
                                     scoring = "r2", cv = 5) #r^2 metric
cross_validation_mse = cross_val_score(LinearRegression(), features_train, target_train,
                                     scoring = "neg_mean_squared_error", cv = 5)
mean_CV_r2 = np.mean(cross_validation_r2)
mean_CV_MSE = np.mean(cross_validation_mse)

print(f"5-Folf Cross Validation using as metric R^2 is : {mean_CV_r2 }")
print(f"5-Folf Cross Validation using as metric RMSE is : {np.sqrt(-mean_CV_MSE)}")

#Model1. Visualizations for multiple linear model1

fig, axes = plt.subplots(1, 2, figsize = (12, 6))
fig.suptitle('Multiple Linear Regression Model no 1', fontweight = 'bold', fontsize = 14)

#Actual VS predicte values
df_mult_linear_regression_model1 = pd.DataFrame({"Actual" : target_test, "Predictions" : prediction_model1})
sns.regplot(x = df_mult_linear_regression_model1['Actual'],
            y = df_mult_linear_regression_model1['Predictions'], color = "red", ax = axes[0]);
axes[0].set_title("Predictions VS Actual", fontweight = "bold")

#Cross Validations R^2 and Train R^2
x = ['R_Square : 5-Fold CV', "R_Square : Train Set"]
y = [mean_CV_r2, r2_score(target_train, multiple_linear_model1.predict(features_train))]

sns.barplot(x = x, y = y, ax = axes[1])
axes[1].set_title("R Square in Train subset and after 5-Fold CV", fontweight = "bold");

#Coefficients and intercept
print(f"Intercept : \n {multiple_linear_model1.intercept_}")
print(f"Coefficients : \n {multiple_linear_model1.coef_}")
print("-"*63)

# Predictions for a random sample : Model1
features_prediction = features.copy() # I made a copy with scaled features for model1 
target_prediction = target.copy()
target_prediction = pd.DataFrame(target_prediction.tail(100)).reset_index(drop = True)
features_prediction = features_prediction.tail(100).reset_index(drop = True) # 120 sample random
features_prediction #Scaled sample with StandardScaler for the last 100 values

#Predictions by random selection of features
predicted_values_randomly_selected = pd.DataFrame(multiple_linear_model1.predict(features_prediction))
features_prediction['observed_charges'] = target_prediction
features_prediction['predicted'] = predicted_values_randomly_selected
features_prediction.head()


plt.figure(figsize = (10, 6))
sns.regplot(x = features_prediction['observed_charges'], y = features_prediction['predicted'], 
           color = 'lightblue')
plt.title('Random Features Prediction', fontweight = "bold");

#Multiple linear model2
features_robust #features have been scaled with Robust Scaler
features_robust.drop('region', axis = 1, inplace = True)
target_robust #target has not been normalized

#splitting into train and test set
featuresRobust_train, featuresRobust_test, targetRobust_train, targetRobust_test = train_test_split(
   features_robust, target_robust, test_size = 0.3, random_state = 42)

print(f"Features train size is {featuresRobust_train.shape}. Number of obs = {len(featuresRobust_train)}")
print(f"Features test size is {featuresRobust_test.shape}. Number of obs = {len(featuresRobust_test)}")
print(f"Target train size is {targetRobust_train.shape}. Number of obs = {len(targetRobust_train)}")
print(f"Target test size is {targetRobust_test.shape}. Number of obs = {len(targetRobust_test)}")

#Training Multiple_Linear_Model2
results_model2 = [] #saving results of model1

multiple_linear_model2 = LinearRegression()
multiple_linear_model2.fit(featuresRobust_train, targetRobust_train)
prediction_model2 = multiple_linear_model2.predict(featuresRobust_test)
score_model2 = r2_score(targetRobust_test, prediction_model2)
results_model2.append(score_model2)

#Cross Validation for multiple Linear model 2
#5-foldCV 
cross_validation_r2_model2 = cross_val_score(LinearRegression(), featuresRobust_train, targetRobust_train,
                                     scoring = "r2", cv = 5) #r^2 metric
cross_validation_mse_model2 = cross_val_score(LinearRegression(), featuresRobust_train, targetRobust_train,
                                     scoring = "neg_mean_squared_error", cv = 5)
mean_CV_r2_model2 = np.mean(cross_validation_r2_model2)
mean_CV_MSE_model2 = np.mean(cross_validation_mse_model2)

print(f"5-Folf Cross Validation using as metric R^2 is : {mean_CV_r2_model2}")
print(f"5-Folf Cross Validation using as metric RMSE is : {np.sqrt(-mean_CV_MSE_model2)}")

#Model2. Visualizations for multiple linear model2

fig, axes = plt.subplots(1, 2, figsize = (12, 6))
fig.suptitle('Multiple Linear Regression Model no 2', fontweight = 'bold', fontsize = 14)

#Actual VS predicte values
df_mult_linear_regression_model2 = pd.DataFrame({"Actual" : targetRobust_test, "Predictions" : prediction_model2})
sns.regplot(x = df_mult_linear_regression_model2['Actual'],
            y = df_mult_linear_regression_model2['Predictions'], color = "red", ax = axes[0]);
axes[0].set_title("Predictions VS Actual in MLR2", fontweight = "bold")

#Cross Validations R^2 and Train R^2
x_model2 = ['R_Square : 5-Fold CV', "R_Square : Train Set"]
y_model2 = [mean_CV_r2_model2, r2_score(targetRobust_train, multiple_linear_model2.predict(featuresRobust_train))]

sns.barplot(x = x_model2, y = y_model2, ax = axes[1])
axes[1].set_title("R Square in Train subset and after 5-Fold CV in MLR2", fontweight = "bold");

#Coefficients and intercept : Model 2
print(f"Intercept : \n {multiple_linear_model2.intercept_}")
print(f"Coefficients : \n {multiple_linear_model2.coef_}")
print("-"*63)
