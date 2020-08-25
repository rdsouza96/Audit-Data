# -*- coding: utf-8 -*-
"""Project 1 (6).ipynb

# <font color='red'> 
## Rebecca D'souza



### Project description:
- Please read the Data Set Information section to learn about this dataset. 
- Data description is also provided for the dataset.
- Read data into Jupyter notebook, use pandas to import data into a data frame
- Preprocess data: Explore data, check for missing data and apply data scaling. Justify the type of scaling used.

### Regression Task:
- Apply all the regression models you've learned so far. If your model has a scaling parameter(s) use Grid Search to find the best scaling parameter. Use plots and graphs to help you get a better glimpse of the results. 
- Then use cross validation to find average training and testing score. 
- Your submission should have at least the following regression models: KNN repressor, linear regression, Ridge, Lasso, polynomial regression, SVM both simple and with kernels. 
- Finally find the best regressor for this dataset and train your model on the entire dataset using the best parameters and predict buzz for the test_set.

### Classification task:
- Decide aboute a good evaluation strategy and justify your choice.
- Find best parameters for following classification models: KNN classifcation, Logistic Regression, Linear Supprt Vector Machine, Kerenilzed Support Vector Machine, Decision Tree. 
- Which model gives the best results?

### Deliverables:
- Submit IPython notebook. Use markdown to provide an inline comments for this project.
- Submit only one notebook. Before submitting, make sure everything runs as expected. To check that, restart the kernel (in the menubar, select Kernel > Restart) and then run all cells (in the menubar, select Cell > Run All).
- Visualization encouraged. 

### Questions regarding project:
- Post your queries related to project on discussion board on e-learning. There is high possibility that your classmate has also faced the same problem and knows the solution. This is an effort to encourage collaborative learning and also making all the information available to everyone. We will also answer queries there. We will not be answering any project related queries through mail.

---
### Data Set Information:
This dataset is taken from a research explained here. 

The goal of the research is to help the auditors by building a classification model that can predict the fraudulent firm on the basis the present and historical risk factors. The information about the sectors and the counts of firms are listed respectively as Irrigation (114), Public Health (77), Buildings and Roads (82), Forest (70), Corporate (47), Animal Husbandry (95), Communication (1), Electrical (4), Land (5), Science and Technology (3), Tourism (1), Fisheries (41), Industries (37), Agriculture (200).

There are two csv files to present data. Please merge these two datasets into one dataframe. All the steps should be done in Python. Please don't make any changes in csv files. Consider ``Audit_Risk`` as target columns for regression tasks, and ``Risk`` as the target column for classification tasks. 

### Attribute Information:
Many risk factors are examined from various areas like past records of audit office, audit-paras, environmental conditions reports, firm reputation summary, on-going issues report, profit-value records, loss-value records, follow-up reports etc. After in-depth interview with the auditors, important risk factors are evaluated and their probability of existence is calculated from the present and past records.


### Relevant Papers:
Hooda, Nishtha, Seema Bawa, and Prashant Singh Rana. 'Fraudulent Firm Classification: A Case Study of an External Audit.' Applied Artificial Intelligence 32.1 (2018): 48-64.

### Importing required packages
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""### Loading the data sets"""

audit_risk_data = pd.read_csv('audit_risk.csv')
trial_data = pd.read_csv('trial.csv')

"""### Merging the two datasets"""

# Not considering duplicate columns in trial_data when merging
common_columns = audit_risk_data.columns.intersection(trial_data.columns).tolist()
merged_audit_risk_data = pd.merge(audit_risk_data, trial_data, how = 'inner', left_on=common_columns, right_on=common_columns)

merged_audit_risk_data.head()

"""### Checking if there is any null value"""

merged_audit_risk_data.info()

"""##### After performing the merge on common columns, we see that only 629 observations out of the 776 observations could be matched in both the datasets.
##### We can see that there is one observation for which Money_Value is null, which we'll mark as NaN

##### As we can see that Location_ID is the only categorical variable (object), which is coded as an integer, but there is probability that some of the values may not be numeric, we'll check for such values and mark them as NaN
"""

merged_audit_risk_data = merged_audit_risk_data.replace(r'[^\d.]+',np.nan,regex=True)
merged_audit_risk_data.info()

merged_audit_risk_data.isna().any()

merged_audit_risk_data = merged_audit_risk_data.dropna()

merged_audit_risk_data.info()
merged_audit_risk_data.describe()

"""## Regression

### Feature Selection
"""

# Commented out IPython magic to ensure Python compatibility.
X = merged_audit_risk_data.loc[:, ~merged_audit_risk_data.columns.isin(['Audit_Risk', 'Risk'])]
y = merged_audit_risk_data.loc[:, merged_audit_risk_data.columns.isin(['Audit_Risk', 'Risk'])]

from sklearn.ensemble import ExtraTreesRegressor

model = ExtraTreesRegressor(random_state=0)
model.fit(X, y['Audit_Risk'])

# %matplotlib inline
def plot_feature_importances(model):
    plt.figure(figsize=(10,8))
    n_features = X.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

plot_feature_importances(model)

"""### Checking for correlation"""

corr_matrix = merged_audit_risk_data.corr().abs()
high_corr_var = np.where(corr_matrix>0.9)
high_corr_var=[(corr_matrix.columns[x],corr_matrix.columns[y]) for x,y in zip(*high_corr_var) if x!=y and x<y]
high_corr_var

"""Now, from the above features with the most importance, we had computed  
1. Total, PARA_B and Risk_B are more than 90% correlated
2. Inherent_Risk, Risk_D and Money_Value are more than 90% correlated
3. Prob and History_Score are more than 90% correlated

We therefore,
Drop PARA_B and Risk_B in favor of Total
Drop Risk_D and Money_Value in favor of Inherent_Risk
Drop History_Score and Money_Value in favor of Prob

Final Features to be used for regression are as under:
TOTAL, Inherent_Risk, Prob, District_Loss, Score, CONTROL_RISK, Score_MV
"""

features_for_regression = ['TOTAL','Inherent_Risk', 'Prob', 'Score', 'CONTROL_RISK', 'District_Loss', 'Score_MV']
X = X.loc[:, features_for_regression]

"""### Outlier Detection"""

from pandas.plotting import scatter_matrix
scatter_matrix(X, figsize = (15,15), c = y['Risk'], alpha = 0.8, marker = 'O')

green_diamond = dict(markerfacecolor='g', marker='D')

fig1 = plt.figure(figsize=(10,10))
ax1 = fig1.add_subplot(111)
ax1.boxplot(X.values, flierprops=green_diamond)

from scipy import stats
data_outliers_removed = merged_audit_risk_data[(np.abs(stats.zscore(merged_audit_risk_data.loc[:, features_for_regression])) < 3).all(axis=1)]

X = data_outliers_removed.loc[:, features_for_regression]
y = data_outliers_removed.loc[:, ['Audit_Risk', 'Risk']]

"""### Splitting the data into training-validation and test data sets"""

from sklearn.model_selection import train_test_split

X_trainval_org, X_test_org, y_trainval, y_test = train_test_split(X, y, random_state = 0, test_size = 0.2)

# We have two output variables Audit_Risk and Risk
# Assessing Audit_Risk is a regression problem whereas assessing Risk is a binary classification problem
y_reg_trainval = y_trainval['Audit_Risk']
y_reg_test = y_test['Audit_Risk']

y_cls_trainval = y_trainval['Risk'].astype(np.int64)
y_cls_test = y_test['Risk'].astype(np.int64)

"""### Scaling the training-validation and test data sets using Standard Scaler"""

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_trainval = scaler.fit_transform(X_trainval_org)
X_test = scaler.fit_transform(X_test_org)

"""### Stratified 5-folds Cross-Validation 
We will be using Stratified 5-folds Cross-Validation such that each of our test fold will have 20 % of the data
"""

from sklearn.model_selection import KFold
kfold = KFold(n_splits=5)

"""### Evaluation Strategy
##### For regression, we will assess the model based on how well it is able to explain the variance in the output variable from the input set, therefore, we will use R2 score for this, wherein the model with the best R2 score will be the best.
##### Also, the difference between the actual value and the predicted value of the output variable should be minimum, therefore we will use MSE (Minimum Squared Error), wherein the model with the least MSE or the highest Negative MSE will be the best.

### K Nearest Neigbors Regressor

We will first have to ascertain what should be the ideal value of k, such that we get the most accuracy.
We will determine the mean Mean Squared Error (MSE) of the 5-fold cross-validation scheme for different values of k.
The ideal k would be the one which gives the least mean MSE of the 5-fold cross-validation scheme.
"""

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV


scoring = {'R2': 'r2', 'MSE': 'neg_mean_squared_error'}
param_grid = {'n_neighbors': range(1, 20)}

grid_search = GridSearchCV(KNeighborsRegressor(), param_grid, cv=kfold, return_train_score=True, scoring=scoring, refit='R2')
grid_search.fit(X_trainval, y_reg_trainval)

print("Best parameters: {}".format(grid_search.best_params_))

print("Best Mean Train MSE: {:.4f}".format(grid_search.cv_results_['mean_train_MSE'][grid_search.best_index_]))
print("Best Mean Train R2: {:.4f}".format(grid_search.cv_results_['mean_train_R2'][grid_search.best_index_]))
print("Best Mean Validation MSE: {:.4f}".format(grid_search.cv_results_['mean_test_MSE'][grid_search.best_index_]))
print("Best Mean Validation R2: {:.4f}".format(grid_search.cv_results_['mean_test_R2'][grid_search.best_index_]))

results = pd.DataFrame(grid_search.cv_results_)
display(results)

"""We will now confirm the best value of k graphically"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

x_axis = range(1,20)
plt.plot(x_axis, grid_search.cv_results_['mean_train_MSE'], c = 'g', label = 'Mean Train MSE')
plt.plot(x_axis, grid_search.cv_results_['mean_test_MSE'], c = 'b', label = ' Mean Validation MSE')
plt.legend()
plt.xlabel('k')
plt.ylabel('MSE')
plt.show()

x_axis = range(1,20)
plt.plot(x_axis, grid_search.cv_results_['mean_train_R2'], c = 'g', label = 'Mean Train R2')
plt.plot(x_axis, grid_search.cv_results_['mean_test_R2'], c = 'b', label = ' Mean Validation R2')
plt.legend()
plt.xlabel('k')
plt.ylabel('R2')
plt.show()

"""##### Thus, we can see that using K Nearest Neighbors regressor, we can get the best MSE and R2 score when k = 1"""

# Modelling test data on the best knn model with no. of neighbors = 1
model = KNeighborsRegressor(1)
model.fit(X_trainval, y_reg_trainval)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print("Test R2-score: {:.4f}".format(r2_score(y_reg_test, model.predict(X_test))))
print("Test MSE: {:.4f}".format(mean_squared_error(y_reg_test, model.predict(X_test))))

"""### Linear Regression (Ordinary Least Squares)"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

scoring = {'R2': 'r2', 'MSE': 'neg_mean_squared_error'}

lreg = LinearRegression()
lreg.fit(X_trainval, y_reg_trainval)

cv_result = cross_validate(lreg, X_trainval, y_reg_trainval, cv=kfold, scoring=scoring)

print("Mean Train MSE: {:.4f}".format(np.mean(cv_result['train_MSE'])))
print("Mean Train R2: {:.4f}".format(np.mean(cv_result['train_R2'])))

print("Mean Validation MSE: {:.4f}".format(np.mean(cv_result['test_MSE'])))
print("Mean Validation R2: {:.4f}".format(np.mean(cv_result['test_R2'])))

# Modelling test data
model = LinearRegression()
model.fit(X_trainval, y_reg_trainval)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print("Test R2-score: {:.4f}".format(r2_score(y_reg_test, model.predict(X_test))))
print("Test MSE: {:.4f}".format(mean_squared_error(y_reg_test, model.predict(X_test))))

"""### Lasso Regression"""

from  sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

scoring = {'R2': 'r2', 'MSE': 'neg_mean_squared_error'}
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

grid_search = GridSearchCV(Lasso(), param_grid, cv=kfold, return_train_score=True, scoring=scoring, refit='R2')
grid_search.fit(X_trainval, y_reg_trainval)

print("Best parameters: {}".format(grid_search.best_params_))

print("Best Mean Train MSE: {:.4f}".format(grid_search.cv_results_['mean_train_MSE'][grid_search.best_index_]))
print("Best Mean Train R2: {:.4f}".format(grid_search.cv_results_['mean_train_R2'][grid_search.best_index_]))
print("Best Mean Validation MSE: {:.4f}".format(grid_search.cv_results_['mean_test_MSE'][grid_search.best_index_]))
print("Best Mean Validation R2: {:.4f}".format(grid_search.cv_results_['mean_test_R2'][grid_search.best_index_]))

results = pd.DataFrame(grid_search.cv_results_)
display(results)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

x_axis = [0.001, 0.01, 0.1, 1, 10, 100]
plt.plot(x_axis, grid_search.cv_results_['mean_train_MSE'], c = 'g', label = 'Mean Train MSE')
plt.plot(x_axis, grid_search.cv_results_['mean_test_MSE'], c = 'b', label = ' Mean Validation MSE')
plt.legend()
plt.xscale('log')
plt.xlabel(r'$\alpha$')
plt.ylabel('MSE')
plt.show()

x_axis = [0.001, 0.01, 0.1, 1, 10, 100]
plt.plot(x_axis, grid_search.cv_results_['mean_train_R2'], c = 'g', label = 'Mean Train R2')
plt.plot(x_axis, grid_search.cv_results_['mean_test_R2'], c = 'b', label = ' Mean Validation R2')
plt.legend()
plt.xscale('log')
plt.xlabel(r'$\alpha$')
plt.ylabel('R2')

# Modelling test data with Lasso with the best parameter alpha = 0.1
model = Lasso(alpha=0.1)
model.fit(X_trainval, y_reg_trainval)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print("Test R2-score: {:.4f}".format(r2_score(y_reg_test, model.predict(X_test))))
print("Test MSE: {:.4f}".format(mean_squared_error(y_reg_test, model.predict(X_test))))

"""### Ridge Regression"""

from  sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

scoring = {'R2': 'r2', 'MSE': 'neg_mean_squared_error'}
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid_search = GridSearchCV(Ridge(), param_grid, cv=kfold, return_train_score=True, scoring=scoring, refit='R2')
grid_search.fit(X_trainval, y_reg_trainval)

print("Best parameters: {}".format(grid_search.best_params_))

print("Best Mean Train MSE: {:.4f}".format(grid_search.cv_results_['mean_train_MSE'][grid_search.best_index_]))
print("Best Mean Train R2: {:.4f}".format(grid_search.cv_results_['mean_train_R2'][grid_search.best_index_]))
print("Best Mean Validation MSE: {:.4f}".format(grid_search.cv_results_['mean_test_MSE'][grid_search.best_index_]))
print("Best Mean Validation R2: {:.4f}".format(grid_search.cv_results_['mean_test_R2'][grid_search.best_index_]))

results = pd.DataFrame(grid_search.cv_results_)
display(results)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

x_axis = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
plt.plot(x_axis, grid_search.cv_results_['mean_train_MSE'], c = 'g', label = 'Mean Train MSE')
plt.plot(x_axis, grid_search.cv_results_['mean_test_MSE'], c = 'b', label = ' Mean Validation MSE')
plt.legend()
plt.xscale('log')
plt.xlabel(r'$\alpha$')
plt.ylabel('MSE')
plt.show()

x_axis = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
plt.plot(x_axis, grid_search.cv_results_['mean_train_R2'], c = 'g', label = 'Mean Train R2')
plt.plot(x_axis, grid_search.cv_results_['mean_test_R2'], c = 'b', label = ' Mean Validation R2')
plt.legend()
plt.xscale('log')
plt.xlabel(r'$\alpha$')
plt.ylabel('R2')

# Modelling test data with Ridge with the best parameter alpha = 10
model = Ridge(alpha=10)
model.fit(X_trainval, y_reg_trainval)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print("Test R2-score: {:.4f}".format(r2_score(y_reg_test, model.predict(X_test))))
print("Test MSE: {:.4f}".format(mean_squared_error(y_reg_test, model.predict(X_test))))

"""### Stochastic Gradient Descent Regressor"""

from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import GridSearchCV

scoring = {'R2': 'r2', 'MSE': 'neg_mean_squared_error'}
param_grid = {'penalty': ['l1', 'l2', 'elasticnet'],
             'alpha' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}

grid_search = GridSearchCV(SGDRegressor(random_state= 0, max_iter = 100000, tol=-np.infty,learning_rate = 'optimal'), param_grid, cv=kfold, return_train_score=True, scoring=scoring, refit='R2')
grid_search.fit(X_trainval, y_reg_trainval)

print("Best parameters: {}".format(grid_search.best_params_))

print("Best Mean Train MSE: {:.4f}".format(grid_search.cv_results_['mean_train_MSE'][grid_search.best_index_]))
print("Best Mean Train R2: {:.4f}".format(grid_search.cv_results_['mean_train_R2'][grid_search.best_index_]))
print("Best Mean Validation MSE: {:.4f}".format(grid_search.cv_results_['mean_test_MSE'][grid_search.best_index_]))
print("Best Mean Validation R2: {:.4f}".format(grid_search.cv_results_['mean_test_R2'][grid_search.best_index_]))

results = pd.DataFrame(grid_search.cv_results_)
display(results)

# Modelling test data with SGDRegressor with the best parameter alpha = 0.01 and penalty = l2 
model = SGDRegressor(alpha = 0.01, penalty = 'l2')
model.fit(X_trainval, y_reg_trainval)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print("Test R2-score: {:.4f}".format(r2_score(y_reg_test, model.predict(X_test))))
print("Test MSE: {:.4f}".format(mean_squared_error(y_reg_test, model.predict(X_test))))

"""### Linear Support Vector Regressor"""

from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV

scoring = {'R2': 'r2', 'MSE': 'neg_mean_squared_error'}
param_grid = {'C' : [0.01, 0.1, 1, 10, 100]}

grid_search = GridSearchCV(LinearSVR(max_iter = 100000), param_grid, cv=kfold, return_train_score=True, scoring=scoring, refit='R2')
grid_search.fit(X_trainval, y_reg_trainval)

print("Best parameters: {}".format(grid_search.best_params_))

print("Best Mean Train MSE: {:.4f}".format(grid_search.cv_results_['mean_train_MSE'][grid_search.best_index_]))
print("Best Mean Train R2: {:.4f}".format(grid_search.cv_results_['mean_train_R2'][grid_search.best_index_]))
print("Best Mean Validation MSE: {:.4f}".format(grid_search.cv_results_['mean_test_MSE'][grid_search.best_index_]))
print("Best Mean Validation R2: {:.4f}".format(grid_search.cv_results_['mean_test_R2'][grid_search.best_index_]))

results = pd.DataFrame(grid_search.cv_results_)
display(results)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

x_axis = [0.01, 0.1, 1, 10, 100]
plt.plot(x_axis, grid_search.cv_results_['mean_train_MSE'], c = 'g', label = 'Mean Train MSE')
plt.plot(x_axis, grid_search.cv_results_['mean_test_MSE'], c = 'b', label = ' Mean Validation MSE')
plt.legend()
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('MSE')
plt.show()

x_axis = [0.01, 0.1, 1, 10, 100]
plt.plot(x_axis, grid_search.cv_results_['mean_train_R2'], c = 'g', label = 'Mean Train R2')
plt.plot(x_axis, grid_search.cv_results_['mean_test_R2'], c = 'b', label = ' Mean Validation R2')
plt.legend()
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('MSE')

# Modelling test data with Linear Support Vector Regressor with the best parameter C = 1
model = LinearSVR(max_iter = 100000, C=1)
model.fit(X_trainval, y_reg_trainval)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print("Test R2-score: {:.4f}".format(r2_score(y_reg_test, model.predict(X_test))))
print("Test MSE: {:.4f}".format(mean_squared_error(y_reg_test, model.predict(X_test))))

"""### Kernelized Support Vector Regression"""

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

scoring = {'R2': 'r2', 'MSE': 'neg_mean_squared_error'}
param_grid = {'C' : [0.01, 0.1, 1, 10], 
              'gamma' : [0.01, 0.1, 1, 10],
              'kernel' : ['rbf', 'linear']}

grid_search = GridSearchCV(SVR(), param_grid, cv=kfold, return_train_score=True, scoring=scoring, refit='R2')
grid_search.fit(X_trainval, y_reg_trainval)

print("Best parameters: {}".format(grid_search.best_params_))

print("Best Mean Train MSE: {:.4f}".format(grid_search.cv_results_['mean_train_MSE'][grid_search.best_index_]))
print("Best Mean Train R2: {:.4f}".format(grid_search.cv_results_['mean_train_R2'][grid_search.best_index_]))
print("Best Mean Validation MSE: {:.4f}".format(grid_search.cv_results_['mean_test_MSE'][grid_search.best_index_]))
print("Best Mean Validation R2: {:.4f}".format(grid_search.cv_results_['mean_test_R2'][grid_search.best_index_]))

results = pd.DataFrame(grid_search.cv_results_)
display(results)

# Modelling test data with Support Vector Regressor with the best parameter C = 10, gamma=0.1 and kernel = rbf
model = SVR(C = 10, gamma=0.1, kernel = 'rbf')
model.fit(X_trainval, y_reg_trainval)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print("Test R2-score: {:.4f}".format(r2_score(y_reg_test, model.predict(X_test))))
print("Test MSE: {:.4f}".format(mean_squared_error(y_reg_test, model.predict(X_test))))

"""### Polynomial Regression"""

from sklearn.preprocessing  import PolynomialFeatures

train_MSE_list = []
test_MSE_list = []
train_R2_list = []
test_R2_list = []

scoring = {'R2': 'r2', 'MSE': 'neg_mean_squared_error'}

lreg = LinearRegression()

for degree in range(1,7):
    poly = PolynomialFeatures(degree)
    X_trainval_poly = poly.fit_transform(X_trainval)
    X_test_poly = poly.fit_transform(X_test)
    lreg.fit(X_trainval_poly, y_reg_trainval)
    train_MSE_list.append(mean_squared_error(y_reg_trainval, lreg.predict(X_trainval_poly)))
    test_MSE_list.append(mean_squared_error(y_reg_test, lreg.predict(X_test_poly)))
    train_R2_list.append(r2_score(y_reg_trainval, lreg.predict(X_trainval_poly)))
    test_R2_list.append(r2_score(y_reg_test, lreg.predict(X_test_poly)))

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

x_axis = range(1,7)
plt.plot(x_axis, train_MSE_list, c = 'g', label = 'Mean Train Score')
plt.plot(x_axis, test_MSE_list, c = 'b', label = ' Mean Test Score')
plt.legend()
plt.xlabel('Degree of the polynomial')
plt.ylabel('MSE')
plt.show()

x_axis = range(1,7)
plt.plot(x_axis, train_R2_list, c = 'g', label = 'Mean Train Score')
plt.plot(x_axis, test_R2_list, c = 'b', label = ' Mean Test Score')
plt.legend()
plt.xlabel('Degree of the polynomial')
plt.ylabel('R2')
plt.yscale('log')

"""###### It can be clearly seen that we get the best R2 and MSE for both training and test when degree = 2"""

# Modelling test data with polynomial regression of degree 2
poly = PolynomialFeatures(2)
X_trainval_poly = poly.fit_transform(X_trainval)
X_test_poly = poly.fit_transform(X_test)
lreg.fit(X_trainval_poly, y_reg_trainval)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print("Test R2-score: {:.4f}".format(r2_score(y_reg_test, lreg.predict(X_test_poly))))
print("Test MSE: {:.4f}".format(mean_squared_error(y_reg_test, lreg.predict(X_test_poly))))

"""## Classification

### Splitting the data into training-validation and test data sets
"""

X = merged_audit_risk_data.loc[:, ~merged_audit_risk_data.columns.isin(['Audit_Risk', 'Risk'])]
y = merged_audit_risk_data.loc[:, merged_audit_risk_data.columns.isin(['Audit_Risk', 'Risk'])]

from sklearn.model_selection import train_test_split

X_trainval_org, X_test_org, y_trainval, y_test = train_test_split(X, y, random_state = 0, test_size = 0.2)

# We have two output variables Audit_Risk and Risk
# Assessing Audit_Risk is a regression problem whereas assessing Risk is a binary classification problem
y_reg_trainval = y_trainval['Audit_Risk']
y_reg_test = y_test['Audit_Risk']

y_cls_trainval = y_trainval['Risk'].astype(np.int64)
y_cls_test = y_test['Risk'].astype(np.int64)

"""### Scaling the training-validation and test data sets using MinMax Scaler"""

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_trainval = scaler.fit_transform(X_trainval_org)
X_test = scaler.fit_transform(X_test_org)

"""### Feature Selection: Principal Component Analysis (PCA)
Now, we shall perform Principal Component Analysis, and determine what would be the ideal number of Prinicipal Components that would explain most of the variation in the data to deal with the high correlation between different features in the dataset
"""

from sklearn.decomposition import PCA

pca = PCA().fit(X_trainval)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Audit Risk Training-Validation Dataset Explained Variance')
plt.show()

from sklearn.decomposition import PCA

pca = PCA().fit(X_test)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Audit Risk Test Dataset Explained Variance')
plt.show()

"""#### As we can see, selecting about 12 principal components can explain about 99 % variance in both the training-validation and test data sets.
#### Therefore, we select 12 as the no. of PCA components
"""

pca = PCA(n_components = 12)
column_names = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12']
X_trainval = pd.DataFrame(pca.fit_transform(X_trainval), columns = column_names)
X_test = pd.DataFrame(pca.fit_transform(X_test), columns = column_names)

X_trainval.head()

"""### Evaluation Strategy

##### The objective of this classification is to find companies that are risky and needs to be targeted for auditing.
##### So, correctly classifying the relevant class (Risk = 1) is very important, but at the same time, we would not really like many instances wherein the company was not risky (Risk = 0) but was targeted for auditing, as it will require time and money.
##### We require a classifier which is precise (how many instances it classifies correctly), but at the same time robust (it does not miss a significant number of instances).
##### Therefore, we will assess the models based on their 'F-1 Score' which tells which model has both the best precision and recall.

### K Nearest Neighbors Classification
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors': range(1, 20)}

grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=kfold, return_train_score=True, scoring='f1')
grid_search.fit(X_trainval, y_cls_trainval)

print("Best parameters: {}".format(grid_search.best_params_))
print("Best Mean Train F1-score: {:.4f}".format(grid_search.cv_results_['mean_train_score'][grid_search.best_index_]))
print("Best Mean Validation F1-score: {:.4f}".format(grid_search.best_score_))

results = pd.DataFrame(grid_search.cv_results_)
display(results)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

x_axis = range(1,20)
plt.plot(x_axis, grid_search.cv_results_['mean_train_score'], c = 'g', label = 'Mean Train Score')
plt.plot(x_axis, grid_search.cv_results_['mean_test_score'], c = 'b', label = ' Mean Validation Score')
plt.legend()
plt.xlabel('k')
plt.ylabel('F1-score')

"""It can be confirmed graphically as well that at k = 1, we have the best mean validation recall."""

knn = KNeighborsClassifier(1)
knn.fit(X_trainval, y_cls_trainval)

from sklearn.metrics import classification_report
print(classification_report(y_cls_test, knn.predict(X_test)))
print('Accuracy: {:.4f}'.format(knn.score(X_test, y_cls_test)))

# Plotting the separation of classes in the test dataset using the best parameters
from mlxtend.plotting import plot_decision_regions

X_b = X_test.values[:, [0,9]]
y_b = y_cls_test.values[:].astype(np.int64)

knn.fit(X_b, y_b) 

plot_decision_regions(X_b, y_b, clf = knn)

"""### Logistic Regression"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
             'penalty': ['l1', 'l2']}

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=kfold, return_train_score=True, scoring='f1')
grid_search.fit(X_trainval, y_cls_trainval)

print("Best parameters: {}".format(grid_search.best_params_))
print("Best Mean Train F1-score: {:.4f}".format(grid_search.cv_results_['mean_train_score'][grid_search.best_index_]))
print("Best Mean Validation F1-score: {:.4f}".format(grid_search.best_score_))

results = pd.DataFrame(grid_search.cv_results_)
display(results)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

x_axis = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
plt.plot(x_axis, results.loc[results['param_penalty'] == 'l1', 'mean_train_score'], c = 'g', label = 'l1 Mean Train Score')
plt.plot(x_axis, results.loc[results['param_penalty'] == 'l1', 'mean_test_score'], c = 'b', label = 'l1 Mean Validation Score')
plt.legend()
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('F1-score')
plt.show()

x_axis = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
plt.plot(x_axis, results.loc[results['param_penalty'] == 'l2', 'mean_train_score'], c = 'r', label = 'l2 Mean Train Score')
plt.plot(x_axis, results.loc[results['param_penalty'] == 'l2', 'mean_test_score'], c = 'y', label = 'l2 Mean Validation Score')
plt.legend()
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('F1-score')

"""As we can see the best fit for logistic regression is when we use l1 regularization with value of C being 10"""

lreg = LogisticRegression(penalty = 'l1', C = 10)
lreg.fit(X_trainval, y_cls_trainval)

from sklearn.metrics import classification_report
print(classification_report(y_cls_test, lreg.predict(X_test)))
print('Accuracy: {:.4f}'.format(lreg.score(X_test, y_cls_test)))

# Commented out IPython magic to ensure Python compatibility.
# Plotting the separation of classes in the test dataset using the best parameters
# %matplotlib inline

from mlxtend.plotting import plot_decision_regions

X_b = X_test.values[:, [0,9]]
y_b = y_cls_test.values[:].astype(np.int64)

lreg.fit(X_b, y_b) 

plot_decision_regions(X_b, y_b, clf = lreg)

"""### Linear Support Vector Machine"""

from sklearn.svm import LinearSVC

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid_search = GridSearchCV(LinearSVC(), param_grid, cv=kfold, return_train_score=True, scoring='f1')
grid_search.fit(X_trainval, y_cls_trainval)
train_score_array = []
test_score_array = []

print("Best parameters: {}".format(grid_search.best_params_))
print("Best Mean Train F1-score: {:.4f}".format(grid_search.cv_results_['mean_train_score'][grid_search.best_index_]))
print("Best Mean Validation F1-score: {:.4f}".format(grid_search.best_score_))

results = pd.DataFrame(grid_search.cv_results_)
display(results)

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

x_axis = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
plt.plot(x_axis, results['mean_train_score'], c = 'g', label = 'Mean Train Score')
plt.plot(x_axis, results['mean_test_score'], c = 'b', label = 'Mean Validation Score')
plt.legend()
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('F1-score')

"""We can see that the best fit for Linear SVC is obtained at C = 1"""

lsvc = LinearSVC(C = 1)
lsvc.fit(X_trainval,y_cls_trainval)

from sklearn.metrics import classification_report
print(classification_report(y_cls_test, lsvc.predict(X_test)))
print('Accuracy: {:.4f}'.format(lsvc.score(X_test, y_cls_test)))

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
from mlxtend.plotting import plot_decision_regions

X_b = X_test.values[:, [0,9]]
y_b = y_cls_test.values[:].astype(np.int64)

lsvc.fit(X_b, y_b) 

plot_decision_regions(X_b, y_b, clf = lsvc)

"""### Kernelized Support Vector Machine"""

from sklearn.svm import SVC

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                 'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                 'kernel': ['linear', 'rbf', 'poly']}

grid_search = GridSearchCV(SVC(), param_grid, cv=kfold, return_train_score=True, scoring='f1')
grid_search.fit(X_trainval, y_cls_trainval)

print("Best parameters: {}".format(grid_search.best_params_))
print("Best Mean Train F1-score: {:.4f}".format(grid_search.cv_results_['mean_train_score'][grid_search.best_index_]))
print("Best Mean Validation F1-score: {:.4f}".format(grid_search.best_score_))

results = pd.DataFrame(grid_search.cv_results_)
display(results)

"""We can see the best fit is obtained with an rbf kernel with C = 0.1 and Gamma = 10"""

ksvc_rbf = SVC(kernel = 'rbf', C = 0.1, gamma = 10)
ksvc_rbf.fit(X_trainval,y_cls_trainval)

from sklearn.metrics import classification_report
print(classification_report(y_cls_test, ksvc_rbf.predict(X_test)))
print('Accuracy: {:.4f}'.format(ksvc_rbf.score(X_test, y_cls_test)))

# Commented out IPython magic to ensure Python compatibility.
# Plotting the separation of classes in the test dataset using the best parameters
# %matplotlib inline

X_b = X_test.values[:, [0,9]]
y_b = y_cls_test.values[:].astype(np.int64)

ksvc_rbf.fit(X_b, y_b) 

plot_decision_regions(X_b, y_b, clf = ksvc_rbf)

"""### Decision Tree Classifier"""

from sklearn.tree import DecisionTreeClassifier

param_grid = {'max_depth': range(1, 10)}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state = 0), param_grid, cv=kfold, return_train_score=True, scoring='f1')
grid_search.fit(X_trainval, y_cls_trainval)

print("Best parameters: {}".format(grid_search.best_params_))
print("Best Mean Train F1-score: {:.4f}".format(grid_search.cv_results_['mean_train_score'][grid_search.best_index_]))
print("Best Mean Validation F1-score: {:.4f}".format(grid_search.best_score_))

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

x_axis = range(1,10)
plt.plot(x_axis, grid_search.cv_results_['mean_train_score'], c = 'g', label = 'Mean Train Score')
plt.plot(x_axis, grid_search.cv_results_['mean_test_score'], c = 'b', label = ' Mean Validation Score')
plt.legend()
plt.xlabel('Max Depth')
plt.ylabel('F1-score')

"""We can see that the best fit is obtained with max depth = 2"""

dtree = DecisionTreeClassifier(random_state = 0, max_depth = 2)
dtree.fit(X_trainval, y_cls_trainval)

print(classification_report(y_cls_test, dtree.predict(X_test)))
print('Accuracy: {:.4f}'.format(dtree.score(X_test, y_cls_test)))

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

X_b = X_test.values[:, [0,9]]
y_b = y_cls_test.values[:].astype(np.int64)

dtree.fit(X_b, y_b) 

plot_decision_regions(X_b, y_b, clf = dtree)

"""### Conclusion

#### Regression

Out of all the models, Support Vector Regressor( C= 10, gamma=0.1, kernel = rbf) gave the best metrics for both validation and test:

###### 1. Validation
Mean Validation MSE: 12.5945
, Mean Validation R2: 0.8785

###### 2. Test
Test R2-score: 0.9249
, Test MSE: 6.1079

#### Classification

Almost all models perform fairly well on the classification task.

K Nearest Neighbors Classifier (n_neighbors = 1) performs the best with the highest metrics for both validation and test:

###### 1. Validation
Mean Validation F1-score: 0.9956

###### 2. Test
                precision    recall  f1-score   support

           0       1.00      1.00      1.00        60
           1       1.00      1.00      1.00        65

Accuracy: 1.0000    
    
However, the decision boundary is not smooth as the model is complex

Therefore, we choose Logistic Regression with almost similar metrics for both validation and test but the decision boundary is smoother

###### 1. Validation
Mean Validation F1-score: 0.9937
    
###### 2. Test  
        precision    recall  f1-score   support

           0       0.98      1.00      0.99        60
           1       1.00      0.98      0.99        65

Accuracy: 0.9920
"""
