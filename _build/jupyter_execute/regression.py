#!/usr/bin/env python
# coding: utf-8

# https://www.kaggle.com/misterkix/boston-house-price-analysis/notebook

# Regression Metrics
# 
# Mean Absolute Error.
# Mean Squared Error.
# Root Mean Squared Error.
# Root Mean Squared Logarithmic Error.
# R Square.
# Adjusted R Square.

# Alternative: https://www.kaggle.com/shreayan98c/boston-house-price-prediction

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


names = ['CRIM', 'ZN','INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv('housing.csv', header=None, delimiter='\s+', names=names)

df.info()


# In[37]:


# Importing the Boston Housing dataset
from sklearn.datasets import load_boston
boston = load_boston()
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names
data.head()
#Adding target variable to dataframe
data['PRICE'] = boston.target 
# Median value of owner-occupied homes in $1000s
data.shape


# In[39]:


df.head()


# In[8]:


print(df.isnull().sum())


# In[9]:


df.describe()


# In[11]:


plt.figure(figsize=(20,15))
sns.heatmap(df.corr(), annot=True)


# In[12]:


X= df.drop(columns='MEDV')
Y= df.MEDV


# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)


# In[26]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)
y_pred_tree = tree_reg.predict(X_test)
r2_score(y_test, y_pred_tree)


# In[32]:


from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train,y_train)
y_pred_linreg = linreg.predict(X_test)
r2_score(y_test, y_pred_linreg)


# In[34]:


from sklearn import metrics
print(metrics.mean_absolute_error(y_test, y_pred_linreg))
print(metrics.mean_squared_error(y_test, y_pred_linreg))


# In[30]:


from math import sqrt

print(sqrt(metrics.mean_squared_error(y_test, y_pred_linreg)))

print(metrics.r2_score(y_test, y_pred_linreg))


# In[43]:


plt.scatter(y_test, y_pred_linreg)
plt.xlabel("MEDV")
plt.ylabel("Predicted MEDV")
plt.title("MEDV vs Predicted MEDV")
plt.show()


# In[22]:


from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)
y_pred_tree = tree_reg.predict(X_test)
r2_score(y_test, y_pred_tree)

