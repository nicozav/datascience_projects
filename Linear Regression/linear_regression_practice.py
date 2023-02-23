#Run Cell
#%%
from cmath import sqrt
from inspect import BoundArguments
from json import load
from math import dist
from statistics import linear_regression
import pandas as pd
import numpy as np

# %%
import matplotlib.pyplot as plt
import seaborn as sns
# %%

#importing the csv that will be used for the data frame

data_1 = pd.read_csv('USA_Housing.csv')
# %%

#Confirming and querying data frame information

data_1.info()

data_1.describe()

data_1.head(15)
# %%

data_1.describe()
# %%
data_1.columns
# %%

#Examining initial visualizations

sns.pairplot(data_1)
# %%

sns.displot(data=data_1, x='Price', kde=True)
# %%

sns.heatmap(data_1.corr(), annot=True)
# %%
data_1.columns
# %%

#Splitting the data frame into two different datasets for training and predictions

X = data_1[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]

y = data_1['Price']
# %%

#Importing train_test_split to for data training and linear regression functions

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

# %%
lm = LinearRegression()
# %%
lm.fit(X_train, y_train)
# %%

print(lm.intercept_)
# %%
lm.coef_
# %%

#Setting the [X] data frame

cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])
# %%
cdf
# %%

#Importing the boston dataset

from sklearn.datasets import load_boston
# %%

boston = load_boston()
# %%

boston.keys()
# %%

print(boston['DESCR'])
# %%
predictions = lm.predict(X_test)
# %%
predictions
# %%
plt.scatter(y_test,predictions)
# %%

#Find the difference between the [y_test] values and [predictions] values
    #Using a histogram to find the distribution of the differences
    
sns.displot((y_test - predictions), kde = True)
# %%

figsize_1 = (15,10)
# %%

#Finding the mean absolute error using the sklearn package

from sklearn import metrics
# %%

metrics.mean_absolute_error(y_test, predictions)
# %%
metrics.mean_squared_error(y_test, predictions)
# %%

#Finding the squareroot of the mean squared error

np.sqrt(metrics.mean_squared_error(y_test, predictions))
# %%

