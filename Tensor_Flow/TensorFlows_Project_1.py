#Run Cell
#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# %%

#Importing the housing dataframe

df = pd.read_csv('/Users/nicholaszavala/Documents/Python/Datasets/Udemy/Python_DataScience/Tensor Flows/kc_house_data.csv')
# %%

df.head(15)
# %%

df.info()
# %%

df.describe().transpose()
# %%

df.isnull().sum()
# %%

#Basic visual exploritory analysis with the data

plt.figure(figsize=(10,6))
sns.displot(df['price'],kde=True,bins=75)
# %%
sns.countplot(df['bedrooms'])
# %%
help(sns.countplot)
# %%

sns.countplot(data=df,x='bedrooms')
# %%

df.corr()['price'].sort_values(ascending=False)
# %%
plt.figure(figsize=(12,8))
sns.scatterplot(data=df,y='lat',x='long',hue='price')
# %%

df.sort_values('price',ascending=False).head(20)
# %%

#Creating a dataframe that does not include the top 1 percent

non_top_1_perc = df.sort_values('price',ascending=False).iloc[216:]
# %%

plt.figure(figsize=(12,8))
sns.scatterplot(data=non_top_1_perc,y='lat',x='long',hue='price',
                edgecolor=None,alpha=0.5,palette='RdYlGn')
# %%

df.columns
# %%

df = df.drop('id',axis=1)
# %%

df.head(10)
# %%

df['date'] = pd.to_datetime(df['date'])
# %%

df['date']
# %%
df['year'] = df['date'].apply(lambda date: date.year)
df['month'] = df['date'].apply(lambda date: date.month)
# %%
df[['date','year','month']]

df.head(15)
# %%

plt.figure(figsize=(12,8))
sns.boxplot(data=df,x='month',y='price')

# %%

#Finding the average price per month, use 'groupby'

df.groupby('month').mean()['price'].round(2).plot()
# %%


df = df.drop('date',axis=1)
# %%
df.head(15)
# %%
df['zipcode'].value_counts().sort_values(ascending=False)
# %%

df = df.drop('zipcode',axis=1)
# %%

df.head(10)
# %%

df['yr_renovated'].value_counts()
# %%

#setting up X and y for train test split

X = df.drop('price',axis=1).values
y = df['price'].values
# %%

from sklearn.model_selection import train_test_split
# %%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
# %%

from sklearn.preprocessing import MinMaxScaler
# %%

scaler = MinMaxScaler()
# %%

X_train = scaler.fit_transform(X_train)
# %%

X_test = scaler.transform(X_test)
# %%

from keras.models import Sequential
from keras.layers import Dense
# %%

#Building the model using the 'sequential' model

model = Sequential()

model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')
# %%

model.fit(x=X_train,y=y_train,
          validation_data=(X_test,y_test),
          batch_size=128, epochs=400)
# %%

losses = pd.DataFrame(model.history.history)
# %%

losses.plot()
# %%

from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score
# %%

predictions = model.predict(X_test)
# %%

predictions
# %%

mean_squared_error(y_test,predictions)
# %%

mean_absolute_error(y_test,predictions)
# %%

explained_variance_score(y_test,predictions)
# %%

plt.figure(figsize=(12,8))
plt.scatter(y_test,predictions)
plt.plot(y_test,y_test,'r')
# %%

single_house = df.drop('price',axis=1).iloc[0]
# %%

single_house = scaler.transform(single_house.values.reshape(-1,19))
# %%

model.predict(single_house)
# %%
