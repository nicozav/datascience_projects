#Run Cell
#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%

#Loading in the titanic training file for the dataframe

train = pd.read_csv('titanic_train.csv')
# %%

train.info()

train.head(5)
# %%

test = pd.read_csv('titanic_test.csv')
# %%

test.info()

test.head(5)
# %%

sns.heatmap(train.isnull(), yticklabels=False, cbar= False, cmap= 'viridis')
# %%
sns.set_style('whitegrid')
# %%

train.info()
# %%

sns.countplot(x='Survived',hue='Pclass',data=train)
 # %%

#Creating a distribution plot with the train dataset

sns.displot(train['Age'].dropna(),bins=30)
# %%

sns.countplot(x='SibSp', data=train)
# %%

#Plotting a historgram using the ['Fare'] column from 'train' dataset

train['Fare'].hist(bins=40, figsize=(12,5))
# %%

import cufflinks as cf
# %%
cf.go_offline()
# %%

train['Fare'].iplot(kind='hist',bins=40)
# %%
plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass',y='Age',data=train)
# %%

#Cleaning the dataset 'train' with average age, function:

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age
# %%

#Applying impute_age(cols) to the 'train' dataset

train['Age'] = train[['Age','Pclass']].apply(impute_age, axis=1)
# %%

#Creating another heatmap to indentify more null values in the columns

sns.heatmap(train.isnull(), yticklabels=False,cbar=False, cmap='viridis')
# %%

train.drop('Cabin', axis=1, inplace=True)
# %%
train.head(5)
# %%

sex = pd.get_dummies(train['Sex'],drop_first=True)
# %%

sex.head(5)
# %%

embark = pd.get_dummies(train['Embarked'],drop_first=True)
# %%

embark.head(5)
# %%

train = pd.concat([train,sex,embark],axis=1)
# %%

train.head(5)
# %%

#Dropping columns from the dataset

train.drop(['Sex','Embarked','Name','Ticket'],axis=1, inplace=True)
# %%

train.tail(5)
# %%

train.drop(['PassengerId'],axis=1,inplace=True)
# %%
train.head(5)
# %%
X = train.drop('Survived', axis=1)
y = train['Survived']
# %%

from sklearn.model_selection import train_test_split
# %%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
# %%

from sklearn.linear_model import LogisticRegression
# %%

logmodel = LogisticRegression()
logmodel = LogisticRegression(max_iter=1000)
# %%

logmodel.fit(X_train,y_train)
# %%

predictions = logmodel.predict(X_test)
# %%

predictions
# %%

from sklearn.metrics import classification_report
# %%

print(classification_report(y_test,predictions))
# %%
