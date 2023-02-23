#Run Cell
#%%
import pandas as pd
import numpy as np
import seaborn as sns
# %%
#Loading seaborn dataset for practicing

tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')
iris = sns.load_dataset('iris')

tips.head(15)
flights.head(15)
iris.head(15)
# %%
#Creating a distribution plot to test 'tips' dataset

sns.displot(tips['total_bill'],kde=0,bins=30)
# %%
# Testing a jointplots with the practice dataset

tips.info()

sns.jointplot(x='total_bill',y='tip',data=tips,kind='hex')

#Make sure to not include single quote when calling the 'data' parameter in the function
# %%

#More testing with the jointplot function

sns.jointplot(x='total_bill',y='tip',data=tips,kind='reg')

#Make sure to not include single quote when calling the 'data' parameter in the function
# %%

#Practice with the pairplot function

sns.pairplot(tips,hue='sex',palette='dark')
# %%

#Practice with the rugplot function

sns.rugplot(tips['total_bill'])
# %%

#Practice with bar plots

sns.barplot(x='sex',y='total_bill',data=tips,estimator=np.std)
# %%

#Practice with countplots

sns.countplot(x='sex',data=tips)
# %%

#Practice with boxplots

sns.boxplot(x='day',y='total_bill',data=tips,hue='smoker')
# %%

#Practice with violin plots

sns.violinplot(x='day',y='total_bill',data=tips)
# %%

#Practice with strip plots

sns.stripplot(x='day',y='total_bill',data=tips,jitter=True,hue='sex',split=True)
# %%

#Practice with factor plots

sns.catplot(x='day',y='total_bill',data=tips,kind='bar')
# %%
corr_tips = tips.corr()

corr_tips
# %%

corr_flights = flights.corr()

corr_flights
# %%

#Practing with heatmaps and correlation

sns.heatmap(corr_tips,annot=True,cmap='coolwarm')
# %%

#Pivot flights dataset

pivot_flights = flights.pivot_table(index='month',columns='year',values='passengers')
# %%

#Converting pivot_flights to a heatmap

sns.heatmap(pivot_flights)
# %%

#Practice with the clustermap fucntion

sns.clustermap(pivot_flights, standard_scale=1)
# %%
iris['species'].unique()
# %%

#More practice with pairplot

sns.pairplot(iris)
# %%

#Continued - practice with the pairgrid functions and parameters

from matplotlib import pyplot as plt
# %%
iris_plot = sns.PairGrid(iris)

# %%

iris_plot.map(plt.scatter)
# %%
iris_plot = sns.PairGrid(iris)
iris_plot.map_diag(sns.histplot)
iris_plot.map_upper(plt.scatter)
iris_plot.map_lower(sns.kdeplot)

# %%
tips.info()
# %%

tips_grid = sns.FacetGrid(col='time',row='smoker',data=tips)
# %%
tips_grid = sns.FacetGrid(col='time',row='smoker',data=tips)
tips_grid.map(plt.scatter,'total_bill','tip')
# %%

#Practice with lmplot

sns.lmplot(x='total_bill',y='tip',data=tips)
# %%
