#Run Cell
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%

from sklearn.datasets import make_blobs
# %%

#Creating the data using the 'make_blobs' function

data = make_blobs(n_samples=200,n_features=2,centers=4,cluster_std=1.8,random_state=101)
# %%

plt.scatter(data[0][:,0],data[0][:,1],c=data[1])
# %%

#Importing the 'KMeans' functoin

from sklearn.cluster import KMeans
# %%

kmeans = KMeans(n_clusters=4)
# %%
kmeans.fit(data[0])
# %%
kmeans.cluster_centers_
# %%
kmeans.labels_
# %%

#Creating the subplots for both sets of cluster data

fig , (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(10,6))

ax1.set_title('K Means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')

ax2.set_title('Original')
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')