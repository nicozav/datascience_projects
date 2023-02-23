#Run Cell
#%%

from operator import concat
import pandas as pd
import numpy as np
import seaborn as sns
# %%

df_1 = pd.read_csv('fake_reg.csv')
# %%

df_1.head(5)
# %%
df_1.describe
# %%
df_1.info()
# %%
df1 = pd.DataFrame(df_1)
# %%

sns.pairplot(df1)
# %%
from sklearn.model_selection import train_test_split
# %%

X = df_1[['feature1','feature2']].values
# %%

y = df_1['price'].values
# %%

#X

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# %%

X_train.shape
# %%
X_test.shape
# %%

from sklearn.preprocessing import MinMaxScaler
# %%
help(MinMaxScaler)
# %%

scaler = MinMaxScaler()
# %%

scaler.fit(X_train)
# %%

X_train = scaler.transform(X_train)
# %%

X_test = scaler.transform(X_test)
# %%

#Installing tensorflow.keras functions

from keras.models import Sequential
from keras.layers import Dense
# %%

help(Sequential)
# %%

#model = Sequential([Dense(4,activation='relu'),
                   # Dense(2,activation='relu'),
                   # Dense(1)])
# %%

#Practice with building sequential models

model = Sequential()

model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))

model.add(Dense(1))

model.compile(optimizer='rmsprop',loss='mse')
# %%

model.fit(x=X_train,y=y_train,epochs=250)
# %%

loss_df = pd.DataFrame(model.history.history)
# %%

loss_df.plot()
# %%

model.evaluate(X_test,y_test,verbose=0)
# %%

test_predictions = model.predict(X_test)
# %%

test_predictions
# %%

test_predictions = pd.Series(test_predictions.reshape(300,))
# %%

test_predictions
# %%

pred_df = pd.DataFrame(y_test,columns=['Test True Y'])
# %%

pred_df.head(5)
# %%

pred_df = pd.concat([pred_df,test_predictions],axis=1)

# %%


pred_df
# %%

pred_df.columns = ['Test True Y', 'Model Predictions']
# %%

pred_df
# %%

sns.scatterplot(data=pred_df,x='Test True Y', y='Model Predictions')
# %%

from sklearn.metrics import mean_absolute_error,mean_squared_error
# %%

mean_absolute_error(pred_df['Test True Y'],pred_df['Model Predictions'])
# %%

mean_squared_error(pred_df['Test True Y'],pred_df['Model Predictions'])**0.5
# %%
