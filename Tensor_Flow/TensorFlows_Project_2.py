#Run Cell
#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# %%

df = pd.read_csv('cancer_classification.csv')
# %%

df.head(10)
# %%
df.describe().transpose()
# %%
df.info()
# %%

sns.countplot(x='benign_0__mal_1',data=df)
# %%

df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')
# %%

plt.figure(figsize=(12,10))
sns.heatmap(df.corr())

# %%

X = df.drop('benign_0__mal_1',axis=1).values
y = df['benign_0__mal_1'].values
# %%

from sklearn.model_selection import train_test_split
# %%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
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
# %%

from keras.layers import Dense, Dropout
# %%

model = Sequential()

model.add(Dense(units=30,activation='relu'))
model.add(Dense(units=15,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam')
# %%

model.fit(x=X_train,y=y_train,epochs=600,validation_data=(X_test,y_test))
# %%

losses = pd.DataFrame(model.history.history)
# %%

losses.head(10)
# %%

losses.plot()
# %%

model = Sequential()

model.add(Dense(units=30,activation='relu'))
model.add(Dense(units=15,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam')
# %%

from keras.callbacks import EarlyStopping
# %%

help(EarlyStopping)
# %%

early_stop = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)
# %%

#Retraining the model to stop early to prevent over training with the datasets

model.fit(x=X_train,y=y_train,epochs=600,validation_data=(X_test,y_test),
          callbacks=[early_stop])
# %%

model_loss = pd.DataFrame(model.history.history)
# %%
model_loss.plot()
# %%

from keras.layers import Dropout
# %%

model = Sequential()

model.add(Dense(units=30,activation='relu'))
model.add(Dropout(rate=0.5))

model.add(Dense(units=15,activation='relu'))
model.add(Dropout(rate=0.5))

model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam')
# %%

model.fit(x=X_train,y=y_train,epochs=600,validation_data=(X_test,y_test),
          callbacks=[early_stop])

# %%

model_loss = pd.DataFrame(model.history.history)
# %%
model_loss.plot()
# %%
predictions = (model.predict(X_test) > 0.5)*1
# %%

from sklearn.metrics import classification_report, confusion_matrix
# %%

print(classification_report(y_test,predictions))
# %%
