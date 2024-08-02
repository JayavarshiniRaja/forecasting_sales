#!/usr/bin/env python
# coding: utf-8

# In[118]:


# import the necessary library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[119]:


train_data = pd.read_csv("D:/Downloads/forecasting-unit-sales-vit-task-2/train.csv")
train_data.head()


# In[120]:


train_data.tail()


# In[121]:



test_data = pd.read_csv(r"D:\Downloads\forecasting-unit-sales-vit-task-2\test.csv")
test_data.head()


# In[122]:


test_data.tail()


# In[123]:


train_data.shape


# In[124]:


test_data.shape


# In[125]:


train_data.describe()


# In[126]:



test_data.describe()


# In[127]:


train_data.info()


# In[128]:


test_data.info()


# In[ ]:





# In[129]:


train_data.isnull().any()


# In[130]:



test_data.isnull().any()


# In[131]:


train_data['year'] = pd.DatetimeIndex(train_data['date']).year
train_data['month'] = pd.DatetimeIndex(train_data['date']).month
train_data['day'] = pd.DatetimeIndex(train_data['date']).day


# In[132]:


train_data.drop('date',axis=1,inplace=True)


# In[133]:


train_data.drop('ID',axis=1,inplace=True)


# In[134]:


train_data['Item Id'] = train_data['Item Id'].astype(str)


# In[135]:


# Replace null values with the mode of 'Item Name'
mode_item_id = train_data['Item Id'].mode()[0]
train_data['Item Id'].fillna(mode_item_id, inplace=True)


# In[136]:


train_data['Item Name'] = train_data['Item Name'].astype(str)


# In[137]:


# Replace null values with the mode of 'Item Name'
mode_item_name = train_data['Item Name'].mode()[0]
train_data['Item Name'].fillna(mode_item_name, inplace=True)


# In[138]:


mode_units = train_data['units'].mode()[0]
train_data['units'].fillna(mode_units, inplace=True)


# In[139]:


mean_ad_spend = train_data['ad_spend'].mean()
train_data['ad_spend'].fillna(mean_ad_spend, inplace=True)


# In[140]:


mode_item_name = test_data['Item Name'].mode()[0]
test_data['Item Name'].fillna(mode_item_name, inplace=True)


# In[141]:


mean_ad_spend = test_data['ad_spend'].mean()
test_data['ad_spend'].fillna(mean_ad_spend, inplace=True)


# In[142]:


fig=plt.figure(figsize=(5,5))
plt.scatter(train_data['ad_spend'],train_data['units'],color='blue')
plt.xlabel('ad_spend')
plt.ylabel('units')
plt.title('Relation between ad_spend and units')
plt.legend()


# In[143]:


sns.heatmap(train_data.corr())


# In[ ]:





# In[144]:


train_data['units'].hist(bins=3)


# In[148]:


sns.lineplot(x='month',y='ad_spend',data=train_data,color='red')


# In[149]:


from sklearn.preprocessing import LabelEncoder
print("Unique values before encoding:")
print(train_data['Item Name'].unique())

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the 'Item Name' feature
train_data['Item Name'] = label_encoder.fit_transform(train_data['Item Name'])

# Print unique values after encoding
print("\nUnique values after encoding:")
print(train_data['Item Name'].unique())


# In[ ]:





# In[150]:



train_data['Item Id'] = label_encoder.fit_transform(train_data['Item Id'])

train_data['anarix_id'] = label_encoder.fit_transform(train_data['anarix_id'])


# In[ ]:





# In[115]:





# In[ ]:





# In[116]:





# In[151]:


train_data.head()


# In[152]:


x = train_data.drop('units', axis=1).values

# Select the 'units' column for y
y = train_data['units'].values


# In[153]:


x


# In[154]:


y


# In[155]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[156]:


x_train.shape


# In[157]:


y_train.shape


# In[158]:


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor


# In[159]:


DecisionTreeRegressor()
df_grid = GridSearchCV(DecisionTreeRegressor(),param_grid = {'criterion':['mse', 'friedman_mse', 'mae', 'poisson'],'splitter': ['best', 'random'],'max_depth': range(1, 11),'min_samples_split': range(10, 60, 10),},cv=5,n_jobs=1,scoring='neg_mean_squared_error')
df_grid.fit(x_train, y_train)
print(df_grid.best_params_)


# In[160]:


df=DecisionTreeRegressor(criterion='mse',max_depth=10,min_samples_split=10,splitter='best')
df.fit(x_train,y_train)


# In[161]:


y_pred_df=df.predict(x_test)
y_pred_df


# In[162]:


y_test


# In[163]:


from sklearn.metrics import r2_score
accur_df=r2_score(y_test,y_pred_df)
print(accur_df)


# In[164]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=1, random_state=10)
rfr.fit(x_train,y_train)
y_pred_rfr=rfr.predict(x_test)
y_pred_rfr


# In[165]:


y_test


# In[166]:


accur_rfr=r2_score(y_test,y_pred_rfr)
print(accur_rfr)


# Model Evaluation for Decision tree Regressor

# In[168]:


from sklearn.metrics import mean_absolute_error,mean_squared_error
mean_absolute_error(y_test,y_pred_df)


# In[169]:


mean_squared_error(y_test,y_pred_df)


# In[170]:


from math import sqrt
sqrt(mean_squared_error(y_test,y_pred_df))


# In[ ]:


Model Evaluation for Random Forest Regressor¶


# In[171]:


mean_absolute_error(y_test,y_pred_rfr)


# In[172]:


mean_squared_error(y_test,y_pred_rfr)


# In[173]:


sqrt(mean_squared_error(y_test,y_pred_rfr))


# In[ ]:




