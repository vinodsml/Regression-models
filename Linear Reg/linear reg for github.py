#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


dataset=pd.read_csv("housing_prices_SLR.csv")
dataset


# In[3]:


dataset.isnull().any()


# In[6]:


x=dataset.iloc[0:, 0:1]
y=dataset.iloc[0:, 1:]
x=x.values
y=y.values


# In[7]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8, random_state=42)


# In[8]:


from sklearn.linear_model import LinearRegression
linear=LinearRegression()
model=linear.fit(x_train,y_train)
prediction=model.predict(x_train)
prediction


# In[9]:


r=model.score(x_train,y_train)
r


# In[ ]:




