#!/usr/bin/env python
# coding: utf-8

# **Mileage Prediction - Regression Analysis**
# 

# **Objective:**
# 
#   The goal of this regression analysis project is to create a precise and dependable model for predicting vehicle mileage. The aim is to develop a mileage prediction model that minimizes prediction errors by leveraging various vehicle attributes and historical mileage data.

# **Data Source:**
# 
# 
# 

# **Import Library**

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


import seaborn as sns


# **Import Data**

# In[5]:


df = pd.read_csv("https://github.com/YBI-Foundation/Dataset/raw/main/MPG.csv")


# In[6]:


df.head()


# In[7]:


df.nunique()


# **Describe Data**

# In[8]:


df.describe()


# **Data visualization**

# In[9]:


sns.pairplot(df,x_vars= ['displacement','horsepower','weight','acceleration','mpg'], y_vars=['mpg']);


# In[10]:


sns.regplot(x = 'displacement', y='mpg', data=df);


# In[11]:


df.info()


# **Data Preprocessing**

# In[12]:


df.corr()


# **Removing Missing Values**

# In[13]:


df = df.dropna()


# In[14]:


df.info()


# **Define Target Variable(y) and Feature Variable(X)**

# In[15]:


df.columns


# In[16]:


y = df['mpg']


# In[17]:


y.shape


# In[18]:


X = df[['displacement', 'horsepower','weight','acceleration']]


# In[19]:


X.shape


# In[20]:


X


# **Scaling Data**

# In[21]:


from sklearn.preprocessing import StandardScaler


# In[22]:


ss = StandardScaler()


# In[23]:


X = ss. fit_transform(X)


# In[24]:


X


# **Train Test Split Data**

# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,random_state=2529)


# In[27]:


X_train.shape, X_test, y_train.shape, y_test.shape


# **Modeling**

# In[28]:


from sklearn.linear_model import LinearRegression


# In[29]:


lr = LinearRegression()


# In[30]:


lr.fit(X_train, y_train)


# In[31]:


lr.intercept_


# In[32]:


lr.coef_


# **Prediction**

# In[33]:


y_pred = lr.predict(X_test)


# In[34]:


y_pred


# **Accuracy**

# In[35]:


from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score


# In[36]:


mean_absolute_error (y_test,y_pred)


# In[37]:


mean_absolute_percentage_error (y_test,y_pred)


# In[38]:


r2_score(y_test,y_pred)


# **Explanation**
# 
#   After the project completion, the model achieved an accuracy of 70%, indicating that it makes correct predictions about 70% of the time. While it may not be perfect, this level of accuracy is still valuable for estimating fuel efficiency in most scenarios. However, continuous efforts can be made to enhance the model's accuracy in future iterations and achieve even better performance.
# 

# In[ ]:





# In[ ]:




