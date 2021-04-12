#!/usr/bin/env python
# coding: utf-8

# # TASK 1: Prediction Using Supervised ML
# # By Srishti Todi

# # 

# # Importing Libraries and reading file

# In[132]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


url= "http://bit.ly/w-data"
df=pd.read_csv(url)


# In[5]:


df.head(10)


# # Understanding data 

# In[6]:


df.info()


# In[15]:


df.shape


# In[7]:


df.describe()


# In[11]:


df.boxplot("Hours")


# In[12]:


df.boxplot("Scores")


# In[18]:


df.isnull().sum()


# ### From the above, we see that there are no outliers and no null values. Thus, our data is ready for modelling.

# # Plotting the points

# In[26]:


x=df.Hours
y=df.Scores
plt.xlabel("Hours studied")
plt.ylabel("Percentage Score")
plt.title("Hours vs Pecentage")
plt.scatter(x,y)
plt.show()


# ### From the graph we can see that x and y points are linearly related.

# # Preparing the data

# In[27]:


X=df["Hours"].values
X=X.reshape(-1,1)
Y=df["Scores"].values
Y=Y.reshape(-1,1)


# In[30]:


from sklearn.model_selection import train_test_split


# In[130]:


X_train,X_test,Y_train,Y_test= train_test_split(X,Y,train_size=0.8,random_state=0)


# # Training the algorithm

# In[32]:


from sklearn.linear_model import LinearRegression


# In[33]:


lr=LinearRegression()


# In[34]:


lr.fit(X_train,Y_train)


# In[36]:


lr.intercept_


# In[38]:


lr.coef_


# In[80]:


line1= lr.predict(X)


# In[81]:


plt.scatter(x,y)
plt.plot(x,line1,color = "r")
plt.show()


# # Making predictions

# In[104]:


Y_pred=lr.predict(X_test)


# In[122]:


Y_test.reshape(1,-1), Y_pred.reshape(1,-1)


# In[100]:


final_pred=lr.predict([[9.25]])
final_pred


# ### The person who has studied for 9.25 hours will score 91.5478073 percentage.

# # Evaluating the algorithm

# In[76]:


from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[67]:


mean_squared_error(Y_pred, Y_test)


# In[68]:


mean_absolute_error(Y_pred, Y_test)


# In[ ]:




