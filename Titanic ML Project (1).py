#!/usr/bin/env python
# coding: utf-8

# In[147]:


# import library

import pandas as pd 
#Panda library help you to read the dataset and most of the data pre processing step will be done with the help of this library.

import numpy as np
#numpy is basially used to work with arrays it may be a single or a multi dimensional array.

import matplotlib.pyplot as plt
#this will be used for visualization

import seaborn as sns 
#it will be also used for visualizatio.


# In[148]:


# read the dataset
train = pd.read_csv('train.csv')


# In[149]:


train.head()


# In[150]:


#EDA


# In[151]:


#missing value
train.isnull()
#true means the value is null


# In[152]:


sns.heatmap(train.isnull(),yticklabels=False)


# In[153]:


sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='viridis')
# cmap is styiling 


# In[154]:


# Data cleaning 


# In[155]:


# we will remove null value from Age and Cabin column.
#before removing null value from age column we will do some analysis.
# there is some relation between pclass and age 
#with respect to passenger class we will find out the average age of the people.
#We will create boxplot
sns.boxplot(x='Pclass', y='Age', data=train)


# In[156]:


#using this boxplot we got the average value of Age with respect to Passenger class.
#Let average value of Age wrt Pclass 1 is 37 and Pcalss 2 is 29 & Pclass 3 is 24
# Based on this boxplot we will replace Null value from Age column.

# we will create a function where Age and Pclass column are input.
def impute_Age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        
        elif Pclass == 2:
            return 29
        
        elif Pclass == 3:
            return 24
    else:
        return Age


# In[157]:


train['Age'] = train[['Age','Pclass']].apply(impute_Age,axis=1)
# for each and very record present in Age and Pclass column, impute_Age function will be applied on them.


# In[158]:


sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='viridis')


# In[159]:


train.drop('Cabin',axis=1, inplace=True)


# In[160]:


train.head()


# In[161]:


sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='viridis')


# In[162]:


train.dropna(inplace=True)
# it will remove the row that contain null values.

#inplace=True
#keyword in a pandas method changes the default behaviour
#such that the operation on the dataframe doesn't return anything


# In[163]:


# Converting Categorical features 

#We will need to convert categorical features to dummy variables using Pandas Otherwise our ML Algo won't be able to directly 
#take those features as input


# In[164]:


train.info()


# In[165]:


sex = pd.get_dummies(train['Sex'], drop_first=True)
sex.head()

# get_dummies function is used to convert categorical variable into dummy variables.
# dummy variable can have value between 0 and 1.

# we can remove the first column because the other two column can represent the first column.

# Male is represented by 1 and female by 0


# In[166]:


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
#axis=1 means column i.e vertical axis
#axis=0 means rows i.e horizontal axis


# In[167]:


train.head()


# In[168]:


train = pd.concat([train,sex],axis=1)


# In[169]:


train.head()
# Survived column is dependent feature and all column are independent feature.
#Our data is ready


# In[170]:


# Building Logistic Regression Model


# In[175]:


from sklearn.model_selection import train_test_split


# In[176]:


x_train, x_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1),train['Survived'],test_size=0.30)


# In[177]:


from sklearn.linear_model import LogisticRegression


# In[178]:


logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)


# In[188]:


#the default solver in LogisticRegression is 'lbfgs' and the maximum number of iterations is 100 by default.
logmodel = LogisticRegression(solver='lbfgs',max_iter=3000)
logmodel.fit(x_train,y_train)


# In[179]:


prediction = logmodel.predict(x_test)


# In[180]:


from sklearn.metrics import confusion_matrix


# In[182]:


accuracy = confusion_matrix(y_test, prediction)


# In[183]:


accuracy


# In[184]:


from sklearn.metrics import accuracy_score


# In[185]:


accuracy = accuracy_score(y_test, prediction)
accuracy


# In[187]:


prediction


# In[ ]:




