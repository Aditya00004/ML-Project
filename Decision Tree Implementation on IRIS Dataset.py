#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


from sklearn.datasets import load_iris


# In[4]:


iris=load_iris()


# In[5]:


iris


# In[5]:


iris.data


# In[6]:


iris.target


# In[7]:


import seaborn as sns


# In[8]:


df=sns.load_dataset('iris')


# In[9]:


df.head()


# In[10]:


#independent feature and dependent features
X=df.iloc[:,:-1]
y=iris.target


# In[11]:


X,y


# In[12]:


### train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=0.33, random_state=42)


# In[13]:


X_train


# In[14]:


from sklearn.tree import DecisionTreeClassifier


# In[15]:





# In[18]:


## Postpruning
treemodel=DecisionTreeClassifier(max_depth=2)


# In[19]:


treemodel.fit(X_train,y_train)


# In[20]:


from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(treemodel,filled=True)


# In[21]:


#prediction
y_pred=treemodel.predict(X_test)


# In[22]:


y_pred


# In[23]:


from sklearn.metrics import accuracy_score,classification_report


# In[24]:


score=accuracy_score(y_pred,y_test)
print(score)


# In[25]:


print(classification_report(y_pred,y_test))


# In[33]:


## Preprunning
parameter={
 'criterion':['gini','entropy','log_loss'],
  'splitter':['best','random'],
  'max_depth':[1,2,3,4,5],
  'max_features':['auto', 'sqrt', 'log2']
    
}


# In[34]:


from sklearn.model_selection import GridSearchCV


# In[35]:


treemodel=DecisionTreeClassifier()
cv=GridSearchCV(treemodel,param_grid=parameter,cv=5,scoring='accuracy')

# GridSearchCV first parametr is estimator:which model you want to apply i.e Decision treee in this case
#               second paramter is param_grid:All the parameter must be in key value pair
#               third parametr is cv:(cross validation)


# In[36]:


cv.fit(X_train,y_train)


# In[37]:


cv.best_params_


# In[42]:


y_test


# In[43]:


y_pred


# In[44]:


y_pred=cv.predict(X_test)


# In[45]:


from sklearn.metrics import accuracy_score, classification_report


# In[46]:


score=accuracy_score(y_pred,y_test)


# In[47]:


score


# In[49]:


print(classification_report(y_pred,y_test))


# In[ ]:




