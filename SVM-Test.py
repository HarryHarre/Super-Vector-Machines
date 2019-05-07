
# coding: utf-8

# In[10]:


import pandas as pd, numpy as np


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


from sklearn.datasets import load_breast_cancer


# In[6]:


cancer = load_breast_cancer()


# In[7]:


cancer.keys()


# In[8]:


print(cancer['DESCR'])


# In[11]:


df_feat = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])


# In[12]:


df_feat.head(2)


# In[13]:


df_feat.info()


# In[14]:


cancer['target']


# In[15]:


cancer['target_names']


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


X = df_feat
y = cancer['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[18]:


from sklearn.svm import SVC


# In[19]:


model = SVC()


# In[20]:


model.fit(X_train,y_train)


# In[21]:


predictions = model.predict(X_test)


# In[22]:


from sklearn.metrics import classification_report,confusion_matrix


# In[23]:


print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


# In[25]:


from sklearn.model_selection import GridSearchCV


# In[26]:


param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}


# In[28]:


grid = GridSearchCV(SVC(),param_grid,verbose=3)


# In[29]:


grid.fit(X_train,y_train)


# In[30]:


grid.best_params_


# In[31]:


grid.best_estimator_


# In[32]:


grid_predictions = grid.predict(X_test)


# In[33]:


print(confusion_matrix(y_test,grid_predictions))
print('\n')
print(classification_report(y_test,grid_predictions))

