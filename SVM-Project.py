
# coding: utf-8

# In[8]:


import pandas as pd, numpy as np


# In[9]:


import matplotlib.pyplot as plt, seaborn as sns


# In[10]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# The Iris Setosa
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg'
Image(url,width=300,height=300)


# In[5]:


# The Iris Versicolor
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg'
Image(url,width=300,height=300)


# In[6]:


# The Iris Virginica
from IPython.display import Image
url = 'http://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg'
Image(url,width=300,height=300)


# In[11]:


iris = sns.load_dataset('iris')


# In[12]:


iris.head()


# In[96]:


sns.pairplot(iris,hue='species',palette='Dark2')


# In[100]:


setosa = iris[iris.species == "setosa"]
sns.kdeplot(setosa.sepal_width,setosa.sepal_length,cmap='plasma',shade=True,shade_lowest=False)


# In[28]:


flower_feats = ['species']


# In[30]:


final_data = pd.get_dummies(iris,columns=flower_feats,drop_first=True)


# In[33]:


from sklearn.model_selection import train_test_split


# In[80]:


#iris_feats = ['species']
#iris_species = pd.DataFrame(iris['species'])
#iris_species = pd.get_dummies(iris,columns=iris_feats,drop_first=True)


# In[108]:


X = iris.drop('species',axis=1)
y = iris['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[109]:


from sklearn.svm import SVC


# In[110]:


model = SVC()


# In[111]:


model.fit(X_train,y_train)


# In[112]:


predictions = model.predict(X_test)


# In[113]:


from sklearn.metrics import classification_report,confusion_matrix


# In[114]:


print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))


# In[88]:


from sklearn.model_selection import GridSearchCV


# In[115]:


param_grid = {'C':[0.1,1.0,10,100],'gamma':[1.0,0.1,0.01,0.001]}


# In[117]:


grid = GridSearchCV(SVC(),param_grid,verbose=2)


# In[118]:


grid.fit(X_train,y_train)


# In[119]:


grid.best_params_


# In[120]:


grid.best_estimator_


# In[121]:


grid_predictions = grid.predict(X_test)


# In[122]:


print(confusion_matrix(y_test,grid_predictions))
print('\n')
print(classification_report(y_test,grid_predictions))

