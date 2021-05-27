#!/usr/bin/env python
# coding: utf-8

# ## collecting data

# In[9]:


import numpy as np # linear algebra
import pandas as pd #data processing
import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


data=pd.read_csv("online_shoppers_intention.csv")


# In[11]:


data.head(10)


# # data wrangling

# In[13]:


missing=data.isnull().sum()


# In[14]:


missing


# In[33]:


data.fillna(0,inplace=True)


# In[34]:


data


# In[35]:


data.isnull().sum()


# In[43]:


data


# # ANALYZING DATA

# In[36]:


x = data.iloc[:, [5, 6]].values
x.shape


# In[37]:


x


# In[39]:


from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters = i, init = 'k-means++', max_iter=300, n_init=10, random_state = 0, algorithm = 'full')
    km.fit(x)
    labels = km.labels_
    wcss.append(km.inertia_)
    
plt.rcParams['figure.figsize'] = (13, 7)
plt.plot(range(1, 11), wcss)
plt.grid()
plt.tight_layout()
plt.title('The Elbow Method', fontsize = 20)
plt.xlabel('No. of Clusters')
plt.ylabel('wcss')
plt.show()


# According to the graph above, the maximum curvature is at the second index, that is, the number of optimal clustering groups for the duration of the product and the bounce rates is 2.

# In[ ]:





# In[42]:


km = KMeans(n_clusters = 2, init = 'k-means++',max_iter=300,n_init=10, random_state = 0, algorithm = 'full')
    
y_means = km.fit_predict(x)

plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 50, c = 'yellow', label = 'Uninterested Customers')
plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 50, c = 'pink', label = 'Target Customers')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:, 1], s = 50, c = 'blue' , label = 'centeroid')

plt.title('ProductRelated Duration vs Bounce Rate', fontsize = 20)
plt.grid()
plt.xlabel('ProductRelated Duration')
plt.ylabel('Bounce Rates')
plt.legend()
plt.show()


# In[69]:


x=data['Revenue']


# ## PRINT CONFUSION MATRIX, CLASSIFICATION REPORT, ACCURACY SCORE

# In[82]:


from sklearn.metrics import confusion_matrix
print( confusion_matrix(x,y_means))


# In[83]:


from sklearn.metrics import classification_report
print(classification_report(x,y_means))


# In[84]:


from sklearn.metrics import accuracy_score
score=accuracy_score(x,y_means)
print(f'Accuracy: {round(score*100,2)}%')


# In[ ]:




