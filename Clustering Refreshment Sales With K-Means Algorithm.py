#!/usr/bin/env python
# coding: utf-8

# # Clustering  Refreshment Sales With K-Means Algorithm

# In[1]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import operator
import random as rd
from matplotlib import cm
from itertools import cycle, islice
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('D:/Materi Kuliah/Semester 7/PD/Proyek/data/Refreshment_Sales.csv')
data.head()


# In[3]:


data.describe()


# In[4]:


data.info()


# In[5]:


df = pd.DataFrame(data)


# In[6]:


df['capacity']
df.capacity = (df.capacity.replace(r'[mllt]+$', '', regex=True).astype(float) *                 df.capacity.str.extract(r'[\d\.]+([mllt]+)', expand=False)
                .replace(['ml','lt'], [10**1, 10**3]).astype(int))
print(df.capacity)


# In[7]:


data.count()


# In[8]:


data.dtypes


# In[9]:


data['date'].duplicated().sum()


# In[10]:


missing_value = pd.DataFrame(data.isnull().sum())


# In[11]:


missing_value = missing_value.rename(columns ={'index':'Variables', 0:'Missing percentage'})
missing_value['Missing percentage'] = (missing_value['Missing percentage']/len(data))*100


# In[12]:


missing_value


# In[13]:


data.sort_values(['sales'], axis=0,ascending=True, inplace=True) 
data


# In[14]:


features = ['quantity', 'capacity', 'sales', 'month']


# In[15]:


select_df = data[features]


# In[16]:


select_df.columns


# In[17]:


select_df


# In[18]:


def pd_centers(featuresUsed, centers):
    colNames = list(featuresUsed)
    colNames.append('prediction')
    
    Z = [np.append(A, index) for index, A in enumerate(centers)]
    P = pd.DataFrame(Z, columns=colNames)
    P['prediction'] = P['prediction'].astype(int)
    return P


# In[19]:


def parallel_plot(data):
    my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(data)))
    plt.figure(figsize=(15,8)).gca().axes.set_ylim([-3,+3])
    parallel_coordinates(data, 'prediction', color = my_colors, marker='o')


# In[20]:


X = data[["sales", "month"]]

#Visualise data points
plt.figure(figsize=(12,6))
plt.scatter(X["month"], X["sales"])
plt.xlabel('month')
plt.ylabel('sales')
plt.show()


# In[21]:


plt.figure(figsize=(10,7), dpi =100)
sort = data.sort_values(['month'], axis=0,ascending=True, inplace=True) 
plot = sns.countplot(data['month'], data=sort)
plot.set_xticklabels(plot.get_xticklabels(), rotation = 45)
plt.tight_layout()

print("Refreshment Data of Sales")


# In[22]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import KMeans


# In[23]:


data.head()


# In[24]:


data.shape


# In[25]:


data.isnull().sum()


# In[26]:


summary = data.describe()
summary = summary.transpose()
summary.head()


# In[27]:


x=data.iloc[:,[7,5]].values 


plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
plt.scatter(x[:,0],x[:,1], c='black', s=7)


# In[28]:


from scipy import stats
x=x[(np.abs(stats.zscore(x))<3).all(axis=1)]


# In[29]:


x.shape


# In[30]:


from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

select_df


# In[37]:


kmeans = KMeans(n_clusters=12).fit(select_df)
centroids = kmeans.cluster_centers_
print(centroids)

plt.scatter(select_df['quantity'], select_df['sales'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.xlabel('Quantity')
plt.ylabel('Sales')


# In[32]:


for brand in list(data['brand'].unique()):
    print(brand, data[data['brand'] == brand]['sales'].sum())


# In[33]:


for i in range(0, data['brand'].count()):
    if data.loc[i,'sales'] == '$':
        data.loc[i,'sales'] = 1
    elif data.loc[i,'sales'] == "$$ - $$$":
        data.loc[i,'sales'] = 2
    elif data.loc[i,'sales'] == '$$$$':
        data.loc[i,'sales'] = 3
data['sales'][np.isnan(data['sales'])] = 0


# In[34]:


sns.countplot(x='month', data = data, hue = 'brand')


# In[35]:


fig, axes = plt.subplots(1,1, figsize=(7,5))
columns = ['sales']
i=0
y_ax = list()
x_ax = list(data['brand'].unique())

for col in columns:
    for brand in list(data['brand'].unique()):
        na = data[data['brand'] == brand]['sales'].sum()
        y_ax.append(na)  
    axes.bar(x_ax, y_ax)
    axes.set_xticklabels(x_ax, rotation = 90)
    axes.set_ylabel('{} Sales Count Mean'.format(col))
    axes.set_xlabel('brand')
    plt.tight_layout()

