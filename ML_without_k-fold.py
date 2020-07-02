#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from scipy.linalg import svd
import pandas as pd
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
# In[4]:


ilpd = pd.read_csv('ill.csv')


# In[5]:


ilpd = ilpd.replace(np.NaN,(ilpd.mean()) )


# In[6]:


X = ilpd[['Age','Gender','TB','DB','Alkphos','Sgpt','Sgot','TP','ALB','A']]
labelencoder_X = LabelEncoder()
X['Gender'] = labelencoder_X.fit_transform(X['Gender'])
Y = ilpd['Selector']
print(X.head())
pca = decomposition.PCA(n_components=5)
pca.fit(X)
X = pca.transform(X)



# In[7]:


from sklearn.preprocessing import MinMaxScaler
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.14,random_state = 42) 
scaler = MinMaxScaler()
X_train= scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
y_train=y_train.to_numpy()
y_test=y_test.to_numpy()


# In[ ]:



def knn(row,data,re,k):#euclidean
    d=[]
    #for u in data:
    for i in range(len(data)):
        s=0
        for j in range(1,len(data[i])):
            s+=(data[i][j]-row[j])**2
        
        d.append([s**(1/2),re[i]])
    
    d.sort(key=lambda x:x[0])
    top=d[:k]
    result={}
    for i in top:
        if(i[1] not in result):
            result[i[1]]=0
        result[i[1]]+=1*(1/i[0])
    m=-1
    final_classifier=0
    for i,j in result.items():
        if(m<j):
            m=j
            final_classifier=i
    return final_classifier


# In[ ]:


acc=[]

for j in range(3,500):
    count=0
    for i in range(82):
        temp=knn(X_test[i],X_train,y_train,j)
        if(temp==y_test[i]):
            count+=1
    acc.append(count/82)
print(max(acc))
print(acc.index(max(acc))+3)



def knn_m(row,data,re,k):#manhatten
    d=[]
    #for u in data:
    for i in range(len(data)):
        s=0
        for j in range(1,len(data[i])):
            s+=abs(data[i][j]-row[j])
        
        d.append([s,re[i]]) 
    d.sort(key=lambda x:x[0])
    top=d[:k]
    result={}
    for i in top:
        if(i[1] not in result):
            result[i[1]]=0
        result[i[1]]+=1*(1/i[0])
    m=-1
    final_classifier=0
    for i,j in result.items():
        if(m<j):
            m=j
            final_classifier=i
    return final_classifier

import matplotlib.pyplot as plt  
index = []
for i in range(3,500):
    index.append(i)
plt.plot(index, acc) 
  
# naming the x axis 
plt.xlabel('K-values') 
# naming the y axis 
plt.ylabel('Accuracy') 
  

# function to show the plot 
plt.show() 


# In[9]:


acc=[]

for j in range(3,500):
    count=0
    for i in range(82):
        temp=knn_m(X_test[i],X_train,y_train,j)
        if(temp==y_test[i]):
            count+=1
    acc.append(count/82)
print(max(acc))
print(acc.index(max(acc))+3)


# In[10]:



# In[12]:


import matplotlib.pyplot as plt  
index = []
for i in range(3,500):
    index.append(i)
plt.plot(index, acc) 
  
# naming the x axis 
plt.xlabel('K-values') 
# naming the y axis 
plt.ylabel('Accuracy') 
  

# function to show the plot 
plt.show() 


# In[ ]:




