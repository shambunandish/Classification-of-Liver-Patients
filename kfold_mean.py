import numpy as np
from scipy.linalg import svd
import pandas as pd
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
# In[3]:
il = pd.read_csv('ill.csv')
# In[4]:
il = il.replace(np.NaN,(il.mean()) )

# In[5]:


from sklearn.preprocessing import MinMaxScaler
X = il[['Age','Gender','TB','DB','Alkphos','Sgpt','Sgot','TP','ALB','A']]
labelencoder_X = LabelEncoder()
X['Gender'] = labelencoder_X.fit_transform(X['Gender'])
#print(X)
Y = il['Selector']
pca = decomposition.PCA(n_components=5)
pca.fit(X)
X = pca.transform(X)
scaler = MinMaxScaler()
X= scaler.fit_transform(X)


# In[6]:


#il=il.to_numpy()
print(X)


# In[7]:


# import random

# a = random.randrange(1,583)
# b = random.randrange(1,583)
# print(a,b)#544 42


# In[8]:


Xk =[]
Yk=[]
for i in (X):
    Xk.append(i)
for i in (Y):
    Yk.append(i)
Xk.append(X[544])
Xk.append(X[42])
Yk.append(Y[544])
Yk.append(Y[42])
#print(len(Xk))
#print(len(Yk))


# In[9]:


k1x = []
k1y=[]
k2x = []
k2y=[]
k3x = []
k3y=[]
k4x = []
k4y=[]
k5x = []
k5y=[]


# In[10]:


for i in range(0,117):
    k1x.append(Xk[i])
    k1y.append(Yk[i])


# In[11]:


for i in range(117,234):
    k2x.append(Xk[i])
    k2y.append(Yk[i])


# In[ ]:





# In[12]:


for i in range(234,351):
    k3x.append(Xk[i])
    k3y.append(Yk[i])


# In[13]:


for i in range(351,468):
    k4x.append(Xk[i])
    k4y.append(Yk[i])


# In[14]:


for i in range(468,585):
    k5x.append(Xk[i])
    k5y.append(Yk[i])


# In[15]:


k1x = np.array(k1x)
k2x = np.array(k2x)
k3x = np.array(k3x)
k4x = np.array(k4x)
k5x = np.array(k5x)
k1y = np.array(k1y)
k2y = np.array(k2y)
k3y = np.array(k3y)
k4y = np.array(k4y)
k5y = np.array(k5y)


# In[16]:


print(len(k1x),len(k1y))
tr1x = []
tr1y = []


# In[17]:


for i in range(0,117):
    tr1x.append(k1x[i])
    tr1x.append(k2x[i])
    tr1x.append(k3x[i])
    tr1x.append(k4x[i])


# In[18]:


print(len(tr1x))
for i in range(0,117):
    tr1y.append(k1y[i])
    tr1y.append(k2y[i])
    tr1y.append(k3y[i])
    tr1y.append(k4y[i])


# In[19]:


print(len(tr1y))


# In[20]:


def knn(row,data,re,k):#manhatten
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


# In[21]:


acc=[]#first 4

for j in range(3,468):
    count=0
    for i in range(117):
        temp=knn(k5x[i],tr1x,tr1y,j)
        if(temp==k5y[i]):
            count+=1
    acc.append(count/117)
print(max(acc))
print(acc.index(max(acc))+3)


# In[22]:


tr2x = []
tr2y = []
for i in range(0,117):
    tr2x.append(k1x[i])
    tr2x.append(k2x[i])
    tr2x.append(k3x[i])
    tr2x.append(k5x[i])
for i in range(0,117):
    tr2y.append(k1y[i])
    tr2y.append(k2y[i])
    tr2y.append(k3y[i])
    tr2y.append(k5y[i])


# In[23]:


acc1=[]
#1 2 3 5
for j in range(3,468):
    count=0
    for i in range(117):
        temp=knn(k4x[i],tr2x,tr2y,j)
        if(temp==k4y[i]):
            count+=1
    acc1.append(count/117)
print(max(acc1))
print(acc1.index(max(acc1))+3)


# In[24]:


tr3x = []
tr3y = []
for i in range(0,117):
    tr3x.append(k1x[i])
    tr3x.append(k2x[i])
    tr3x.append(k4x[i])
    tr3x.append(k5x[i])
for i in range(0,117):
    tr3y.append(k1y[i])
    tr3y.append(k2y[i])
    tr3y.append(k4y[i])
    tr3y.append(k5y[i])
print(len(tr3y))


# In[25]:


acc2=[]
#1 2 4 5
for j in range(3,468):
    count=0
    for i in range(117):
        temp=knn(k3x[i],tr3x,tr3y,j)
        if(temp==k3y[i]):
            count+=1
    acc2.append(count/117)
print(max(acc2))
print(acc2.index(max(acc2))+3)


# In[26]:


tr4x = []
tr4y = []
for i in range(0,117):
    tr4x.append(k1x[i])
    tr4x.append(k3x[i])
    tr4x.append(k4x[i])
    tr4x.append(k5x[i])
for i in range(0,117):
    tr4y.append(k1y[i])
    tr4y.append(k3y[i])
    tr4y.append(k4y[i])
    tr4y.append(k5y[i])
print(len(tr3y))


# In[27]:


acc3=[]
#1 3 4 5
for j in range(3,468):
    count=0
    for i in range(117):
        temp=knn(k2x[i],tr4x,tr4y,j)
        if(temp==k2y[i]):
            count+=1
    acc3.append(count/117)
print(max(acc3))
print(acc3.index(max(acc3))+3)


# In[28]:


tr5x = []
tr5y = []
for i in range(0,117):
    tr5x.append(k2x[i])
    tr5x.append(k3x[i])
    tr5x.append(k4x[i])
    tr5x.append(k5x[i])
for i in range(0,117):
    tr5y.append(k2y[i])
    tr5y.append(k3y[i])
    tr5y.append(k4y[i])
    tr5y.append(k5y[i])
print(len(tr3y))


# In[29]:


acc4=[]
#2 3 4 5
for j in range(3,468):
    count=0
    for i in range(117):
        temp=knn(k1x[i],tr5x,tr5y,j)
        if(temp==k1y[i]):
            count+=1
    acc4.append(count/117)
print(max(acc4))
print(acc4.index(max(acc4))+3)


# In[ ]:





# In[ ]:




