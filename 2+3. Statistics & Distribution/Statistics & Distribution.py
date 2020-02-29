
# coding: utf-8

# Understanding Descriptive Statistics with python

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


# In[2]:


path="house_prices.csv"
df = pd.read_csv(path)


# In[3]:


print(df.head())


# In[4]:


print(df.shape)


# In[5]:


df.info()


# In[7]:


saleprice = df['SalePrice']

mean=saleprice.mean()
median=saleprice.median()
mode=saleprice.mode()

print('Mean: ',mean,'\nMedian: ',median,'\nMode: ',mode[0])
plt.figure(figsize=(10,5))
plt.hist(saleprice,bins=100,color='grey')
plt.axvline(mean,color='red',label='Mean')
plt.axvline(median,color='yellow',label='Median')
plt.axvline(mode[0],color='green',label='Mode')
plt.xlabel('SalePrice')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[8]:


print("Start Value", saleprice.cumsum().head())


# Measures to find the spread of data

# In[9]:


print("Minimum Value", saleprice.min()) #maximum value of salePrice


# In[10]:


print("Maximum Value", saleprice.max()) #minimum value of salePrice


# In[11]:


#Range
print("Range", saleprice.max()-saleprice.min())


# In[12]:


#variance
print("Variance", saleprice.var())


# In[13]:


from math import sqrt

#standard deviation
std = sqrt(saleprice.var())
print("Standard Deviation", std)


# In[14]:


#skewness
print("Skewness", saleprice.skew())


# In[15]:


#kutosis
print("Kutosis", saleprice.kurt())


# In[16]:


#convert pandas DataFrame object to numpy array and sort
h = np.asarray(df['SalePrice'])
h = sorted(h)
 
#use the scipy stats module to fit a normal distirbution with same mean and standard deviation
fit = stats.norm.pdf(h, np.mean(h), np.std(h)) 
 
#plot both series on the histogram
plt.plot(h,fit,'-',linewidth = 2,label="Normal distribution with same mean and var")
plt.hist(h,normed=True,bins = 100,label="Actual distribution")      
plt.legend()
plt.show() 


# We can see int the above graph that it is positively skewed with skewness score 1.93 and also has positive kurtosis(k=6.735)

# In[18]:


#checking correlation of 4 countinous variables
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
corelation=df[['LotArea','GrLivArea','GarageArea','SalePrice']].corr()
print(corelation)

sns.heatmap(corelation)


# In[19]:


#covariance
print("Covariance", df[['LotArea','GrLivArea','GarageArea','SalePrice']].cov().head())


# In[20]:


# #50 percentile i.e median
# np.percentile(df['salary'], 50)

print("Median q2", saleprice.quantile(0.5))


# In[21]:


# q75 = np.percentile(df['salary'], 75)
# q75

q3 = saleprice.quantile(0.75)
print("q3", q3)


# In[22]:


#25th percentile
# q25 = np.percentile(df['salary'], 25)
q1 = saleprice.quantile(0.25)
print("q1", q1)


# In[23]:


#interquartile range
IQR = q3  - q1
print("IQR", IQR)


# In[24]:


plt.boxplot(saleprice)
plt.show()

