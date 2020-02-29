# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Exploratory Data Analysis 

# In[2]:


DATAPATH = 'Advertising.csv'

data = pd.read_csv(DATAPATH)
print(data.head())


# In[3]:


data.drop(['Unnamed: 0'], axis=1, inplace=True)


# In[4]:


print(data.head())


# In[5]:


print(data.columns)


# In[6]:


def scatter_plot(feature, target):
    plt.figure(figsize=(16, 8))
    plt.scatter(
        data[feature],
        data[target],
        c='black'
    )
    plt.xlabel("Money spent on {} ads ($)".format(feature))
    plt.ylabel("Sales ($k)")
    plt.show()


# In[7]:


scatter_plot('TV', 'sales')


# In[8]:


scatter_plot('radio', 'sales')


# In[9]:


scatter_plot('newspaper', 'sales')


# ## Modelling 

# ### Multiple linear regression - least squares fitting 

# In[10]:


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

Xs = data.drop(['sales'], axis=1)
y = data['sales'].values.reshape(-1,1)

lin_reg = LinearRegression()

MSEs = cross_val_score(lin_reg, Xs, y, scoring='neg_mean_squared_error', cv=5)

mean_MSE = np.mean(MSEs)

print(mean_MSE)


# ### Ridge regression 

# In[11]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

alpha = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

ridge_regressor = GridSearchCV(ridge, parameters,scoring='neg_mean_squared_error', cv=5)

ridge_regressor.fit(Xs, y)


# In[12]:


print("Ridge Best Param", ridge_regressor.best_params_)


# In[13]:


print("Ridge Best Score", ridge_regressor.best_score_)


# ### Lasso 

# In[14]:


from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv = 5)

lasso_regressor.fit(Xs, y)


# In[15]:


print("Lasso Best Param", lasso_regressor.best_params_)


# In[16]:


print("Lasso Best Score", lasso_regressor.best_score_)

