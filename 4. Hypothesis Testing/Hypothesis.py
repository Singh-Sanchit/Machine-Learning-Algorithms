
# coding: utf-8

# In[3]:


import pandas as pd
from scipy import stats
from statsmodels.stats import weightstats as stests


# In[4]:


path="blood_pressure.csv"
df = pd.read_csv(path)


# In[5]:


df[['bp_before','bp_after']].describe()
print(df.head(5))


# In[6]:


ttest,pval = stats.ttest_rel(df['bp_before'], df['bp_after'])
print(pval)


# In[7]:


if pval<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")


# In[8]:


ztest ,pval1 = stests.ztest(df['bp_before'], x2=df['bp_after'], value=0,alternative='two-sided')
print(float(pval1))


# In[9]:


df_anova = pd.read_csv('PlantGrowth.csv')
df_anova = df_anova[['weight','group']]

grps = pd.unique(df_anova.group.values)
d_data = {grp:df_anova['weight'][df_anova.group == grp] for grp in grps}
 
F, p = stats.f_oneway(d_data['ctrl'], d_data['trt1'], d_data['trt2'])

print("p-value for significance is: ", p)

if p<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")


# In[10]:


import statsmodels.api as sm
from statsmodels.formula.api import ols

df_anova2 = pd.read_csv("crop_yield.csv")


# In[11]:


model = ols('Yield ~ C(Fert)*C(Water)', df_anova2).fit()

# Seeing if the overall model is significant
print(f"Overall model F({model.df_model: .0f},{model.df_resid: .0f}) = {model.fvalue: .3f}, p = {model.f_pvalue: .4f}")


# In[12]:


print(model.summary())


# In[13]:


res = sm.stats.anova_lm(model, typ= 2)
print(res)


# In[14]:


df_chi = pd.read_csv('chi-test.csv')


# In[15]:


contingency_table=pd.crosstab(df_chi["Gender"],df_chi["Like Shopping?"])
print('contingency_table :-\n',contingency_table)


# In[16]:


#Observed Values
Observed_Values = contingency_table.values 
print("Observed Values :-\n",Observed_Values)


# In[17]:


b=stats.chi2_contingency(contingency_table)
Expected_Values = b[3]
print("Expected Values :-\n",Expected_Values)


# In[18]:


no_of_rows=len(contingency_table.iloc[0:2,0])
no_of_columns=len(contingency_table.iloc[0,0:2])
df11=(no_of_rows-1)*(no_of_columns-1)
print("Degree of Freedom:-",df)
alpha = 0.05


# In[19]:



from scipy.stats import chi2
chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
chi_square_statistic=chi_square[0]+chi_square[1]
print("chi-square statistic:-",chi_square_statistic)


# In[20]:


critical_value=chi2.ppf(q=1-alpha,df=df11)
print('critical_value:',critical_value)


# In[21]:


#p-value
p_value=1-chi2.cdf(x=chi_square_statistic,df=df11)
print('p-value:',p_value)


# In[22]:


print('Significance level: ',alpha)
print('Degree of Freedom: ',df11)
print('chi-square statistic:',chi_square_statistic)
print('critical_value:',critical_value)
print('p-value:',p_value)


# In[23]:


if chi_square_statistic>=critical_value:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")
    
if p_value<=alpha:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")

