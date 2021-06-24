#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#reading the dataset
df_old=pd.read_csv('coviddata.csv')


# In[5]:


df_old.head(5)


# In[8]:


#subsetting the data having location as india (cases in india)
subset_india=df_old[df_old['location']=='India']
subset_india


# In[6]:


df_old.columns


# In[7]:


df_old.shape


# In[9]:


df_old.drop(['continent','iso_code','total_deaths','total_cases_per_million','date',
       'new_cases_per_million', 'total_deaths_per_million','total_tests',
       'new_deaths_per_million','new_tests_smoothed', 'new_tests_smoothed_per_thousand','aged_70_older',"cvd_death_rate","cvd_death_rate",'hospital_beds_per_thousand'], axis=1,inplace=True)


# In[10]:


df_old.columns


# In[11]:


df_old["new_tests"]=df_old.new_tests.replace().astype(float)
df_old["total_tests_per_thousand"]=df_old.total_tests_per_thousand.replace().astype(float)


# In[13]:


df_old["new_tests"].unique()


# In[14]:


df_old["new_tests"].fillna(df_old["new_tests"].mean(),inplace=True,axis=0) 
df_old["total_tests_per_thousand"].fillna(df_old["total_tests_per_thousand"].mean(),inplace=True,axis=0)


# In[15]:


df_old.shape


# In[16]:


#checking the null values

null_values=df_old.isnull().sum().tolist()
print(null_values)
pd=[]
for a in null_values:
    if a >= (29591/2):
        null_values.remove(a)
    else:
        pd.append(a)
print(pd)


# In[17]:


df_old.isnull().sum()


# In[18]:


#dropping the columns which have more than 50 prcnt of rows as null

df=df_old.dropna(thresh=df_old.shape[0]*0.5,how='all',axis=1)


# In[19]:


print(df_old.shape)
df.shape


# In[20]:


df.columns


# In[23]:


df.head()


# In[24]:


#import datetime as dt


# In[25]:


#converting datetime format to the ordinal form
#import pandas as pd
#df['date'] =  pd.to_datetime(df['date'], infer_datetime_format=True)


# In[26]:


#df["date"]=pd.to_datetime(df["date"])
#df["date"]=df["date"].map(dt.datetime.toordinal)


# In[21]:


df.dtypes


# In[22]:


#dropping the columns having categorical values
df.drop(['location','gdp_per_capita'],axis=1,inplace=True)


# In[23]:


#for the remaining columns filling the null values with mean
for column in df:
    df[column].fillna(df[column].mean(), inplace=True)
df.isnull().sum()


# In[24]:


sns.boxplot(df['stringency_index'])
Q3 = df.stringency_index.quantile(.75)
Q1 = df.stringency_index.quantile(.25)
IQR = Q3 - Q1
Median = df.stringency_index.median()
print("Q1 Value:",Q1)
print("Median Value:",df.stringency_index.median())
print("Q3 Value:",Q3)
print("Upper whisker limit:",(Q3 + 1.5*IQR))
print("Lowerr whisker limit:",(Q1 - 1.5*IQR))
plt.savefig('1 bf', dpi=300, bbox_inches='tight')


# In[25]:


df.shape


# In[26]:


a= df[(df['stringency_index'] < 3.230000000000011)].index
df.drop(a,axis=0, inplace=True)


# In[27]:


sns.boxplot(df['stringency_index'])
plt.savefig('1af', dpi=300, bbox_inches='tight')


# In[28]:


sns.boxplot(df['median_age'])
Q3 = df.median_age.quantile(.75)
Q1 = df.median_age.quantile(.25)
IQR = Q3 - Q1
Median = df.median_age.median()
print("Q1 Value:",Q1)
print("Median Value:",df.median_age.median())
print("Q3 Value:",Q3)
print("Upper whisker limit:",(Q3 + 1.5*IQR))
print("Lowerr whisker limit:",(Q1 - 1.5*IQR))
plt.savefig('2', dpi=300, bbox_inches='tight')


# In[29]:


sns.boxplot(df['aged_65_older'])
Q3 = df.aged_65_older.quantile(.75)
Q1 = df.aged_65_older.quantile(.25)
IQR = Q3 - Q1
Median = df.aged_65_older.median()
print("Q1 Value:",Q1)
print("Median Value:",df.aged_65_older.median())
print("Q3 Value:",Q3)
print("Upper whisker limit:",(Q3 + 1.5*IQR))
print("Lowerr whisker limit:",(Q1 - 1.5*IQR))
plt.savefig('3', dpi=300, bbox_inches='tight')


# In[30]:


sns.boxplot(df['population'])
Q3 = df.population.quantile(.75)
Q1 = df.population.quantile(.25)
IQR = Q3 - Q1
Median = df.population.median()
print("Q1 Value:",Q1)
print("Median Value:",df.population.median())
print("Q3 Value:",Q3)
print("Upper whisker limit:",(Q3 + 1.5*IQR))
print("Lowerr whisker limit:",(Q1 - 1.5*IQR))
plt.savefig('3', dpi=300, bbox_inches='tight')


# In[31]:


df.shape


# In[32]:


c= df[(df['population'] > 75583126.0)].index
df.drop(c,axis=0, inplace=True)


# In[33]:


sns.boxplot(df['population'])
plt.savefig('4af', dpi=300, bbox_inches='tight')


# In[34]:


df.shape


# In[35]:


df.columns


# In[36]:


sns.boxplot(df['male_smokers'])
Q3 = df.male_smokers.quantile(.75)
Q1 = df.male_smokers.quantile(.25)
IQR = Q3 - Q1
Median = df.male_smokers.median()
print("Q1 Value:",Q1)
print("Median Value:",df.male_smokers.median())
print("Q3 Value:",Q3)
print("Upper whisker limit:",(Q3 + 1.5*IQR))
print("Lowerr whisker limit:",(Q1 - 1.5*IQR))
plt.savefig('5bf', dpi=300, bbox_inches='tight')


# In[37]:


b= df[(df['male_smokers'] > 60 )].index
e= df[(df['male_smokers'] < 10)].index

df.drop(b,axis=0, inplace=True)
df.drop(e,axis=0, inplace=True)
b.shape
e.shape


# In[38]:


sns.boxplot(df['male_smokers'])
plt.savefig('5af', dpi=300, bbox_inches='tight')


# In[39]:


sns.boxplot(df['female_smokers'])
Q3 = df.female_smokers.quantile(.75)
Q1 = df.female_smokers.quantile(.25)
IQR = Q3 - Q1
Median = df.female_smokers.median()
print("Q1 Value:",Q1)
print("Median Value:",df.female_smokers.median())
print("Q3 Value:",Q3)
print("Upper whisker limit:",(Q3 + 1.5*IQR))
print("Lowerr whisker limit:",(Q1 - 1.5*IQR))
plt.savefig('6bf', dpi=300, bbox_inches='tight')


# In[40]:


f= df[(df['female_smokers'] > 40 )].index
df.drop(f,axis=0, inplace=True)
f.shape


# In[41]:


sns.boxplot(df['female_smokers'])
plt.savefig('6af', dpi=300, bbox_inches='tight')


# In[42]:


g= df[(df['female_smokers'] > 30 )].index
df.drop(g,axis=0, inplace=True)
g.shape


# In[43]:


sns.boxplot(df['diabetes_prevalence'])
Q3 = df.diabetes_prevalence.quantile(.75)
Q1 = df.diabetes_prevalence.quantile(.25)
IQR = Q3 - Q1
Median = df.diabetes_prevalence.median()
print("Q1 Value:",Q1)
print("Median Value:",df.diabetes_prevalence.median())
print("Q3 Value:",Q3)
print("Upper whisker limit:",(Q3 + 1.5*IQR))
print("Lowerr whisker limit:",(Q1 - 1.5*IQR))
plt.savefig('7bf', dpi=300, bbox_inches='tight')


# In[44]:


h= df[(df['diabetes_prevalence'] > 17 )].index
df.drop(h,axis=0, inplace=True)
h.shape


# In[45]:


sns.boxplot(df['diabetes_prevalence'])
plt.savefig('7af', dpi=300, bbox_inches='tight')


# In[46]:


sns.boxplot(df['population_density'])
Q3 = df.population_density.quantile(.75)
Q1 = df.population_density.quantile(.25)
IQR = Q3 - Q1
Median = df.population_density.median()
print("Q1 Value:",Q1)
print("Median Value:",df.population_density.median())
print("Q3 Value:",Q3)
print("Upper whisker limit:",(Q3 + 1.5*IQR))
print("Lowerr whisker limit:",(Q1 - 1.5*IQR))
plt.savefig('8bf', dpi=300, bbox_inches='tight')


# In[47]:


k= df[(df['population_density'] > 450 )].index
df.drop(k,axis=0, inplace=True)
k.shape


# In[48]:


sns.boxplot(df['population_density'])
plt.savefig('8af', dpi=300, bbox_inches='tight')


# In[59]:


df.shape


# findng out mean mode and median for all the columns and plotting the box plots

# In[63]:


sns.jointplot(x="date", y="total_cases", data=df,size=10,color="g")
# used subset of data to see a more clear plot
plt.savefig('1.png', dpi=300, bbox_inches='tight')


# In[64]:


sns.jointplot(y="new_tests", x="total_cases", data=df,color="y")
# used subset of data to see a more clear plot
plt.savefig('14.png', dpi=300, bbox_inches='tight')


# In[65]:


sns.jointplot(x="aged_65_older", y="new_deaths", data=df,color="b")
plt.savefig('13.png', dpi=300, bbox_inches='tight')


# In[ ]:


######################MODEL


# In[51]:


#linear regression model

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
y=df['total_cases']
x=df.drop(['total_cases'],axis=1)


# In[52]:


def x_and_y(df):                                                            #dropping the target column from the train set
    not_target = df.drop(["total_cases"], axis = 1)
    target = df["total_cases"]
    return not_target, target


# In[53]:


train,test=train_test_split(df,test_size=0.3)
reg_all=LinearRegression()


# In[54]:


x_train, y_train = x_and_y(train)
x_test, y_test = x_and_y(test)


# In[55]:


reg_all.fit(x_train,y_train)


# In[56]:


y_pred=reg_all.predict(x_test)


# In[57]:


print("R^2: {}".format(reg_all.score(x_test, y_test)))
rmse = np.sqrt(y_test,y_pred)
print("Root Mean Squared Error: {}".format(rmse))

#accuracy
reg_all.score(x_test,y_test)
#reg_all.score(x_train,y_train)


# In[58]:


#random forest regressor

from sklearn.ensemble import RandomForestRegressor


rf = RandomForestRegressor(n_estimators=25)
           
rf.fit(x_train, y_train) 


# In[59]:


from sklearn.metrics import mean_squared_error as MSE


# In[60]:


y_pred = rf.predict(x_test).reshape(-1,1)
x_pred=rf.predict(x_train)


# In[61]:


rmse_test = MSE(y_test,y_pred)**(1/2)
print('Test set RMSE of rf: {:.2f}'.format(rmse_test))

#rmse_test = MSE(x_train,x_pred)**(1/2)
#print('Test set RMSE of rf: {:.2f}'.format(rmse_test))

#accuracy
print(rf.score(x_train,y_train))
print(rf.score(x_test,y_test))


# In[ ]:





# In[ ]:





# In[ ]:




