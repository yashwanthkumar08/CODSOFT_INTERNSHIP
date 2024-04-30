#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


get_ipython().system('pip install plotly')


# In[3]:


import warnings


# In[4]:


warnings.filterwarnings('ignore')


# In[5]:


df=pd.read_csv('Titanic-Dataset.csv')


# In[6]:


df                                                       #we are printing the entire dataset basically


# In[7]:


df.info() #types of data 


# In[8]:


df.describe()  #stats of the dataset


# In[9]:


df=df.drop(['PassengerId','Name','Ticket','Cabin'],axis=1) #retain whates needed 


# In[10]:


df


# In[11]:


df.isna()


# In[12]:


df.isna().sum()   #to find the missing values


# In[13]:


df=df.dropna(subset=['Embarked'])   #all missing embarked values are dropped


# In[14]:


df


# In[15]:


import seaborn as sss  #to use graphs


# In[16]:


sss.boxplot(df['Age'])  #outliers


# In[17]:


sss.kdeplot(df['Age'])     #normal


# In[18]:


sss.boxplot(df['Fare'])  #outliers


# In[19]:


sss.kdeplot(df['Fare'])   #right sqewed


# In[20]:


sss.kdeplot(df['Pclass'])   #normal graph


# In[21]:


sss.boxplot(df['Pclass'])  #no outliers


# In[22]:


dfage=df['Age']
dffare=df['Fare']
#we need to fix age and fare 


# In[23]:


import numpy as np
dfage_abs=dfage.abs()  #avoid non negative numbers
dffare_abs=dffare.abs()

changedage=dfage_abs.apply(lambda x: x**(1/2)) #reducing the values using square root
changedfare=dffare_abs.apply(lambda x: np.log1p(x))

df['Age']=changedage
df['Fare']=changedfare


# In[24]:


df['Age']


# In[25]:


df['Fare']


# In[26]:


sss.boxplot(df['Fare']) 


# In[27]:


sss.kdeplot(df['Fare'])


# In[28]:


sss.boxplot(df['Age']) 


# In[29]:


sss.kdeplot(df['Age']) 


# In[30]:


df


# In[31]:


from sklearn.preprocessing import OneHotEncoder   #we are converting string values to numeric using ENCODING technique


# In[32]:


attr_val=['Sex','Embarked'] 
data_to_encode=df[attr_val]

data_to_enocde=data_to_encode.dropna() #remove missing values as encoder will not work

encoder=OneHotEncoder(drop='first') #we remove sex_female,embarked_c because the if other attributes show 0,then ofcourse sex_female =1 ,embarked_c=1

encoded_data=encoder.fit_transform(data_to_encode) #transform data into encoded data

encoded_df=pd.DataFrame(encoded_data.toarray(),columns=encoder.get_feature_names_out(attr_val)) #converting encoded data into dataframe

df.reset_index(drop=True,inplace=True)#reset indices 
encoded_df.reset_index(drop=True,inplace=True)

df_encoded=pd.concat([df,encoded_df],axis=1)#joining old dataframe with new dataframe 

df=df_encoded.drop(columns=attr_val)  #remove original columns that got encoded


# In[33]:


df


# In[34]:


sss.boxplot(df['Age'])


# In[35]:


sss.kdeplot(df['Age'])


# In[36]:


df.isna().sum()  #we still have outliers in age


# In[37]:


#we could not eliminate missing values,so we use linear regression to predict missing values in age attribute
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer


# In[38]:


#seperating rows with values and without values
without_age=df[df['Age'].isnull()]
with_age=df.dropna(subset=['Age'])

#training using with_age 
x_train=with_age.drop(columns=['Age'])
y_train=with_age['Age']
regressor_obj=LinearRegression()
regressor_obj.fit(x_train,y_train)

#predicting missing data
missing_x_values=without_age.drop(columns=['Age'])
predicted_age=regressor_obj.predict(missing_x_values)

#replacing missing values with predicted values in the dataset
df.loc[df['Age'].isnull(),'Age']=predicted_age


# In[39]:


df


# In[40]:


df.isna().sum()   #no outliers left in age


# In[41]:


#remove any duplicates if present
df=df.drop_duplicates()


# In[42]:


df  #around 100 rows were removed


# In[43]:


#we have finished the preprocessing of data ,now we need to do visualization


# In[44]:


import matplotlib.pyplot as plt
import seaborn as sss


# In[45]:


sss.countplot(df,x='Pclass',hue='Survived')


# In[46]:


#we can see 3rd class people have not survived ,so 3rd class is in lower berth of ship
#1st and second class have survived

df


# In[49]:


df['Family']=df['SibSp']+df['Parch']
df['Family']   #we created dataset of people containing family


# In[50]:


df['Alone']=df['Family']==0  #create dataset of people who dont have family


# In[51]:


df=df.drop(['SibSp','Parch'],axis=1)


# In[52]:


sss.boxplot(df)


# In[53]:


#outliers in family ,age,embarked,fare
#standardize the age first

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scale_age=['Age']

df[scale_age]=scaler.fit_transform(df[scale_age])


# In[54]:


sss.boxplot(df)


# In[55]:


rel_mat=df.corr()
rel_with_surv=rel_mat['Survived'].drop('Survived')


# In[56]:


rel_with_surv


# In[57]:


#lets assign a threshold and find values nearest to target variable
threshold=0.2
close_rel_with_surv=rel_with_surv[abs(rel_with_surv)>threshold]


# In[59]:


close_rel_with_surv


# In[60]:


ID=df[['Pclass','Age','Fare','Sex_male']]
D=df['Survived']


# In[61]:


ID


# In[62]:


D


# In[63]:


from sklearn.model_selection import train_test_split
ID_train,ID_test,D_train,D_test=train_test_split(ID,D,test_size=0.2,random_state=42) 


# In[64]:


ID_train


# In[65]:


ID_test


# In[66]:


D_train


# In[67]:


D_test


# In[80]:


#using logistic regression model for training
#first technique
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report 

#training the model
model1=LogisticRegression()
model1.fit(ID_train,D_train)
D_pred=model1.predict(ID_test)

#lets determine the model accuracy 
accuracy=accuracy_score(D_test,D_pred)
print(accuracy)


# In[82]:


#using decsion tree second method
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score

model2=DecisionTreeClassifier()
model2.fit(ID_train,D_train)
D_pred=model2.predict(ID_test)

accuracy=accuracy_score(D_test,D_pred)
print(accuracy)


# In[85]:


get_ipython().system('pip install xgboost')
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score

model3=XGBClassifier()
model3.fit(ID_train,D_train)
D_pred=model3.predict(ID_test)

accuracy=accuracy_score(D_test,D_pred)
print(accuracy)


# In[93]:


#using radnom forest method
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score

model4=RandomForestClassifier()
model4.fit(ID_train,D_train)

D_pred=model4.predict(ID_test)

accuracy=accuracy_score(D_test,D_pred)
print(accuracy)

print(classification_report(D_test,D_pred))
      


# In[94]:


#at the moment we have random forest giving  best accuracy


# In[ ]:




