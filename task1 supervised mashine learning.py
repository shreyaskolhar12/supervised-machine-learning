#!/usr/bin/env python
# coding: utf-8

# # THE SPARKS FOUNDATION
# 
# ## TASK1-TO EXPLORE SUPERVISED MASHINE LEARNING(LINEAR REGRESSION)
# 
# ## NAME- SHREYAS SACHIN KOLHAR

# In[2]:


##Importing important libraries---
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


Data = pd.read_csv('C:\\Users\\dell\\Desktop\\student_scores.csv')
print("Data is successfully imported")
print (Data)


# In[3]:


print("Head")
print(Data.head())


# In[4]:


print("tail")
print(Data.tail())


# In[5]:


print("describe :")
print(Data.describe())


# In[6]:


plt.boxplot(Data)
plt.show()


# ### Visualizing Data.

# In[7]:


##ploting Scatter plot----
plt.xlabel('Hours',fontsize=15)
plt.ylabel('Scores',fontsize=15)
plt.title('Hours studied vs Score', fontsize=10)
plt.scatter(Data.Hours,Data.Scores,color='blue',marker='*')
plt.show()


# ### This plot shows that as much of hours you study high score you will secure.

# In[8]:


X = Data.iloc[:,:-1].values
Y = Data.iloc[:,1].values
X


# In[9]:


Y


# ### Preparing data and splitting into train and test sets.

# In[10]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state = 0,test_size=0.2)


# In[11]:


## We have Splitted Our Data Using 80:20 RULe(PARETO)
print("X train.shape =", X_train.shape)
print("Y train.shape =", Y_train.shape)
print("X test.shape =", X_test.shape)
print("Y test.shape =", Y_test.shape)


# ### Training the model.

# In[12]:


from sklearn.linear_model import LinearRegression
linreg=LinearRegression()


# In[13]:


##Fitting Training Data
linreg.fit(X_train,Y_train)
print("Training our algorithm is finished")


# In[14]:


print("B0 =",linreg.intercept_,"\nB1 =",linreg.coef_)## β0 is Intercept & Slope of the line is β1.,"

##plotting the REGRESSION LINE---
Y0 = linreg.intercept_ + linreg.coef_*X_train


# In[15]:


##plotting on train data
plt.scatter(X_train,Y_train,color='green',marker='+')
plt.plot(X_train,Y0,color='orange')
plt.xlabel("Hours",fontsize=15)
plt.ylabel("Scores",fontsize=15)
plt.title("Regression line(Train set)",fontsize=10)
plt.show()


# ### Test data.

# In[16]:


Y_pred=linreg.predict(X_test)##predicting the Scores for test data
print(Y_pred)


# In[17]:


#now print the Y_test.
Y_test


# In[18]:


#plotting line on test data
plt.plot(X_test,Y_pred,color='red')
plt.scatter(X_test,Y_test,color='black',marker='+')
plt.xlabel("Hours",fontsize=15)
plt.ylabel("Scores",fontsize=15)
plt.title("Regression line(Test set)",fontsize=10)
plt.show()


# ### Comparing scores (Actual vs Predicted)

# In[19]:


Y_test1 = list(Y_test)
prediction=list(Y_pred)
df_compare = pd.DataFrame({ 'Actual':Y_test1,'Result':prediction})
df_compare


# ### Accuracy of the model.

# In[24]:


from sklearn import metrics
metrics.r2_score(Y_test,Y_pred)##Goodness of fit Test


# #### Above 94% shows above is a good fitted model.
# 
# ### Predicting the error

# In[25]:


from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[26]:


MSE = metrics.mean_squared_error(Y_test,Y_pred)
root_E = np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))
Abs_E = np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))
print("Mean Squared Error = ",MSE)
print("Root Mean Squared Error = ",root_E)
print("Mean Absolute Error = ",Abs_E)


# ### Predicting the score.

# In[23]:


Prediction_score = linreg.predict([[9.25]])
print("predicted score for a student studying 9.25 hours :",Prediction_score)


# # conclusion:
# 
# ### From the above result we can say that if a student studied for 9.25 hours then the student will secure 93.69 marks
# 
# # THANK YOU !!
