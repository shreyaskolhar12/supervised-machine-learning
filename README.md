# supervised-machine-learning
THE SPARKS FOUNDATION
TASK1-TO EXPLORE SUPERVISED MASHINE LEARNING(LINEAR REGRESSION)
NAME- SHREYAS SACHIN KOLHAR
##Importing important libraries---
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
Data = pd.read_csv('C:\\Users\\dell\\Desktop\\student_scores.csv')
print("Data is successfully imported")
print (Data)
Data is successfully imported
    Hours  Scores
0     2.5      21
1     5.1      47
2     3.2      27
3     8.5      75
4     3.5      30
5     1.5      20
6     9.2      88
7     5.5      60
8     8.3      81
9     2.7      25
10    7.7      85
11    5.9      62
12    4.5      41
13    3.3      42
14    1.1      17
15    8.9      95
16    2.5      30
17    1.9      24
18    6.1      67
19    7.4      69
20    2.7      30
21    4.8      54
22    3.8      35
23    6.9      76
24    7.8      86
print("Head")
print(Data.head())
Head
   Hours  Scores
0    2.5      21
1    5.1      47
2    3.2      27
3    8.5      75
4    3.5      30
print("tail")
print(Data.tail())
tail
    Hours  Scores
20    2.7      30
21    4.8      54
22    3.8      35
23    6.9      76
24    7.8      86
print("describe :")
print(Data.describe())
describe :
           Hours     Scores
count  25.000000  25.000000
mean    5.012000  51.480000
std     2.525094  25.286887
min     1.100000  17.000000
25%     2.700000  30.000000
50%     4.800000  47.000000
75%     7.400000  75.000000
max     9.200000  95.000000
plt.boxplot(Data)
plt.show()

Visualizing Data.
##ploting Scatter plot----
plt.xlabel('Hours',fontsize=15)
plt.ylabel('Scores',fontsize=15)
plt.title('Hours studied vs Score', fontsize=10)
plt.scatter(Data.Hours,Data.Scores,color='blue',marker='*')
plt.show()

This plot shows that as much of hours you study high score you will secure.
X = Data.iloc[:,:-1].values
Y = Data.iloc[:,1].values
X
array([[2.5],
       [5.1],
       [3.2],
       [8.5],
       [3.5],
       [1.5],
       [9.2],
       [5.5],
       [8.3],
       [2.7],
       [7.7],
       [5.9],
       [4.5],
       [3.3],
       [1.1],
       [8.9],
       [2.5],
       [1.9],
       [6.1],
       [7.4],
       [2.7],
       [4.8],
       [3.8],
       [6.9],
       [7.8]])
Y
array([21, 47, 27, 75, 30, 20, 88, 60, 81, 25, 85, 62, 41, 42, 17, 95, 30,
       24, 67, 69, 30, 54, 35, 76, 86], dtype=int64)
Preparing data and splitting into train and test sets.
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state = 0,test_size=0.2)
## We have Splitted Our Data Using 80:20 RULe(PARETO)
print("X train.shape =", X_train.shape)
print("Y train.shape =", Y_train.shape)
print("X test.shape =", X_test.shape)
print("Y test.shape =", Y_test.shape)
X train.shape = (20, 1)
Y train.shape = (20,)
X test.shape = (5, 1)
Y test.shape = (5,)
Training the model.
from sklearn.linear_model import LinearRegression
linreg=LinearRegression()
##Fitting Training Data
linreg.fit(X_train,Y_train)
print("Training our algorithm is finished")
Training our algorithm is finished
print("B0 =",linreg.intercept_,"\nB1 =",linreg.coef_)## β0 is Intercept & Slope of the line is β1.,"

##plotting the REGRESSION LINE---
Y0 = linreg.intercept_ + linreg.coef_*X_train
B0 = 2.018160041434683 
B1 = [9.91065648]
##plotting on train data
plt.scatter(X_train,Y_train,color='green',marker='+')
plt.plot(X_train,Y0,color='orange')
plt.xlabel("Hours",fontsize=15)
plt.ylabel("Scores",fontsize=15)
plt.title("Regression line(Train set)",fontsize=10)
plt.show()

Test data.
Y_pred=linreg.predict(X_test)##predicting the Scores for test data
print(Y_pred)
[16.88414476 33.73226078 75.357018   26.79480124 60.49103328]
#now print the Y_test.
Y_test
array([20, 27, 69, 30, 62], dtype=int64)
#plotting line on test data
plt.plot(X_test,Y_pred,color='red')
plt.scatter(X_test,Y_test,color='black',marker='+')
plt.xlabel("Hours",fontsize=15)
plt.ylabel("Scores",fontsize=15)
plt.title("Regression line(Test set)",fontsize=10)
plt.show()

Comparing scores (Actual vs Predicted)
Y_test1 = list(Y_test)
prediction=list(Y_pred)
df_compare = pd.DataFrame({ 'Actual':Y_test1,'Result':prediction})
df_compare
Actual	Result
0	20	16.884145
1	27	33.732261
2	69	75.357018
3	30	26.794801
4	62	60.491033
Accuracy of the model.
from sklearn import metrics
metrics.r2_score(Y_test,Y_pred)##Goodness of fit Test
0.9454906892105356
Above 94% shows above is a good fitted model.
Predicting the error
from sklearn.metrics import mean_squared_error,mean_absolute_error
MSE = metrics.mean_squared_error(Y_test,Y_pred)
root_E = np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))
Abs_E = np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))
print("Mean Squared Error = ",MSE)
print("Root Mean Squared Error = ",root_E)
print("Mean Absolute Error = ",Abs_E)
Mean Squared Error =  21.5987693072174
Root Mean Squared Error =  4.6474476121003665
Mean Absolute Error =  4.6474476121003665
Predicting the score.
Prediction_score = linreg.predict([[9.25]])
print("predicted score for a student studying 9.25 hours :",Prediction_score)
predicted score for a student studying 9.25 hours : [93.69173249]
conclusion:
From the above result we can say that if a student studied for 9.25 hours then the student will secure 93.69 marks
THANK YOU !!
