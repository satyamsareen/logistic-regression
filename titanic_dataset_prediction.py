import numpy as np
import pandas as pd
from sklearn import preprocessing as p
from sklearn.linear_model import LogisticRegression
data_frame=pd.read_csv("titanic_x_y_train.csv")
test_frame=pd.read_csv("titanic_x_test.csv")
print(data_frame.columns)
print(data_frame.head())
print(test_frame.head())
del data_frame["Name"]
del test_frame["Name"]
del data_frame["Cabin"]
del test_frame["Cabin"]
del data_frame["Ticket"]
del test_frame["Ticket"]
print("data frame shape is",data_frame.shape)
print("test frame shape is",test_frame.shape)
def getNumber(str):
    if str=="male":
        return 1
    else:
        return 2
def embarked_to_num(str):
    if str=="Q":
        return 1
    elif str=="S":
        return 2
    else:
        return 3
data_frame["gender"]=data_frame["Sex"].apply(getNumber)
test_frame["gender"]=test_frame["Sex"].apply(getNumber)
data_frame["Embarked_num"]=data_frame["Embarked"].apply(embarked_to_num)
test_frame["Embarked_num"]=test_frame["Embarked"].apply(embarked_to_num)
del data_frame["Sex"]
del test_frame["Sex"]
del data_frame["Embarked"]
del test_frame["Embarked"]
data_frame["Survived_new"]=data_frame["Survived"]
del data_frame["Survived"]
print(data_frame.head())
print(data_frame.columns)
meanS= data_frame[data_frame.Survived_new==1].Age.mean()
data_frame["Age"]=np.where(pd.isnull(data_frame.Age) & data_frame["Survived_new"]==1 ,meanS,data_frame["Age"])
meanNS=data_frame[data_frame.Survived_new==0].Age.mean()
data_frame.Age.fillna(meanNS,inplace=True)
#---------------------------------------------------------------------------------------------------------------
mean1= test_frame[test_frame.Embarked_num==1].Age.mean()
mean2= test_frame[test_frame.Embarked_num==2].Age.mean()
test_frame["Age"]=np.where(pd.isnull(test_frame.Age) & test_frame["Embarked_num"]==1 ,mean1,test_frame["Age"])
test_frame["Age"]=np.where(pd.isnull(test_frame.Age) & test_frame["Embarked_num"]==2 ,mean2,test_frame["Age"])
mean3= test_frame[test_frame.Embarked_num==3].Age.mean()
test_frame.Age.fillna(mean3,inplace=True)
print(data_frame.describe())
print(data_frame.head())
clf=LogisticRegression(multi_class="ovr",solver="liblinear")
# print(data_frame.iloc[:,0])
print("data frame shape is",data_frame.shape)
print("test frame shape is",test_frame.shape)
for i in range(data_frame.shape[1]-1):
    for j in range(i,7):
        data_frame.insert(i+7, "mult"+str(i)+str(j),data_frame.iloc[:,i]*data_frame.iloc[:,j])
        test_frame.insert(i + 7, "mult" + str(i)+str(j),test_frame.iloc[:,i]*test_frame.iloc[:,j])
print(data_frame.columns)
print(test_frame.columns)
print("data frame shape is",data_frame.shape)
print("test frame shape is",test_frame.shape)
data=np.array(data_frame)
# test=scaler.transform(test)
# test=test[:,0:test.shape[1]-1]
last=data_frame.shape[1]-1
clf.fit(data[:,0:last],data[:,last])
print(clf.score(data[:,0:last],data[:,last]))
print(test_frame.isnull().sum())
np.savetxt("predictions_titanic.csv",clf.predict(test_frame), delimiter=',',fmt="%d")
