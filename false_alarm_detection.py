import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import accuracy_score

df1 = pd.read_excel("C:\\Users\\HP\\Downloads\\False Alarm Cases.xlsx")
df1.info()
print(df1.head())
df1.drop(['Case No.','Unnamed: 8','Unnamed: 9','Unnamed: 10'],axis=1,inplace=True)
df1.info()


x = df1.drop('Spuriosity Index(0/1)',axis=1)
y = df1['Spuriosity Index(0/1)']

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2)

lr = LogisticRegression()
lr.fit(x_train,y_train)
joblib.dump(lr,'FAD_data.pkl')


lr = joblib.load('FAD_data.pkl')

y_pred = lr.predict(x_test)
print(y_pred)

print("the accuracy of given model is ",accuracy_score(y_pred,y_test)*100,"%")

