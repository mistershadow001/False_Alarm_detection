

import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
from flask import Flask
from flask import request

app = Flask(__name__)
@app.route('/train')
def train():
    df1 = pd.read_excel("C:\\Users\\HP\\Downloads\\False Alarm Cases.xlsx")
    df1.drop(['Case No.','Unnamed: 8','Unnamed: 9','Unnamed: 10'],axis=1,inplace=True)
    x = df1.drop('Spuriosity Index(0/1)',axis=1)
    y = df1['Spuriosity Index(0/1)']
    lr = LogisticRegression()
    lr.fit(x,y)
    joblib.dump(lr, 'FAD_data.pkl')
    return "model trained successfully"



@app.route('/prediction',methods=['POST'])
def test():
    data = request.get_json()
    lr = joblib.load('FAD_data.pkl')
    y_pred = lr.predict(data,index=[0])
    if y_pred == 0:
        return "The Alarm is True"
    else:
        return "The Alarm is False"



app.run(port=8080)


