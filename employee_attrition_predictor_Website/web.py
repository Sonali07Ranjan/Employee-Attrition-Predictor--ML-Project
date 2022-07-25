# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 20:20:05 2022

@author: sonal
"""

from flask import Flask,render_template,request
import pickle
import pandas as pd
app=Flask(__name__)
std_scaler=pickle.load(open('std_scaler.pickle','rb'))
oh_encoder=pickle.load(open('ohe.pickle','rb'))
model=pickle.load(open('model.pickle','rb'))
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/result',methods=['POST'])
def predict_model():
    age=int(request.form['Age'])
    department=(request.form['Department'])
    business_travel=request.form['BusinessTravel']
    education_field=request.form['EducationField']
    job_role=request.form['JobRole']
    marital_status=request.form['MaritalStatus']
    monthly_income=float(request.form['MonthlyIncome'])
    tot_wrk_yrs=int(request.form['TotalWorkingYears'])
    yrs_at_compny=int(request.form['YearsAtCompany'])
        
    df = pd.DataFrame(columns=['Age', 'BusinessTravel', 'Department', 'EducationField',
                                'JobRole','MaritalStatus', 'MonthlyIncome', 
                                'TotalWorkingYears','YearsAtCompany'])
    df.loc[0] = [age,business_travel,department,education_field,job_role,marital_status,
                  monthly_income,tot_wrk_yrs,yrs_at_compny]
    #print(df.loc[0])
    #outlier handling
    df['TotalWorkingYears'].where(~(df.TotalWorkingYears > 13.5), other=13.5, inplace=True)
    df['YearsAtCompany'].where(~(df.YearsAtCompany > 18), other=18, inplace=True)

    #Feature engineering
    df['Age']=pd.cut(df['Age'],bins=[17,25,35,45,55,64],labels=['25 & Below','26-35','36-45','46-55','56 & Above'])
    df['MonthlyIncome']=pd.cut(df['MonthlyIncome'],bins=[10000,25000,50000,75000,100000,125000,150000,175000,200000],
                         labels=['10k-25k','25k-50k','50k-75k','75k-1L','1L-1.25L','1.25L-1.5L','1.5L-1.75L','1.75L-2L'])
    df['BusinessTravel']=df['BusinessTravel'].map({'Travel_Frequently': 2,'Travel_Rarely': 1, 'Non-Travel': 0})
    #One hot encoding
    ohencoded = oh_encoder.transform(df)
    df = pd.DataFrame(ohencoded.todense(), columns=oh_encoder.get_feature_names_out())
    #Standard scaling
    df_std=std_scaler.transform(df)
    df=pd.DataFrame(df_std,columns = df.columns)
    #Model prediction
    output=model.predict(df)
    output=output.item()
    if(output==0):
        att="No"
    else:
        att="Yes"    
    #print("Attrition possibility: ",att)
    #prediction=att
    #return prediction
    return render_template('result.html',prediction=att,age=age,business_travel=business_travel,department=department,education_field=education_field,job_role=job_role,marital_status=marital_status,
                  monthly_income=monthly_income,tot_wrk_yrs=tot_wrk_yrs,yrs_at_compny=yrs_at_compny)
if __name__ == '__main__':
    app.run()