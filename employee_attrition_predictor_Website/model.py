# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 19:26:52 2022

@author: sonal
"""
#importing necessary libraries
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

#reading the dataset
df=pd.read_csv('HR_attrition_dataset.csv')
#dropping zero-variant features
df.drop(['EmployeeCount', 'EmployeeID','Over18','StandardHours'],axis=1,inplace=True)

#missing value handling
df.fillna({'NumCompaniesWorked':df['NumCompaniesWorked'].mode()[0],'TotalWorkingYears':df['TotalWorkingYears'].mode()[0]},inplace=True)
# Handling outliers by interquartile rule based capping
outlier_lst=['NumCompaniesWorked','YearsSinceLastPromotion','TotalWorkingYears','YearsAtCompany']
for i in outlier_lst:
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3 - Q1
    u_lim=(Q3+1.5*IQR)
    df[i].where(df[i] <=u_lim,u_lim, inplace=True)  #replacing all upper outliers with upper-limit
    df[i]=df[i].astype('int')
#feature engineering on Age and MonthlyIncome
df['Age']=pd.cut(df['Age'],bins=[17,25,35,45,55,64],labels=['25 & Below','26-35','36-45','46-55','56 & Above'])
df['Age'] = df['Age'].astype('object')
#Function to categorise income into slabs
def categorise_income(row):  
    if row['MonthlyIncome'] > 10000 and row['MonthlyIncome'] <= 25000:
        return '10k-25k'
    elif row['MonthlyIncome'] > 25001 and row['MonthlyIncome'] <= 50000:
        return '25k-50k'
    elif row['MonthlyIncome'] > 50001 and row['MonthlyIncome'] <= 75000:
        return '50k-75k'
    elif row['MonthlyIncome'] > 75001 and row['MonthlyIncome'] <= 100000:
        return '75k-1L'
    elif row['MonthlyIncome'] > 100001 and row['MonthlyIncome'] <= 125000:
        return '1L-1.25L'
    elif row['MonthlyIncome'] > 125001 and row['MonthlyIncome'] <= 150000:
        return '1.25L-1.5L'
    elif row['MonthlyIncome'] > 150001 and row['MonthlyIncome'] <= 175000:
        return '1.5L-1.75L'
    elif row['MonthlyIncome'] > 10000 and row['MonthlyIncome'] <= 200000:
        return '1.75L-2L'
df['MonthlyIncome'] = df.apply(lambda row: categorise_income(row), axis=1)
# feature selection/elimination
df.drop(columns=['Education','StockOptionLevel','JobLevel','Gender','DistanceFromHome','PercentSalaryHike','TrainingTimesLastYear','NumCompaniesWorked','YearsWithCurrManager','YearsSinceLastPromotion'],inplace=True)
# Encoding target variable
df['Attrition']=df['Attrition'].map({'Yes': 1, 'No': 0})
# Label encoding BusinessTravel feature
df['BusinessTravel']=df['BusinessTravel'].map({'Travel_Frequently': 2,'Travel_Rarely': 1, 'Non-Travel': 0})
# Feature and Target separation for train test split with 30% testing data
X=df.drop(['Attrition'],axis=1)
y=df['Attrition']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=.3,stratify = y)
# One hot encoding remaining categorical features
ohe = make_column_transformer(
    (OneHotEncoder(drop='first'), ['Age','MonthlyIncome','Department','JobRole','MaritalStatus','EducationField']),
    remainder='passthrough')
ohencoded_X_train = ohe.fit_transform(X_train)
X_train = pd.DataFrame(ohencoded_X_train.todense(), columns=ohe.get_feature_names_out())
ohencoded_X_test=ohe.transform(X_test)
X_test = pd.DataFrame(ohencoded_X_test.todense(), columns=ohe.get_feature_names_out())
ohencoded=open('ohe.pickle','wb')
pickle.dump(ohe,ohencoded)
ohencoded.close()
# Feature scaling- Standardisation
ss_scale=StandardScaler()
X_train_std=ss_scale.fit_transform(X_train)
X_train=pd.DataFrame(X_train_std,columns = X_train.columns)
X_test_std=ss_scale.transform(X_test)
X_test=pd.DataFrame(X_test_std,columns = X_test.columns)
standardize=open('std_scaler.pickle','wb')
pickle.dump(ss_scale,standardize)
standardize.close()
# model building
best_model=KNeighborsClassifier(weights= 'distance', n_neighbors= 35, metric= 'euclidean')
best_model.fit(X_train, y_train)
# =============================================================================
# prediction = best_model.predict(X_test)
# print(confusion_matrix(y_test, prediction))
# print('Accuracy:',accuracy_score(y_test, prediction))
# print('F1_score:',f1_score(y_test, prediction))
# print('Recall:',recall_score(y_test, prediction))
# print('Precision:',precision_score(y_test, prediction))
# print('ROC_AUC:',roc_auc_score(y_test, prediction))
# =============================================================================
model=open('model.pickle','wb')
pickle.dump(best_model,model)
model.close()
