# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 10:26:19 2023

@author: domingosdeeularia
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 09:42:57 2023

@author: domingosdeeulariadumba
"""



""" IMPORTING LIBRARIES """


# EDA and Plotting

import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as pp
pp.style.use('ggplot')


# Machine Learning Modules
from xgboost import XGBRegressor as XGB
from sklearn.model_selection import train_test_split as tts, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression as LinReg, Ridge, Lasso
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score

# To save and deploy the best ML model

import joblib as jbl
import gradio as gd


# For ignoring warnings

import warnings
warnings.filterwarnings('ignore')




"""" DATA CLEANING AND PREPARATION """


  
   
df = pd.read_csv("LaptopPrice_Dataset.csv")


df.columns


df = df.drop(['Unnamed: 0'], axis = 1)


   
name = df['Name'].str.split(' ', n = 1, expand = True)

df['Brand'] = name[0]

df['Product'] = name[1].str.split('Dual Core| Hexa Core| Quad Core| Core',
                                  expand = True)[0]


prc_type = ['Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'Intel Core i9',
            'AMD Ryzen 3', 'AMD Ryzen 5', 'AMD Ryzen 7', 'AMD Ryzen 9']

df['ProcessorType'] = 'Other'

for k in range(len(df)):
    for i in prc_type:
        if i in str(df['Processor'][k]):
                    df['ProcessorType'][k] = i
                
core_type = ['Dual Core', 'Quad Core', 'Hexa Core', 'Octa Core']

df['ProcessorCore'] = 'Core'  

for j in range(len(df)):
    for k in core_type:
        if k in str(df['Processor'][j]):
                    df['ProcessorCore'][j] = k

df['ProcessorGen'] = 'Undefined'

for i in range(len(df)):
    for j in range(7,12):
        if str(j)+'th' in (df['Processor'][i]) or str(j)+'th' in (df['Name'][i]):
            df['ProcessorGen'][i] = str(j)+'th'


    
df['RAM'] = df['RAM'].str.split(' ', expand = True)[0]

df['RAM'] = df['RAM'].astype(int)  

  
df['RAM'][df['RAM'] == 'Upgradable'] = 8

df['RAM'] = df['RAM'].astype(int)

   
df['OpSys'] = ''

for k in range(len(df)):
    if 'Windows' in str(df['Operating System'][k]):
        df['OpSys'][k] = 'windows'
    elif 'Mac' in str(df['Operating System'][k]):
       df['OpSys'][k] = 'Mac'
    else:
       df['OpSys'][k] = 'Other OS' 
       
   
df['Storage'] = df['Storage'].replace('1 TB', '1000 GB', regex = True)

Storage = df['Storage'].str.split('|', expand = True)

df['HDD Storage'] = ''

df['SSD Storage'] = ''

for j in range(len(Storage)):
    if 'HDD' in str(Storage[0][j]):
        df['HDD Storage'][j] = Storage[0][j]
        df['SSD Storage'][j] = Storage[1][j]
    else:
        df['SSD Storage'][j] = Storage[0][j]
        df['HDD Storage'][j] = Storage[1][j]

df['HDD Storage'] = df['HDD Storage'].str.split('GB', expand = True)[0]

df['HDD Storage'] = df['HDD Storage'].fillna(0).astype(int)

df['SSD Storage'] = df['SSD Storage'].str.split('GB', expand = True)[0]

df['SSD Storage'] = df['SSD Storage'].fillna(0).astype(int)
   
    
df['SSD Storage'][df['SSD Storage'] == 'M.2 Slot for SSD Upgrade'] = 1000

df['SSD Storage'] = df['SSD Storage'].fillna(0).astype(int)


  
df['Display'] = df['Display'].replace('15.6 inches', '39.62 cm', regex = True)

for i in range(len(df)):
            if 'cm' not in str(df['Display'][i]):
                    df['Display'][i] = 'Other'
                    
df['Display'] = df['Display'].str.split('cm', expand = True)[0]


df['Warranty'] = df['Warranty'].replace('One-year', '1 Year', regex = True)

for i in range(len(df)):
    if 'Months' in str(df['Warranty'][i][:9]):
        df['Warranty'][i] = str(float(str(df['Warranty'][i])[:2])/12)+ ' Year'
    elif 'Year' not in str(df['Warranty'][i][:7]):
        df['Warranty'][i] = 'Other'
        
df['Warranty'] = df['Warranty'].str.split('Year', expand = True)[0]

for i in range(len(df)):
    if df['Warranty'][i] != 'Other':
       df['Warranty'][i] =  df['Warranty'][i] + ' Year'


df['Price'] = df['Price'].replace(['₹', ','], '', regex = True).astype(float)


df_ltp = pd.concat([df['Brand'], df['Product'], df['ProcessorType'],
                  df['ProcessorCore'], df['ProcessorGen'], df['RAM'],
                  df['OpSys'], df['HDD Storage'], df['SSD Storage'], 
                  df['Display'], df['Warranty'], df['Rating'], df['Price']],
                 axis = 1)




"""" EXPLORATORY DATA ANALYSIS """



df_ltp.describe().iloc[:,:-1]

df_ltp.describe()['Price']


df_ltp.info()


  
numeric_vls = list(df_ltp.describe().columns)

for i in numeric_vls:
    sb.set(rc = {'figure.figsize':(20,10)}, font_scale = 1)
    sb.displot(df_ltp[i])
    pp.savefig ('{0}_displot.png'.format(i))
    pp.close()

    
for j in numeric_vls:
    pp.figure()
    sb.set(rc = {'figure.figsize':(16,10)}, font_scale = 2)
    sb.kdeplot(df_ltp[j])
    pp.savefig ('{0}_kdeplot.png'.format(j))
    pp.close()

 
brplt_vars = ['Brand']+list(df_ltp.columns)[2:-2]

for j in brplt_vars:
    pp.figure()
    sb.set(rc = {'figure.figsize':(20,10)}, font_scale = 2)
    sb.barplot(x = df_ltp[j], y = df_ltp['Price'])
    pp.xticks(rotation = 55)
    pp.savefig ('{0}_barplot.png'.format(j))
    pp.close()
    
sb.pairplot(df_ltp, x_vars = numeric_vls[:3],
            y_vars = ['Price'], height = 8, aspect = .8, kind = 'scatter')
pp.savefig("scatterplot.png")
pp.close()

sb.set(rc={'figure.figsize':(20,10)}, font_scale= 2)
sb.stripplot(x='Rating', y='Price', data=df_ltp)
pp.savefig("Rating_stripplot.png")
pp.close()


sb.heatmap(df_ltp.corr(), annot=True)
pp.savefig("correlation_heatmap.png")
pp.close()





X, y = pd.get_dummies(df_ltp.iloc[:,:-1],
                      drop_first = True), df_ltp.iloc[:,-1]

def check_model(X, y, model):
    
    X_scaled = pd.DataFrame(RobustScaler().fit_transform(X), 
                            columns = X.columns)
       
    if model == Lasso:
        
        data = X_scaled
        X_train, X_test, y_train, y_test = tts(data, y, test_size = 0.2,
                                               random_state = 97)
        feature_scale = X_train.std(axis=0) 
        alphas = [0.01 * scale for scale in feature_scale]
        param_grid = {'alpha': alphas}
        grid_search = GridSearchCV(model(), param_grid, cv=5, 
                                   scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_params_['alpha']
        final_model = model(alpha = best_model).fit(X_train, y_train)
        y_model_pred = final_model.predict(X_test)
        mse_model = mean_squared_error(y_test, y_model_pred)
        r2_model = r2_score(y_test, y_model_pred)
        
    else:
        # Recursive Feature Elimination, RFE 
        rfe = RFE(model(), n_features_to_select = None)
        rfe_fit = rfe.fit(X_scaled,y)
        data = X_scaled[X_scaled.columns[rfe_fit.support_.tolist()]]
        
        X_train, X_test, y_train, y_test = tts(data, y, test_size = 0.2,
                                               random_state = 97)
        if model == Ridge:
            
            feature_scale = X_train.std(axis=0) 
            alphas = [0.01 * scale for scale in feature_scale]
            param_grid = {'alpha': alphas}
            grid_search = GridSearchCV(model(), param_grid, cv=5, 
                                       scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_params_['alpha']
            final_model = model(alpha = best_model).fit(X_train, y_train)
            y_model_pred = final_model.predict(X_test)
            mse_model = mean_squared_error(y_test, y_model_pred)
            r2_model = r2_score(y_test, y_model_pred)
            
        if model == XGB:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5]}
            
            grid_search = GridSearchCV(model(), param_grid, cv=5,
                                       scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            final_model = model(**best_model.get_xgb_params()
                                ).fit(X_train, y_train)
            y_model_pred = final_model.predict(X_test)
            mse_model = mean_squared_error(y_test, y_model_pred)
            r2_model = r2_score(y_test, y_model_pred)
            
        elif model == LinReg:
            final_model = model().fit(X_train, y_train)
            y_model_pred = final_model.predict(X_test)
            mse_model = mean_squared_error(y_test, y_model_pred)
            r2_model = r2_score(y_test, y_model_pred)
                        
    return mse_model, r2_model, final_model

  
XGB_metrics = check_model(X, y, XGB)[:-1]
LinReg_metrics = check_model(X, y, LinReg)[:-1]
Lasso_metrics = check_model(X, y, Lasso)[:-1]
Ridge_metrics = check_model(X, y, Ridge)[:-1]

df_metrics = pd.DataFrame([XGB_metrics, LinReg_metrics, Lasso_metrics,
                           Ridge_metrics], columns=['MSE', 'R2'], index = ['XGB', 'Linear Regression',
      
mse, r2, model = check_model(X, y, Ridge)


jbl.dump(model,"LaptopPriceModel.sav")


    
def predict(Brand, Product, ProcessorType, ProcessorCore,
                      ProcessorGen, RAM, Opsys, HDD_Storage,
                      SSD_Storage, Display, Warranty, Rating):
    
        model=joblib.load('LaptopPriceModel.sav')

        outcome = model.predict([[Brand, Product, ProcessorType, ProcessorCore,
                              ProcessorGen, RAM, Opsys, HDD_Storage,
                              SSD_Storage, Display, Warranty, Rating]])
        return outcome


brand = gd.Textbox(label = "Enter the brand of the laptop")
product = gd.Textbox(label = "Enter the product type")
processortype = gd.Textbox(label = "Enter the processor type")
processorcore = gd.Textbox(label = "Enter the processor core")
processorgen = gd.Textbox(label = "Enter the processor generation")
ram = gd.Number(label = "Enter the RAM")
opsys = gd.Textbox(label = "Enter the Operating System")
hddstorage = gd.Number(label = "Enter the HDD Storage")
ssdstorage = gd.Number(label = "Enter the SSD Storage")
display = gd.Textbox(label = "Enter the Display")
warranty = gd.Textbox(label = "Enter the Warranty")
rating = gd.Number(label = "Enter the Rating")

Price = gd.Number()

webapp = gd.Interface(fn = predict, inputs = [brand, product, processortype, processorcore,
                                              processorgen, ram,  opsys,
                                              hddstorage, ssdstorage, display,

webapp.launch(share = 'True')
______________________________________________END_____________________________________________


