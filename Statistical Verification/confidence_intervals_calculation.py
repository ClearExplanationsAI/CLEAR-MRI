#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#In order for this code to be executed one needs the detailed spreadsheets that are created after each model's training
#and testing process. The code can then read the spreadsheets and calculate the confidence intervals based on the method
#described in Appendix A1. The confidence intervals are the ones shown in Tables 1, 4 and 5 in the paper. 


# In[ ]:


import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import math


# In[ ]:


#Calculate confidence intervals

#change to the respective directory containing all spreadsheets with results for each model
path = 'C:/Users/marga/Desktop/Spreadsheets/'

#Save the confidence intervals in a new spreadsheet
conf_intervals_ss = pd.DataFrame(columns=['Model','Values','Mean accuracy','Std error','CI(+/-)'])

files = [f for f in listdir(path) if isfile(join(path, f))]

row=0

#Loop through each file in the directory
for file in files:

    parts = file.split('.')
        
    test_xl = pd.read_excel(path+file,sheet_name='Sheet1')
    
    folds = test_xl['val group'].copy(deep=True).values
    folds = folds[0:80]
    
    #Iterate through test_acc, bal_acc, impair1, bal_impair1 with are the 4 accuracies reported in the tables
    for column in ['test_acc','bal_acc','impair1','bal_impair1']:
    
        model_accs = test_xl[column].copy(deep=True).values
        model_accs = model_accs[0:80]

        fold_accs1 = []
        fold_accs2 = []
        fold_accs3 = []
        fold_accs4 = []

        for i in range(len(folds)):
            if folds[i]==1:
                fold_accs1.append(model_accs[i])
            elif folds[i]==2:
                fold_accs2.append(model_accs[i])
            elif folds[i]==3:
                fold_accs3.append(model_accs[i])
            else:
                fold_accs4.append(model_accs[i])

        #Calculate the average accuracy across each fold
        mean_fold1 = np.mean(fold_accs1)
        mean_fold2 = np.mean(fold_accs2)
        mean_fold3 = np.mean(fold_accs3)
        mean_fold4 = np.mean(fold_accs4)
        
        #Combine the 4 accuracies in one array
        model_folds = np.array([mean_fold1,mean_fold2,mean_fold3,mean_fold4])

        #Dof correction method
        down_scale = 0.45
        N = len(model_folds)
        apparent_dof = N-1
        effective_dof = apparent_dof*down_scale

        #std error and confidence intervals
        model_stddev = np.std(model_folds,ddof=1)
        model_stderror = model_stddev / math.sqrt(effective_dof+1)
        ts = t.ppf(0.025,effective_dof)
        #print(ts)
        model_CI = np.abs(ts)*model_stderror

        #Save the descriptive statistics and confidence intervals
        conf_intervals_ss.loc[row,'Model'] = parts[0] 
        conf_intervals_ss.loc[row,'Values'] = column
        conf_intervals_ss.loc[row,'Mean accuracy'] = np.mean(model_folds)
        conf_intervals_ss.loc[row,'Std error'] = model_stderror
        conf_intervals_ss.loc[row,'CI(+/-)'] = model_CI
        
        row+=1


# In[ ]:


#Logistic regression accuracies
reg_accs = []

#Logistic regression were hard-coded as they were not part of a spreadsheet that was in the same format with the rest of the models
reg_test = np.array([0.846666666666667,0.846666666666667,0.842222222222222,0.84])
bal_reg_test = np.array([0.818942367279488,0.813158438339186,0.804138263444192,0.803966122701921])
reg_impair1 = np.array([0.781818181818182,0.781818181818182,0.76969696969697,0.768181818181818])
bal_reg_impair1 = np.array([0.78042328042328,0.780092592592593,0.767636684303351,0.766203703703704])

reg_accs.append(reg_test)
reg_accs.append(bal_reg_test)
reg_accs.append(reg_impair1)
reg_accs.append(bal_reg_impair1)

#Perform the same calculations for logistic regression and save the results at the end of the new spreadsheet
for fold,column in zip(reg_accs,['test_acc','bal_acc','impair1','bal_impair1']):
    reg_stddev = np.std(fold,ddof=1)
    reg_stderror = reg_stddev / math.sqrt(effective_dof+1)
    reg_CI = np.abs(ts)*reg_stderror
    
    conf_intervals_ss.loc[row,'Model'] = 'Logistic regression'
    conf_intervals_ss.loc[row,'Values'] = column
    conf_intervals_ss.loc[row,'Mean accuracy'] = np.mean(fold)
    conf_intervals_ss.loc[row,'Std error'] = reg_stderror
    conf_intervals_ss.loc[row,'CI(+/-)'] = reg_CI
    
    row+=1
    
conf_intervals_ss.to_excel(path+'model_stats.xlsx')

