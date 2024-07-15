#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#In order for this code to be executed one needs the detailed spreadsheets that are created after each model's training
#and testing process and contain the individual predictions and probabilities for each patient. 
#The code can then read the spreadsheets and calculate AUC and F1 scors along with the confidence intervals based on the method
#described in Appendix A1. This corresponds to Table 2 in the paper.


# In[ ]:


import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import math
from sklearn.metrics import roc_auc_score, PrecisionRecallDisplay, precision_score,f1_score
from sklearn import metrics


# In[ ]:


#Calculate AUC and F1 scores

#Change to the directory containing the detailed spreadsheets
path = 'C:/Users/marga/Desktop/AUC spreadsheets/'

#Spreadsheet with the results for all the patients
auc_f1_scores_ss = pd.DataFrame(columns=['Model','AUC','Std error','CI(+/-)','F1','Std error f1','CI F1'])
#Spreadsheet with the results for severe or moderate aphasic categories of patients
auc_f1_scores_ss2 = pd.DataFrame(columns=['Model','AUC','Std error','CI(+/-)','F1','Std error f1','CI F1'])

files = [f for f in listdir(path) if isfile(join(path, f))]

row=0

#Loop through all the files in the directory
for file in files:

    parts = file.split('.')
        
    spreadsheet = pd.read_excel(path+file,sheet_name='Sheet1')
    
    #separate the impaired patients in a separate spreadsheet
    impaired_sheet = spreadsheet[spreadsheet.impair==1]
    
    auc_scores = []
    f1_scores = []
    impair_auc_scores = []
    impair_f1_scores = []
    
    #Calculate AUC and F1 scores for each of the 4 validation folds
    for val_group in range(1,5):
        
        fpr, tpr, thresholds = metrics.roc_curve(spreadsheet.label[spreadsheet.valgroup==val_group], spreadsheet.prob[spreadsheet.valgroup==val_group])
        roc_auc = metrics.auc(fpr, tpr)
        
        f1_stat = f1_score(spreadsheet.label[spreadsheet.valgroup==val_group], spreadsheet.prediction[spreadsheet.valgroup==val_group])
        
        #collect the 4 scores into one array
        auc_scores.append(roc_auc)
        f1_scores.append(f1_stat)
        
        #impair1 scores
        fpr, tpr, thresholds = metrics.roc_curve(impaired_sheet.label[impaired_sheet.valgroup==val_group], impaired_sheet.prob[impaired_sheet.valgroup==val_group])
        impair_roc_auc = metrics.auc(fpr, tpr)
        
        impair_f1_stat = f1_score(impaired_sheet.label[impaired_sheet.valgroup==val_group], impaired_sheet.prediction[impaired_sheet.valgroup==val_group])
    
        impair_auc_scores.append(impair_roc_auc)
        impair_f1_scores.append(impair_f1_stat)
    
    #Dof correction described in Appendix A1
    down_scale = 0.45
    N = len(auc_scores)
    apparent_dof = N-1
    effective_dof = apparent_dof*down_scale

    #std error and confidence intervals for AUC scores
    auc_stddev = np.std(auc_scores,ddof=1)
    auc_stderror = auc_stddev / math.sqrt(effective_dof+1)
    ts = t.ppf(0.025,effective_dof)
    auc_CI = np.abs(ts)*auc_stderror
    
    #F1 scores CI
    f1_stddev = np.std(f1_scores,ddof=1)
    f1_stderror = f1_stddev / math.sqrt(effective_dof+1)
    ts = t.ppf(0.025,effective_dof)
    f1_CI = np.abs(ts)*f1_stderror
    
    #std error and confidence intervals for impair AUC scores
    impair_auc_stddev = np.std(impair_auc_scores,ddof=1)
    impair_auc_stderror = impair_auc_stddev / math.sqrt(effective_dof+1)
    ts = t.ppf(0.025,effective_dof)
    impair_auc_CI = np.abs(ts)*impair_auc_stderror
    
    #impair F1 scores CI
    impair_f1_stddev = np.std(impair_f1_scores,ddof=1)
    impair_f1_stderror = impair_f1_stddev / math.sqrt(effective_dof+1)
    ts = t.ppf(0.025,effective_dof)
    impair_f1_CI = np.abs(ts)*impair_f1_stderror

    #Save the AUC and F1 scores along with the descriptive statistics for all patients
    auc_f1_scores_ss.loc[row,'Model'] = parts[0] 
    auc_f1_scores_ss.loc[row,'AUC'] = np.mean(auc_scores)
    auc_f1_scores_ss.loc[row,'Std error'] = auc_stderror
    auc_f1_scores_ss.loc[row,'CI(+/-)'] = auc_CI
    auc_f1_scores_ss.loc[row,'F1'] = np.mean(f1_scores)
    auc_f1_scores_ss.loc[row,'Std error f1'] = f1_stderror
    auc_f1_scores_ss.loc[row,'CI F1'] = f1_CI
    
    #Save the AUC and F1 scores along with the descriptive statistics for the language impaired patients
    auc_f1_scores_ss2.loc[row,'Model'] = parts[0] 
    auc_f1_scores_ss2.loc[row,'AUC'] = np.mean(impair_auc_scores)
    auc_f1_scores_ss2.loc[row,'Std error'] = impair_auc_stderror
    auc_f1_scores_ss2.loc[row,'CI(+/-)'] = impair_auc_CI
    auc_f1_scores_ss2.loc[row,'F1'] = np.mean(impair_f1_scores)
    auc_f1_scores_ss2.loc[row,'Std error f1'] = impair_f1_stderror
    auc_f1_scores_ss2.loc[row,'CI F1'] = impair_f1_CI

    row+=1
    
auc_f1_scores_ss.to_excel(path+'AUC_F1_model_stats.xlsx')
auc_f1_scores_ss2.to_excel(path+'impair_AUC_F1_model_stats.xlsx')

