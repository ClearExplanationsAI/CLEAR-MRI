#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##Statistical comparison of the Hybrid ROIs w/ ResNet-18 model with the rest of the models trained and reported in the paper 
#"Predicting recovery following stroke: Deep learning, multimodal data and feature selection using explainable AI" by White et al.

#In order for this code to be executed one needs the detailed spreadsheets that are created after each model's training
#and testing process. The code can then read the spreadsheets and perform a 2-tailed test to compare balanced accuracies
#between the model that performed best in the mentioned work and the rest of the models whose results are presented in
#Table 1 of the paper.


# In[ ]:


import numpy as np
import pandas as pd
import scipy.stats as stats
from os import listdir
from os.path import isfile, join
import math
from scipy.stats import t


# In[ ]:


#change to the path that contains the spreadsheets with the results
path = 'C:/Users/marga/Desktop/Spreadsheets/'

files = [f for f in listdir(path) if isfile(join(path, f))]

print(files)

names_ss = dict()

#Create a dictionary to help with the organisation of the results from the statistical tests
for ss in files:
    parts = ss.split('.')
    if parts[0]=='CH8zoom_ResNet':
        names_ss['CH8zoom_ResNet']='ROIs-ResNet18'
    elif parts[0]=='CHS7_200zoom_ResNet':
        names_ss['CHS7_200zoom_ResNet']='Hybrid ROIs-ResNet18'
    elif parts[0]=='DAFT':
        names_ss['DAFT']='DAFT'
    elif parts[0]=='MultiRes':
        names_ss['MultiRes']='Early Fusion Hybrid-ResNet18'
    elif parts[0]=='ResNet3D':
        names_ss['ResNet3D']='MRI Scans-ResNet3D'
    elif parts[0]=='stitch_sym_First_paper':
        names_ss['stitch_sym_First_paper']='Hybrid Stitched Image-Lightweight CNN'
    elif parts[0]=='stitch_sym_ResNet':
        names_ss['stitch_sym_ResNet']='Hybrid Stitched Image-ResNet18'
    elif parts[0]=='Stitch0First_paper':
        names_ss['Stitch0First_paper']='Stitched Images-Lightweight CNN'
    elif parts[0]=='stitch0ResNet':
        names_ss['stitch0ResNet']='Stitched Images-ResNet18'
    elif parts[0]=='whiteNS_zoom_ResNet':
        names_ss['whiteNS_zoom_ResNet']='White Matter Tracts-ResNet18'
    elif parts[0]=='WHS200zoom_ResNet':
        names_ss['WHS200zoom_ResNet']='Hybrid White Matter Tracts-ResNet18'

#print(names_ss)


# In[ ]:


#Logistic regression balanced accuracies for each of the 4 validation folds
#I have hard-coded the numbers for logistic regression as they were not part of a spreadsheet in the same format with
#the rest of the models.

balanced_reg_test = np.array([0.818942367279488,0.813158438339186,0.804138263444192,0.803966122701921])


# In[ ]:


#Create a spreadsheet to save the statistical tests results
df_correction_ss = pd.DataFrame(columns=['Hybrid ROIs ResNet vs <Model>','Values','t-stat','p-value'])

row=0

#Search in the files in your directory with the spreadsheet to find the one for the Hybrid ROIs w/ ResNet-18
for ss in files:
    parts = ss.split('.')
    if parts[0]=='CHS7_200zoom_ResNet':
        test_xl = pd.read_excel(path+ss,sheet_name='Sheet1')
    
        #reads the numbers corresponding to the 4 validation groups
        folds = test_xl['val group'].copy(deep=True).values
        folds = folds[0:80]
        
        #reads the balanced test accuracies for the Hybrid ROIs model
        hybrid_ROIs_accs = test_xl['bal_acc'].copy(deep=True).values
        hybrid_ROIs_accs = hybrid_ROIs_accs[0:80]

        fold_accs1 = []
        fold_accs2 = []
        fold_accs3 = []
        fold_accs4 = []

        #Split the balanced accuracies into 4 groups, based on the validation group (=fold) used during training
        for i in range(len(folds)):
            if folds[i]==1:
                fold_accs1.append(hybrid_ROIs_accs[i])
            elif folds[i]==2:
                fold_accs2.append(hybrid_ROIs_accs[i])
            elif folds[i]==3:
                fold_accs3.append(hybrid_ROIs_accs[i])
            else:
                fold_accs4.append(hybrid_ROIs_accs[i])

        #calculate the mean balanced accuracy for each fold
        mean_fold1 = np.mean(fold_accs1)
        mean_fold2 = np.mean(fold_accs2)
        mean_fold3 = np.mean(fold_accs3)
        mean_fold4 = np.mean(fold_accs4)

        #Hybrid ROIs w/ ResNet accuracies combined in one array
        hybrid_ROIs_folds = np.array([mean_fold1,mean_fold2,mean_fold3,mean_fold4])
        
        #remove the Hybrid ROIs spreadsheet from the list of files as the average fold accuracies have been calculated above
        files.remove(ss)
        

#Perform the statistical test following the method for the degrees of freedom correction described in Appendix A1 of the paper
down_scale = 0.45
N = len(hybrid_ROIs_folds)
apparent_dof = N-1
effective_dof = apparent_dof*down_scale

if effective_dof<=1:
    print('ERROR: Effective degrees of freedom are too small to calculate a p-value')

diff_xy = hybrid_ROIs_folds-balanced_reg_test
mean_diff_xy = np.mean(diff_xy)
diffFromMean = diff_xy-mean_diff_xy
squared_diffFromMean = np.square(diffFromMean)
sum_squared_diffFromMean = np.sum(squared_diffFromMean)

#Calculate t-statistic
denominator = np.sqrt(sum_squared_diffFromMean/apparent_dof)
numerator = mean_diff_xy*np.sqrt(effective_dof+1)
t_val = numerator/denominator

p_val = 2*(1-t.cdf(np.abs(t_val),effective_dof))

#print('p-value for scaled dof method: ', p_val)

#save the result for the comparison between Hybrid ROIs model and Logistic regression
df_correction_ss.loc[row,'Hybrid ROIs ResNet vs <Model>'] = 'Logistic regression'
df_correction_ss.loc[row,'Values'] = 'bal_acc'
df_correction_ss.loc[row,'t-stat'] = t_val
df_correction_ss.loc[row,'p-value'] = p_val

row += 1
               

#Loop through the rest of the files in the directory performing the same paired t-test as above and save the results to 
#the spreadsheet
for file in files:

    parts = file.split('.')
        
    
    test_xl = pd.read_excel(path+file,sheet_name='Sheet1')
    test_xl.head()
    
    folds = test_xl['val group'].copy(deep=True).values
    folds = folds[0:80]
    #print(folds)


    model_accs = test_xl['bal_acc'].copy(deep=True).values
    model_accs = ResNet_accs[0:80]
    #print(ResNet_accs)

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

    mean_fold1 = np.mean(fold_accs1)
    mean_fold2 = np.mean(fold_accs2)
    mean_fold3 = np.mean(fold_accs3)
    mean_fold4 = np.mean(fold_accs4)

    #print(mean_fold1,mean_fold2,mean_fold3,mean_fold4)

    #Average balanced accuracies for the respective model saved into one array
    model_folds = np.array([mean_fold1,mean_fold2,mean_fold3,mean_fold4])

    #Paired t-test with degrees of freedom correction
    down_scale = 0.45
    N = len(ResNet_folds)
    apparent_dof = N-1
    effective_dof = apparent_dof*down_scale

    if effective_dof<=1:
        print('ERROR: Effective degrees of freedom are too small to calculate a p-value')

    diff_xy = hybrid_ROIs_folds-ResNet_folds
    mean_diff_xy = np.mean(diff_xy)
    diffFromMean = diff_xy-mean_diff_xy
    squared_diffFromMean = np.square(diffFromMean)
    sum_squared_diffFromMean = np.sum(squared_diffFromMean)

    #Calculate t-statistic
    denominator = np.sqrt(sum_squared_diffFromMean/apparent_dof)
    numerator = mean_diff_xy*np.sqrt(effective_dof+1)
    t_val = numerator/denominator

    p_val = 2*(1-t.cdf(np.abs(t_val),effective_dof))

    #print('p-value for scaled dof method: ', p_val)


    df_correction_ss.loc[row,'Hybrid ROIs ResNet vs <Model>'] = names_ss[parts[0]]
    df_correction_ss.loc[row,'Values'] = column
    df_correction_ss.loc[row,'t-stat'] = t_val
    df_correction_ss.loc[row,'p-value'] = p_val

    row+=1
        
df_correction_ss.to_excel(path+'hybrid_ROIs_vs_all_2tails.xlsx')

