# Outstanding - Calculate_Perturbations() does not allow for categorical multi-class
#                                       create missing_log_df function
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader
from math import log10, floor, log, exp
from sympy import symbols, solve, simplify
from skimage.segmentation import mark_boundaries
import copy
import cv2
import io
import re
from PIL import Image
import torch
from sklearn.linear_model import SGDClassifier

import CLEAR_settings, CLEAR_image


class CLEARPerturbation(object):
    # Contains features specific to a particular b-perturbation
    def __init__(self):
        self.wTx = None
        self.nncomp_idx = None
        self.target_feature = None
        self.obs = None
        self.newnn_class = None
        self.raw_weights = None
        self.raw_eqn = None
        self.raw_data = None
        self.adj_raw_data = None
        self.target_prob = None

def getSufficientCauses(wPerturb, explainer):
    #This is relative to the contrast image i.e. a completely infilled image where all segments = 0
    sufficient_causes = []
    intercept_negWeights = wPerturb.intercept
    # for i in wPerturb.raw_weights:
    #     if i <0:
    #         intercept_negWeights += i
    for idx, val in enumerate(wPerturb.raw_weights):
        #From the logistics equation ln(y/(1-y)) = wTx
        if (val + intercept_negWeights) > np.log(CLEAR_settings.sufficiency_threshold/(1-CLEAR_settings.sufficiency_threshold)):
            s= int(wPerturb.raw_eqn[idx][-2:])
            pred = Check_Sufficient_Cause(explainer,[s])
            sufficient_causes.append([wPerturb.raw_eqn[idx],format(pred[0][1], '.3f')])

    if len(sufficient_causes) == 0 and CLEAR_settings.max_sufficient_causes ==2:
        for j in range (0,len(wPerturb.raw_weights-1)):
            for k in range (j+1,len(wPerturb.raw_weights) ):
                if intercept_negWeights +  wPerturb.raw_weights[j] + wPerturb.raw_weights[k] > \
                        np.log(CLEAR_settings.sufficiency_threshold/(1-CLEAR_settings.sufficiency_threshold)):
                    s = int(wPerturb.raw_eqn[j][-2:])
                    t = int(wPerturb.raw_eqn[k][-2:])
                    pred= Check_Sufficient_Cause(explainer, [s,t])
                    sufficient_causes.append([wPerturb.raw_eqn[j],wPerturb.raw_eqn[k],format(pred[0][1], '.3f')])

    return(sufficient_causes)

def Check_Sufficient_Cause(explainer, sufficient_causes):
        row = np.zeros(len(explainer.feature_list))
        for i in sufficient_causes:
            row[i] = 1
        infilled_img = CLEAR_image.Create_infill(row, explainer.segments, explainer.image, explainer.GAN_array)
        img = infilled_img[np.newaxis,...]
        preds= CLEAR_image.Get_image_predictions(explainer.model, img)
        return preds





def Calculate_Perturbations(explainer, results_df):
    """ b-perturbations are now calculated and stored
        in the nncomp_df dataframe. If CLEAR calculates a b-perturbation
        that is infeasible, then the details of the b-perturbation
        are stated in the missing_log_df dataframe. CLEAR will classify
        a b-perturbation as being infeasible if it is outside 'the feasibility
        range' it calculates for each feature.
    """
    print("\n Calculating b-counterfactuals \n")
    nncomp_df = pd.DataFrame(columns=['observation', 'feature', 'orgFeatValue', 'orgAiProb',
                                      'actPerturbedFeatValue', 'AiProbWithActPerturbation', 'estPerturbedFeatValue',
                                      'errorPerturbation', 'regProbWithActPerturbation',
                                      'errorRegProbActPerturb', 'orgClass','sufficient'])
    wPerturb = CLEARPerturbation()
    wPerturb.nncomp_idx = 1
    missing_log_df = pd.DataFrame(columns=['observation', 'feature', 'reason', 'perturbation'])
    i=0
    s1 = pd.Series(results_df.local_data[i], explainer.feature_list)
    s2 = pd.DataFrame(columns=explainer.feature_list)
    s2 = s2.append(s1, ignore_index=True)
    x = symbols('x')
    if  results_df.loc[i, 'features'][0] == '1':
        results_df.loc[i, 'features'].remove('1')
        temp=results_df.loc[i, 'weights'].tolist()
        temp.pop(0)
        results_df.loc[i, 'weights'] = np.array(temp)
    wPerturb.raw_eqn = results_df.loc[i, 'features'].copy()
    wPerturb.raw_weights = results_df.loc[i, 'weights']
    wPerturb.raw_data = results_df.loc[i, 'local_data'].tolist()
    wPerturb.intercept = results_df.loc[i, 'intercept']
    results_df['sufficient']= [getSufficientCauses(wPerturb,explainer)]
    counterfactuals_processed = 0
    for counterf_index in range(0, explainer.counterf_rows_df.shape[0]):
        counterfactuals_processed += 1
        wPerturb.target_feature_weight = 0
        wPerturb.target_feature = explainer.counterf_rows_df.loc[counterf_index,'feature']
        # target_feature = "_".join(wPerturb.target_feature) if len(wPerturb.target_feature)>1 else wPerturb.target_feature
        # set target probability for b-perturbation
        wPerturb.target_prob = CLEAR_settings.binary_decision_boundary
        # establish if all features are in equation
        #if not (target_feature in wPerturb.raw_eqn):
        counter_in_eq = all(s in wPerturb.raw_eqn for s in wPerturb.target_feature)
        if not counter_in_eq:
            interactions_check = "_".join(wPerturb.target_feature) if len(wPerturb.target_feature) > 1 else wPerturb.target_feature
            counter_in_eq = interactions_check in wPerturb.raw_eqn
        if not counter_in_eq:
            if missing_log_df.empty:
                idx = 0
            else:
                idx = missing_log_df.index.max() + 1
            missing_log_df.loc[idx, 'observation'] = i
            missing_log_df.loc[idx, 'feature'] = wPerturb.target_feature
            missing_log_df.loc[idx, 'reason'] = 'not in raw equation'
            continue

        # Create equation string
        obsData_df = pd.DataFrame(columns=explainer.feature_list)
        obsData_df.loc[0] = results_df.loc[i, 'local_data']
        str_eqn, wPerturb.target_feature_weight = generateString(explainer, results_df, i, wPerturb)
        str_eqn = str_eqn.replace('x', '0')
        wPerturb.wTx = simplify(str_eqn)
        nncomp_df = catUpdateNncomp(explainer, nncomp_df, wPerturb, counterf_index, results_df)

    nncomp_df.observation = nncomp_df.observation.astype(int)
    nncomp_df.reset_index(inplace=True, drop=True)

    """
    Determines the actual values of the AI decision boundary for numeric features. This will then be used 
    for determining the fidelity errors of the CLEAR perturbations.
    """
    return nncomp_df, missing_log_df


def catUpdateNncomp(explainer ,nncomp_df, wPerturb, counterf_index, results_df):
    AiProbWithActPerturbation = explainer.counterf_rows_df.loc[counterf_index, 'prediction']
    wPerturb.nncomp_idx += 1
    nncomp_df.loc[wPerturb.nncomp_idx, 'observation'] = 0
    nncomp_df.loc[wPerturb.nncomp_idx, 'feature'] = wPerturb.target_feature
    nncomp_df.loc[wPerturb.nncomp_idx, 'AiProbWithActPerturbation'] = np.float64(AiProbWithActPerturbation)
    nncomp_df.loc[wPerturb.nncomp_idx, 'orgAiProb'] = results_df.loc[0, 'nn_forecast']
    nncomp_df.loc[wPerturb.nncomp_idx, 'orgClass'] = results_df.loc[
        0, 'regression_class']  # needs correcting not sure if regression class needs reporting
    nncomp_df.loc[wPerturb.nncomp_idx, 'orgFeatValue'] = wPerturb.target_feature
    if explainer.data_type == 'image':
        nncomp_df.loc[wPerturb.nncomp_idx, 'actPerturbedFeatValue'] = 'infilled'
    else:
        nncomp_df.loc[wPerturb.nncomp_idx, 'actPerturbedFeatValue'] = explainer.counterf_rows_df.loc[counterf_index, 'feature']
    if CLEAR_settings.regression_type == 'multiple':
        regProbWithActPerturbation = wPerturb.wTx
    else:
        regProbWithActPerturbation = 1 / (1 + exp(-wPerturb.wTx))
    nncomp_df.loc[wPerturb.nncomp_idx, 'regProbWithActPerturbation'] = np.float64(regProbWithActPerturbation)
    nncomp_df.loc[wPerturb.nncomp_idx, 'errorRegProbActPerturb'] = \
        round(abs(regProbWithActPerturbation - AiProbWithActPerturbation), 2)
    return (nncomp_df)




def generateString(explainer, results_df, observation, wPerturb):
    # For categorical target features str_eqn is used to calculate the c-counterfactuals
    raw_data = wPerturb.raw_data
    str_eqn = '+' + str(results_df.loc[observation, 'intercept'])

    for raw_feature in wPerturb.raw_eqn:
        if raw_feature == '1':
            pass
        elif raw_feature in wPerturb.target_feature:
            str_eqn += "+" + str(wPerturb.raw_weights[wPerturb.raw_eqn.index(raw_feature)]) + "*x"
            wPerturb.target_feature_weight = wPerturb.raw_weights[wPerturb.raw_eqn.index(raw_feature)]
        elif raw_feature in explainer.feature_list:
            new_term = raw_data[explainer.feature_list.index(raw_feature)] * wPerturb.raw_weights[
                wPerturb.raw_eqn.index(raw_feature)]
            str_eqn += "+ " + str(new_term)
        elif raw_feature == str(wPerturb.target_feature) + "_sqrd":
            str_eqn += "+" + str(wPerturb.raw_weights[wPerturb.raw_eqn.index(raw_feature)]) + "*x**2"
        elif raw_feature.endswith('_sqrd'):
            new_term = raw_feature.replace('_sqrd', '')
            new_term = (raw_data[explainer.feature_list.index(new_term)] ** 2) * wPerturb.raw_weights[
                wPerturb.raw_eqn.index(raw_feature)]
            str_eqn += "+ " + str(new_term)
        elif '_' in raw_feature:
            interaction_terms = raw_feature.split('_')
            if interaction_terms[0] in wPerturb.target_feature:
                new_term = str(raw_data[explainer.feature_list.index(interaction_terms[1])] \
                               * wPerturb.raw_weights[wPerturb.raw_eqn.index(raw_feature)]) + "*x"
            elif interaction_terms[1] in wPerturb.target_feature:
                new_term = str(raw_data[explainer.feature_list.index(interaction_terms[0])] \
                               * wPerturb.raw_weights[wPerturb.raw_eqn.index(raw_feature)]) + "*x"
            else:
                new_term = str(raw_data[explainer.feature_list.index(interaction_terms[0])]
                               * raw_data[explainer.feature_list.index(interaction_terms[1])]
                               * wPerturb.raw_weights[wPerturb.raw_eqn.index(raw_feature)])
            str_eqn += "+ " + new_term
        else:
            print("error in processing equation string")
        pass
    return str_eqn, wPerturb.target_feature_weight


def Summary_stats(nncomp_df, missing_log_df):
    """ Create summary statistics and frequency histogram
    """
    if nncomp_df.empty:
        print('no data for plot')
        return
    less_target_sd = 0
    temp_df = nncomp_df.copy(deep=True)
    temp_df = temp_df[~temp_df.errorPerturbation.isna()]
    if temp_df['errorPerturbation'].count() != 0:
        less_target_sd = temp_df[temp_df.errorPerturbation <= 0.25].errorPerturbation.count()
        x = temp_df['errorPerturbation']
        x = x[~x.isna()]
        ax = x.plot.hist(grid=True, bins=20, rwidth=0.9)
        plt.title(
            'perturbations = ' + str(temp_df['errorPerturbation'].count()) + '  Freq Counts <= 0.25 sd = ' + str(
                less_target_sd)
            + '\n' + 'regression = ' + CLEAR_settings.regression_type + ', score = ' + CLEAR_settings.score_type
            + ', sample = ' + str(CLEAR_settings.num_samples)
            + '\n' + 'max_predictors = ' + str(CLEAR_settings.max_predictors)
            + ', regression_sample_size = ' + str(CLEAR_settings.regression_sample_size))
        plt.xlabel('Standard Deviations')
        fig = ax.get_figure()
        fig.savefig(CLEAR_settings.CLEAR_path + 'hist' + datetime.now().strftime("%Y%m%d-%H%M") + '.png',
                    bbox_inches="tight")
    else:
        print('no numeric feature data for histogram')
        temp_df = nncomp_df.copy(deep=True)
    # x=np.array(nncomp_df['errorPerturbation'])

    filename1 = CLEAR_settings.CLEAR_path + 'wPerturb_' + datetime.now().strftime("%Y%m%d-%H%M") + '.csv'
    nncomp_df.to_csv(filename1)
    filename2 = CLEAR_settings.CLEAR_path + 'missing_' + datetime.now().strftime("%Y%m%d-%H%M") + '.csv'
    missing_log_df.to_csv(filename2)
    output = [CLEAR_settings.sample_model, less_target_sd]
    filename3 = 'batch.csv'
    try:
        with open(CLEAR_settings.CLEAR_path + filename3, 'a') as file1:
            writes = csv.writer(file1, delimiter=',', skipinitialspace=True)
            writes.writerow(output)
        file1.close()
    except:
        pass
    return


def Single_prediction_report(results_df, nncomp_df, single_regress, explainer):
    # rectangular_segments = np.genfromtxt("C:/Users/adamp/Dropbox/CLEAR MRI - desk/632_by_760_segments.csv", delimiter=',')
    # rectangular_segments = np.genfromtxt("C:/Users/adamp/Dropbox/CLEAR MRI - desk/Temp 632_by_760_segments.csv", delimiter=',')
    rectangular_segments = np.genfromtxt(CLEAR_settings.images_path + CLEAR_settings.reactangle_segments ,
                                         delimiter=',')
    if CLEAR_settings.case_study == 'MRI':
        if CLEAR_settings.image_segment_type in ('single segment','single slice'):
            seg_names_df = pd.read_csv(CLEAR_settings.CLEAR_path + 'Single_segment_names.csv')
        else:
            seg_names_df = pd.read_csv(CLEAR_settings.CLEAR_path + 'Brain_Atlas_Segments_with_3_NatLab.csv') #'Brain_Atlas_Segments_rest_of_brain.csv')

    if nncomp_df.empty:
        print('no counterfactuals found')
    if len(explainer.class_labels)==2:
        explanandum= explainer.class_labels[1]



    def round_sig(x, sig=2):
        if type(x) == np.ndarray:
            x = x[0]
        if x == 0:
            y= 0
        else:
            y= round(x, sig - int(floor(log10(abs(x)))) - 1)
        return y

    results_row = results_df.index.values[0]
    j = results_df.index.values[0]
    org_features = []
    if CLEAR_settings.regression_type == 'multiple':
        regression_formula = 'prediction = ' + str(round_sig(results_df.intercept[results_row]))
    else:
        regression_formula = '<font size = "4.5">prediction =  [ 1 + e<sup><b>-w<sup>T</sup>x</sup></b> ]<sup> -1</sup></font size><br><br>' \
                             + '<font size = "4.5"><b><i>w</i></b><sup>T</sup><b><i>x</font size></i></b> =  ' + str(
            round_sig(results_df.intercept[results_row]))

    for i in range(len(results_df.features[results_row])):
        if results_df.features[results_row][i] == '1':
            continue
        if ("_" in results_df.features[results_row][i]) and ("_sqrd" not in results_df.features[results_row][i]):
            results_df.features[results_row][i] = "(" + results_df.features[results_row][i] + ")"
        Dd_idx = [m.start() for m in re.finditer('Dd', results_df.features[results_row][i])]
        for t in Dd_idx:
            # t = results_df.features[results_row][i].find("Dd")
            if (results_df.features[results_row][i][t + 5:t+8]).isnumeric():
                seg_number = int(results_df.features[results_row][i][t + 5:t+8])
            elif (results_df.features[results_row][i][t + 5:t+7]).isnumeric():
                 seg_number = int(results_df.features[results_row][i][t + 5:t + 7])
            else:
                print('report error - converting labels')

            org_features.append('Seg'+ str(seg_number))
            if t == Dd_idx[0]:
                final_label = seg_names_df[seg_names_df.Seg_num == seg_number].Seg_name.iloc[0]
            else:
                final_label = final_label + "_" + seg_names_df[seg_names_df.Seg_num == seg_number].Seg_name.iloc[0]
        results_df.features[results_row][i] = final_label

        if results_df.weights[results_row][i] < 0:
            regression_formula = regression_formula + ' - ' + str(-1 * round_sig(results_df.weights[results_row][i])) + \
                                 ' ' + results_df.features[results_row][i]
        else:
            regression_formula = regression_formula + ' + ' + str(round_sig(results_df.weights[results_row][i])) + \
                                 ' ' + results_df.features[results_row][i]
    regression_formula = regression_formula.replace("_sqrd", "<sup>2</sup>")
    regression_formula = regression_formula.replace("_", "*")
    report_AI_prediction = str(round_sig(results_df.nn_forecast[results_row]))
    if CLEAR_settings.score_type == 'adjR':
        regression_score_type = "Adjusted R-Squared"
    else:
        regression_score_type = CLEAR_settings.score_type
    # Creates input data table for report. Gets rid of dummy variables equal to zero
    temp2_df = pd.DataFrame(columns=['Feature', 'Input Value'])
    temp = [col for col in single_regress.data_row.columns \
            if not ((single_regress.data_row.loc[0, col] == 0) and (col in explainer.cat_features))]
    input_data = single_regress.data_row.loc[0, temp]
    k = 0
    for col in input_data.index:
        if col in explainer.cat_features:
            if explainer.data_type == 'image':
                temp2_df.loc[k, 'Feature']=col[5:]
            else:
                temp2_df.loc[k, 'Feature'] = col.replace("Dd", "=")
            temp2_df.loc[k, 'Input Value'] = "1"
        else:
            temp2_df.loc[k, 'Feature'] = col
            temp2_df.loc[k, 'Input Value'] = str(round(input_data.iloc[k], 2))
        k += 1
    inputData_df = temp2_df.copy(deep=True)
    inputData_df.set_index('Feature', inplace=True)
    inputData_df = inputData_df.transpose().copy(deep=True)

    #create counterfactual tables
    temp_df = nncomp_df.copy(deep=True)
    for index, rows in temp_df.iterrows():
        Dd_idx = [x.index('Dd') for x in temp_df.loc[index,'feature']]
        for cnt, t in enumerate(Dd_idx):
            # t = results_df.features[results_row][i].find("Dd")
            if (temp_df.loc[index,'feature'][cnt][t + 5:t + 8]).isnumeric():
                seg_number = int(temp_df.loc[index,'feature'][cnt][t + 5:t + 8])
            elif (temp_df.loc[index,'feature'][cnt][t + 5:t + 7]).isnumeric():
                seg_number = int(temp_df.loc[index,'feature'][cnt][t + 5:t + 7])
            if cnt==0:
                final_label = seg_names_df[seg_names_df.Seg_num == seg_number].Seg_name.iloc[0]
            else:
                final_label = final_label + "_" + seg_names_df[seg_names_df.Seg_num == seg_number].Seg_name.iloc[0]
        temp_df.at[index,'feature'] = final_label
        temp_df.loc[index,'actPerturbedFeatValue'] = 'infilled'

    c_counter_df = temp_df[['feature', 'AiProbWithActPerturbation',
                            'regProbWithActPerturbation', 'errorRegProbActPerturb']].copy()
    c_counter_df.rename(columns={"AiProbWithActPerturbation": "AI using c-counterfactual value",
                                 "regProbWithActPerturbation": "regression forecast using c-counterfactual",
                                 "errorRegProbActPerturb": "fidelity error"},
                        inplace=True)

    # sorted unique feature list for the 'select features' checkbox
    feature_box = results_df.features[results_row]
    feature_box = ",".join(feature_box).replace('(', '').replace(')', '').replace('_', ',').split(",")
    for x in ['sqrd', '1']:
        if x in feature_box:
            feature_box.remove(x)
    feature_box = list(set(feature_box))
    #repeat for original features
    # results_df.weights needs pre-processing prior to sending to HTML
    weights = results_df.weights.values[0]
    spreadsheet_data = results_df.spreadsheet_data.values[0]
    if len(weights) == len(spreadsheet_data) + 1:
        weights = np.delete(weights, [0])
    weights = weights.tolist()

    # calculate feature importance
    feat_importance_df = pd.DataFrame(columns=feature_box)
    for y in feature_box:
        temp = 0
        cnt = 0
        for z in results_df.features[results_row]:
            if y in z:
                if y == z:
                    temp += results_df.weights[results_row][cnt] * results_df.spreadsheet_data[results_row][cnt]
                elif '_sqrd' in z:
                    temp += results_df.weights[results_row][cnt] * (results_df.spreadsheet_data[results_row][cnt])
                elif z.count('_') == 1:
                    temp += (results_df.weights[results_row][cnt] * results_df.spreadsheet_data[results_row][cnt]) / 2
                elif z.count('_') == 2:
                    temp += (results_df.weights[results_row][cnt] * results_df.spreadsheet_data[results_row][cnt]) / 3
                elif z.count('_') == 3:
                    temp += (results_df.weights[results_row][cnt] * results_df.spreadsheet_data[results_row][cnt]) / 4
                else:
                    print('error in reporting feature importance')
            cnt += 1
        feat_importance_df.loc[0, y] = temp
    # normalise by largest absolute value
    t = feat_importance_df.iloc[0, :].abs()
    top_seg = pd.to_numeric(t).idxmax()
    # create feature importance bar chart
    max_display = min(feat_importance_df.shape[1],12) #select top 12 features
    counterfactual_segs = []  #add any missing counterfactuuals
    for index, rows in c_counter_df.iterrows():
        for index2 in range(0,len(c_counter_df.loc[index,'feature'])):
            temp = c_counter_df.loc[index, 'feature'][index2]
            if temp not in counterfactual_segs:
                counterfactual_segs.append(temp)
    counterfactual_segs= list(set(counterfactual_segs))
    for index, rows in c_counter_df.iterrows():
        feature = c_counter_df.at[index, 'feature']
        feature = str(feature).replace('[', "")
        feature = str(feature).replace(']', "")
        c_counter_df.at[index, 'feature'] = feature
    #counterfactual_segs = [np.int32(i[-2:]) for i in counterfactual_segs]
    #so need index of counterfactual_segs in feature_importance_df
    #temp = feat_importance_df.columns.to_list()
    # counterfactual_idx= [temp.index(i) for i in counterfactual_segs]
    # counterfactual_idx = np.array(counterfactual_idx)
    #display_features = np.sort(feat_importance_df.abs().values.argsort(1)[0][-max_display:])
    # display_features = np.union1d(display_features,counterfactual_idx)
    display_features = feat_importance_df.abs().values.argsort()[0][::-1][:max_display]
    ax = feat_importance_df.iloc[:, display_features].plot.barh(width=1)
    ax.patches[0].set_color([0.08,0.05,1])  # sets colour for 'background' segment
    ax.legend(fontsize=12)
    ax.invert_yaxis()
    ax.margins(y=0)
    ax.yaxis.set_visible(False)
    leg = ax.legend()
    patches = leg.get_patches()
    bar_colours = [x.get_facecolor() for x in patches]
    colour_report= False
    if colour_report == True:
        seg2bar = bar_colours[2]
        seg4bar = bar_colours[4]
        bar_colours[4]= seg2bar
        bar_colours[2] = seg4bar
    fig = ax.get_figure()
    fig.set_figheight(5)
    fig.set_figheight(6.5)
    fig.tight_layout()
    fig.savefig('Feature_plot.png', bbox_inches="tight")

    #Create regression scatterplot
    fig = plt.figure()
    plt.scatter(single_regress.neighbour_df.loc[:, 'prediction'], single_regress.after_center_option, c='green',
                s=10)
    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), c="red", linestyle='-')
    plt.xlabel('Target AI System')
    if CLEAR_settings.regression_type == 'logistic':
        plt.ylabel('CLEAR Logistics Regression')
    elif CLEAR_settings.regression_type == 'multiple':
        plt.ylabel('CLEAR Multiple Regression')
    else:
        plt.ylabel('CLEAR Polynomial Regression')
    fig.savefig('CLEAR_plot.png', bbox_inches="tight")
    pd.set_option('colheader_justify', 'left', 'precision', 2)
    env = Environment(loader=FileSystemLoader('.'))

    #create segmented image
    if explainer.data_type == 'image':
        colour_seg = np.ones(explainer.rectangular_image.shape)
        for m in display_features:
            bar_idx = np.where(m == display_features)[0][0]
            seg_num = seg_names_df[seg_names_df['Seg_name'] == feat_importance_df.columns[m]].Seg_num.values[0]
            for n in range(3):
                colour_seg[:, :, n][rectangular_segments ==seg_num ] = bar_colours[bar_idx][n]
            bar_idx += 1
        im1=explainer.rectangular_image
        im2=cv2.addWeighted(im1, 0.2, colour_seg, 0.8, 0)
        plt.imshow(im2)
        plt.show()
        if CLEAR_settings.debug_mode is True:
            plt.imshow(im2)
            plt.show()

        # t = np.genfromtxt("C:/Users/adamp/Dropbox/CLEAR MRI - desk/Temp 632_by_760_segments.csv", delimiter=',')
        # t_image = Image.fromarray(t).convert('RGB')
        # t = np.array(t_image)
        # t = t*150
        # t[:, :, 0] = t[:, :,0 ]*1.2
        # t[:, :, 1] = t[:, :, 1]*0.8
        # t[:, :, 2] = t[:, :, 2]*1.6
        # t = (t - t.min()) / (t.max() - t.min())
        # plt.imshow(t)
        # plt.show()
        # im1=explainer.rectangular_image
        # im1 =(im1- im1.min())/(im1.max()-im1.min()) #normalises between 0 and 1
        # im2=cv2.addWeighted(im1, 0.5, t, 0.5, 0)
        # plt.imshow(im2)
        # plt.show()
        # plt.imsave('all_brain_segs.png', t, dpi=3000)

        # t = np.genfromtxt("C:/Users/adamp/Dropbox/CLEAR MRI - desk/Temp 632_by_760_segments.csv", delimiter=',')
        # q = (t==0).astype(int)  # hence segments are false i.e 0
        # from skimage import measure
        # high_threshold_blobs = measure.label(q,connectivity=1) #this treats 0 as background hence all segments are lumped together in blob
        # unique, counts = np.unique(high_threshold_blobs, return_counts=True)
        # plt.imshow(high_threshold_blobs)
        # plt.show()
        #
        #
        # mask= np.zeros(high_threshold_blobs.shape)
        # mask[high_threshold_blobs>1] = 1
        # plt.imshow(mask)
        # plt.show()
        #
        #
        # mask= np.zeros(high_threshold_blobs.shape)
        # mask[high_threshold_blobs==1] = 1
        # plt.imshow(mask)
        # plt.show()
        #
        # mask= np.zeros(high_threshold_blobs.shape)
        # mask[high_threshold_blobs==0] = 1
        # plt.imshow(mask)
        # plt.show()
        #
        # mask= np.zeros(high_threshold_blobs.shape)
        # mask[high_threshold_blobs==0] = 1
        # plt.imshow(mask)
        # plt.show()





        plt.imsave('key_segments.png', im2, dpi=3000)


    #write to HTML
        if  len(results_df.loc[0,'sufficient'])==0:
            sufficient_causes_out = []
        else:
            sufficient_causes_out = [x[-5:] for x in results_df.loc[0,'sufficient']]
        template = env.get_template("CLEAR_Image_report.html")
        template_vars = {"title": "CLEAR Statistics",
                         "input_data_table": inputData_df.to_html(index=False, classes='mystyle'),
                         "dataset_name":CLEAR_settings.image_model_name ,
                         "explanadum": explanandum,
                         "observation_number": explainer.observation_num,
                         "regression_formula": regression_formula,
                         "prediction_score": round_sig(results_df.Reg_Score[results_row]),
                         "regression_score_type": regression_score_type,
                         "regression_type": CLEAR_settings.regression_type,
                         "AI_prediction": report_AI_prediction,
                         "cat_counterfactual_table": c_counter_df.to_html(index=False, classes='mystyle'),
                         "feature_list": feature_box,
                         "spreadsheet_data": spreadsheet_data,
                         "weights": weights,
                         "intercept": results_df.intercept.values[0],
                         "sufficient":  sufficient_causes_out
                         }
        with open('CLEAR_Image_report_full.html', 'w') as fh:
            fh.write(template.render(template_vars))
    batch_Xrays = False
    if batch_Xrays is True:
        if c_counter_df.empty is True:
            c_counter_df.loc[0,'feature']= 'None'
        c_counter_df['threshold'] = CLEAR_settings.image_high_segment_threshold
        c_counter_df['regression score']=round_sig(results_df.Reg_Score[results_row])
        c_counter_df['Xray_ID'] = explainer.batch
        c_counter_df['top_seg'] = top_seg
        c_counter_df['Seg00_too_large'] = False
        c_counter_df['poor_data']= explainer.poor_data
        if 'Seg00' in feature_box:
            if feat_importance_df.loc[0,'Seg00']/feat_importance_df.loc[0,top_seg]>0.5:  #was 0.25
               c_counter_df['Seg00_too_large'] = True
        # c_counter_df.to_pickle(CLEAR_settings.CLEAR_path+'c_counter_df.pkl')

        plt.close('all')










    return(c_counter_df)

