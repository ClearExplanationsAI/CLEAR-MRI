

"""
Functions for CLEAR to create local regressions

Outstanding work - perform_regression() accuracy & decision boundaries for multi-class
"""

from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm
import copy
from sklearn.preprocessing import PolynomialFeatures
import CLEAR_settings
import CLEAR_image
from scipy.spatial.distance import cdist
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.stats.outliers_influence import variance_inflation_factor
import itertools
import statsmodels.formula.api as smf

""" specify input parameters"""

kernel_type = 'Euclidean'  # sets distance measure for the neighbourhood algorithms


class CLEARSingleRegression(object):
    # Contains features specific to a particular regression
    def __init__(self, data_row):
        self.features = 0
        self.prediction_score = 0
        self.nn_forecast = 0
        self.local_prob = 0
        self.regression_class = 0
        self.spreadsheet_data = 0
        self.after_center_option = 0
        self.data_row = 0
        self.accuracy = 0
        self.intercept = 0
        self.coeffs = 0
        self.observation_num = 0
        self.data_row = data_row
        self.additional_weighting = 0
        self.local_df = 0
        self.neighbour_df = 0


def Run_Regressions(X_test_sample, explainer):
    get_counterfactuals(explainer, X_test_sample)
    results_df = pd.DataFrame(columns=['image','Reg_Score', 'intercept', 'features', 'weights',
                                       'nn_forecast', 'reg_prob', 'regression_class',
                                       'spreadsheet_data', 'local_data', 'accuracy'])

    print('Performing step-wise regressions \n')
    data_row = pd.DataFrame(columns=explainer.feature_list)
    data_row = data_row.append(X_test_sample.iloc[0], ignore_index=True)
    data_row.fillna(0, inplace=True)
    explainer, single_regress = explain_data_point(explainer, data_row)
    print('Processing observation')
    results_df.at[0,'image']= explainer.batch
    results_df.at[0, 'features'] = single_regress.features
    results_df.loc[0, 'Reg_Score'] = single_regress.prediction_score
    results_df.loc[0, 'nn_forecast'] = single_regress.nn_forecast
    results_df.loc[0, 'reg_prob'] = single_regress.local_prob
    results_df.loc[0, 'regression_class'] = single_regress.regression_class
    results_df.at[0, 'spreadsheet_data'] = single_regress.spreadsheet_data
    results_df.at[0, 'local_data'] = data_row.values[0]
    results_df.loc[0, 'accuracy'] = single_regress.accuracy
    results_df.loc[0, 'intercept'] = single_regress.intercept
    results_df.at[0, 'weights'] = single_regress.coeffs

    return (results_df, explainer, single_regress)


def explain_data_point(explainer, data_row):
    single_regress = CLEARSingleRegression(data_row)
    # This is for centering when using logistic regression
    if CLEAR_settings.regression_type == 'logistic':
        if CLEAR_settings.centering is True:
            single_regress.additional_weighting = np.int(CLEAR_settings.num_samples/2) # was 0 and 19 obs added each time but trialling with just adding 39
        else:
            single_regress.additional_weighting = 2
    single_regress.local_df = explainer.master_df.copy(deep=True)
    y = forecast_data_row(explainer, data_row)
    single_regress.local_df.iloc[0, 0:explainer.num_features] = data_row.iloc[0, 0:explainer.num_features]
    single_regress.local_df.loc[0, 'prediction'] = y
    single_regress.nn_forecast = y
    create_neighbourhood(single_regress)
    temp_df=explainer.counterf_rows_df.copy(deep=True)
    if (CLEAR_settings.apply_counterfactual_weights) and (temp_df.empty is False):
        #increase weighting for counterfactuals involving a single segment if there are also other counterfactuals
        counter_list = [len(i) for i in (temp_df.loc[:,'feature'])]
        if max(counter_list)>1 and min(counter_list)==1:
            for w in range(temp_df.shape[0]):
                if len(temp_df.loc[w,'feature'])==1:
                    temp_row = temp_df.loc[w,:].copy(deep=True)
                    adjust_neighbourhood(single_regress, temp_row, 2*np.int(CLEAR_settings.counterfactual_weight * CLEAR_settings.num_samples))
        temp_df=temp_df.drop(['observation', 'feature'], axis=1)

        if 0 < temp_df.shape[0] <10:
            # adjust_neighbourhood(single_regress, single_regress.neighbour_df.tail(temp_df.shape[0]),
            #                      np.int(CLEAR_settings.counterfactual_weight * CLEAR_settings.num_samples))
            adjust_neighbourhood(single_regress, temp_df,
                                 np.int(CLEAR_settings.counterfactual_weight * CLEAR_settings.num_samples))
    # if no centering == False and regression type - logistic then initially add 19 rows of the observation
    # that is to be explained to the neighbourhood dataset. This in effect is the same as adding a weighting of 20
    if explainer.data_type == 'image' and CLEAR_settings.image_infill == 'GAN':
        GAN_row = pd.DataFrame(data=np.zeros((1, single_regress.neighbour_df.shape[1])),
                               columns=single_regress.neighbour_df.columns)
        GAN_row.prediction = explainer.GAN_pred
        GAN_row.target_range = 'GAN'
        adjust_neighbourhood(single_regress, GAN_row,  single_regress.additional_weighting)
    if CLEAR_settings.regression_type == 'multiple':
        perform_regression(explainer, single_regress)
    if (CLEAR_settings.regression_type == 'logistic') and (CLEAR_settings.centering is True):
        adjust_neighbourhood(single_regress, single_regress.neighbour_df.iloc[0, :], single_regress.additional_weighting)  #was 39 for tabular data # changed to 500 for images

    if CLEAR_settings.regression_type == 'logistic':
        perform_regression(explainer, single_regress)
        if single_regress.perfect_separation is True:
            return explainer, single_regress
        centering_weight = single_regress.additional_weighting
        while ((centering_weight < CLEAR_settings.num_samples) and
               ((single_regress.regression_class != single_regress.nn_class) or
                (abs(single_regress.local_prob - single_regress.nn_forecast) > 0.02))):
            print('poor local accuracy, weighting for original image increased. Centering weight is ' + str(centering_weight))
            centering_weight += single_regress.additional_weighting
            adjust_neighbourhood(single_regress, single_regress.neighbour_df.iloc[0, :], centering_weight)
            perform_regression(explainer, single_regress)
    # single_regress.neighbour_df.to_pickle('neighbour_df.pkl')
    if CLEAR_settings.generate_regression_files == True:
        filename1 = CLEAR_settings.CLEAR_path + 'local_' + datetime.now().strftime("%Y%m%d-%H%M%S%f") + '.csv'
        single_regress.neighbour_df.to_csv(filename1)
    return explainer, single_regress

def adjust_neighbourhood(single_regress, target_rows, num_copies):
    # add num_copies more observations
    temp = target_rows.copy(deep=True)
    #temp = pd.DataFrame(columns=single_regress.neighbour_df.columns)
    temp['weight']= num_copies
    #temp = temp.append(target_rows, ignore_index=True)
    temp2 = single_regress.neighbour_df.copy(deep=True)
    temp3 = temp2.append(temp, ignore_index=True)
    temp3 = temp3.reset_index(drop=True)
    single_regress.neighbour_df = temp3.copy(deep=True)

    return single_regress

def forecast_data_row(explainer, data_row):
    image= CLEAR_image.Create_infill(data_row,explainer.segments,explainer.image,explainer.GAN_array)
    image = np.expand_dims(image, axis=0)
    y = CLEAR_image.Get_image_predictions(explainer.model, image)
    y = y[0][explainer.top_idx]
    return (y)


def create_neighbourhood(single_regress):
    single_regress.neighbour_df = single_regress.local_df
    single_regress.neighbour_df['target_range'] = ""
    single_regress.neighbour_df['weight'] = 1
    if CLEAR_settings.generate_regression_files == True:
        filename1 = CLEAR_settings.CLEAR_path + 'local_' + str(
            single_regress.observation_num) + '_' + datetime.now().strftime("%Y%m%d-%H%M%S%f") + '.csv'
        single_regress.neighbour_df.to_csv(filename1)

    return single_regress


def neighbourhood_distances(explainer,single_regress):
    x = single_regress.local_df.loc[0, explainer.cat_features].values.reshape(1, -1)
    y = single_regress.local_df.loc[:, explainer.cat_features].values.reshape(
        single_regress.local_df.shape[0], -1)
    z = cdist(x, y, 'jaccard')
    x = single_regress.local_df.loc[0, explainer.numeric_features].values.reshape(1, -1)
    y = single_regress.local_df.loc[:, explainer.numeric_features].values.reshape(
        single_regress.local_df.shape[0], -1)
    w = cdist(x, y, 'euclidean')
    distances = (len(explainer.numeric_features) / explainer.num_features * w + \
                 len(explainer.cat_features) / explainer.num_features * z).ravel()
    return distances

# creates a dataframe listing counterfactuals
def get_counterfactuals(explainer, X_test_sample):
    temp = copy.deepcopy(explainer.feature_list)
    temp.insert(0, 'observation')
    temp.append('prediction')
    explainer.counterf_rows_df = pd.DataFrame(columns=temp)
    i=0
    data_row = pd.DataFrame(columns=explainer.feature_list)
    data_row = data_row.append(X_test_sample.iloc[i], ignore_index=True)
    data_row.fillna(0, inplace=True)
    y = forecast_data_row(explainer, data_row)
    if explainer.cat_features != []:
        temp_df = explainer.catSensit_df[
        (explainer.catSensit_df['observation'] == i) &
        ((explainer.catSensit_df.probability >= CLEAR_settings.binary_decision_boundary)
         != (y >= CLEAR_settings.binary_decision_boundary))
        & (explainer.catSensit_df['feature1'] != 'S00DdSeg00') ].copy(deep=True)
        if not temp_df.empty:
            # remove redundant counterfactuals ie contain a simpler counterfactual
            temp_df['all'] = temp_df['feature1'] + temp_df['feature2'] + temp_df['feature3'] + temp_df['feature4']
            temp_df.sort_values('feature4', ascending=False)
            temp_df.reset_index(inplace=True, drop=True)
            if temp_df.loc[0,'feature4'] == 'nil':
                for nil_column in range(4,1,-1):
                    temp = 'feature'+str(nil_column)
                    temp_df.sort_values(by=[temp], ascending=False)
                    cnt= 0
                    while temp_df.loc[cnt,temp]== 'nil' and cnt <(temp_df.shape[0]-1):
                        if temp_df.loc[cnt,'probability']== 999:
                            cnt +=1
                            continue
                        features_to_chk = ""
                        for t in range(1,nil_column):
                            u = 'feature' + str(t)
                            if temp_df.loc[cnt,u] != 'nil':
                                features_to_chk +=  temp_df.loc[cnt,u]
                        cnt +=1
                        cnt2 = cnt
                        while cnt2 < temp_df.shape[0]:
                            if features_to_chk in temp_df.loc[cnt2,'all']:
                                temp_df.loc[cnt2,'probability'] = 999
                            cnt2 +=1
                temp_df = temp_df[temp_df.probability != 999]
                temp_df.replace('nil',"", inplace = True, regex = True)
                temp_df.reset_index(inplace=True, drop=True)
     #5 is because image counterfactuals have upto 4 features
            for index, row in temp_df.iterrows():
                features = temp_df.iloc[index, 1:5].to_list()
                s1 = X_test_sample.iloc[0].copy(deep=True)
                cnt = 0
                for feature in features:
                    if feature != '':
                        s1.loc[feature] = 0
                        cnt +=1
                s1.loc['prediction'] = temp_df.loc[index,'probability']
                s1['feature'] = features[0:cnt]
                s1['target_range'] = 'counterf'
                s1['distances'] = np.nan
                s1['observation'] = 0
                explainer.counterf_rows_df = explainer.counterf_rows_df.append(s1, ignore_index=True)
    return explainer




def perform_regression(explainer, single_regress):
    # A stepwise regression is performed. The user can specify a set of features
    # (‘selected’) that will be included in the regression. There are several
    # motivations for including this option: (i) it provides a means of ensuring that
    # the features from the dataset are included ‘unaltered’ i.e. not raised to a
    # power or included in an interaction term. This can lead to explanations that
    # are of greater interpretability, though perhaps of lower fidelity. For example,
    # with the IRIS dataset the user can specify that the regression equation should
    # include ['SepalL', 'SepalW', 'PetalL', 'PetalW'] i.e. the four features in the
    # dataset. CLEAR’s stepwise regression will then add further power/interaction
    # terms. (ii) it enables the user to use their domain knowledge to focus CLEAR’s
    # regressions (iii) it reduces running times.

    # In order to ensure that CLEAR’s regressions explain the c-counterfactuals it
    # identifies, CLEAR adds the dummy variables corresponding to each
    # c-counterfactual.

    # A future enhance
    #
    #
    # ment will be to carry out backward regression.
    selected = ['1']
    single_regress.perfect_separation = False
    if CLEAR_settings.include_features is True:
        #add included features, having first checked that they are in the dataset
        selected = addIncludedFeatures(selected, explainer)
    if explainer.cat_features != []:
        # for each categorical, if c-counterfactual exists include categorical for data_row plus for c-counterfactual
        # Also ensure that dummy trap is avoided. Not applicable for Images as only one variable per segment
        dummy_trap = False
        if dummy_trap is True:
            counterfactualDummies = getCounterfactualDummies(explainer, single_regress.nn_forecast, \
                                                             single_regress.data_row, single_regress.observation_num, \
                                                             dummy_trap)
            for j in counterfactualDummies:
                if j not in selected:
                    selected.append(j)
    #add numeric features with counterfactuals
    if explainer.numeric_features !=[]:
        temp_df = explainer.counterf_rows_df[explainer.counterf_rows_df.observation==single_regress.observation_num]
        if temp_df.empty is False:
            temp= temp_df.feature.to_list()
            for k in temp:
                if (k in explainer.numeric_features) and (k not in selected):
                    selected.append(k)
    if CLEAR_settings.image_all_segments == True and explainer.data_type == 'image':
        selected = ['1']
        #this allows for feature maps generating 'missing segments'
        segs= [('S'+str(x)+'DdSeg'+str(x)) if x >9 else ('S0'+str(x)+'DdSeg0'+str(x)) for x in list(np.unique(explainer.segments))]
        selected = selected + segs
    #add all counterfactual features for images
    counter_list = []
    if explainer.data_type == 'image':
        for index, row in explainer.counterf_rows_df.iterrows():
            if (CLEAR_settings.image_counterfactual_interactions is True) and len(explainer.counterf_rows_df.loc[index,'feature'])==2:
                new_term =explainer.counterf_rows_df.loc[index,'feature'][0] +'_' \
                          + explainer.counterf_rows_df.loc[index,'feature'][1]
                counter_list.append(new_term)
            else:
                counter_list = counter_list + explainer.counterf_rows_df.loc[index,'feature']
        if len(counter_list) > 10:
            counter_list = []
        for k in counter_list:
            if (k not in selected) and (k != 'nil'):
                selected.append(k)
    # Create poly_df excluding any categorical features with low sum
    X = single_regress.neighbour_df.iloc[:, 0:explainer.num_features].copy(deep=True)
    X = X.reset_index(drop=True)
    weights = single_regress.neighbour_df.loc[:,'weight'].copy(deep=True)
    weights = weights.values.astype(int)
    temp = [col for col, val in X.sum().iteritems() \
            if ((val <= CLEAR_settings.counterfactual_weight) and (col in explainer.cat_features) and
                col not in selected)]
    X.drop(temp, axis=1, inplace=True)
    temp = [col for col, val in X.sum().iteritems() \
            if ((val == X.shape[0]) and (col in explainer.cat_features) and
                col not in selected)]
    X.drop(temp, axis=1, inplace=True)


    if CLEAR_settings.no_polynomimals_no_complete_set_interactions is True:
        poly_df = X.copy(deep=True)
        if CLEAR_settings.interactions_from_list is True:
            if explainer.top_pred >=CLEAR_settings.binary_decision_boundary:
                interaction_features = ['S01DdSeg01','S04DdSeg04','S07DdSeg07','S08DdSeg08','S11DdSeg11', 'S13DdSeg13','S14DdSeg14' ,'S17DdSeg17','S29DdSeg29','S37DdSeg37' ,'S51DdSeg51','S52DdSeg52','S57DdSeg57','S60DdSeg60','S61DdSeg61', 'S63DdSeg63', 'S65DdSeg65','S71DdSeg71','S81DdSeg81', 'S85DdSeg85','S86DdSeg86','S88DdSeg88','S90DdSeg90','S117DdSeg117']
            else:
            #interaction_features =  ['S04DdSeg04', 'S08DdSeg08','S24DdSeg24','S103DdSeg103','S52DdSeg52','S94DdSeg94','S45DdSeg45','S104DdSeg104','S70DdSeg70','S49DdSeg49','S54DdSeg54','S20DdSeg20']
                interaction_features = ['S01DdSeg01','S11DdSeg11','S13DdSeg13','S29DdSeg29','S51DdSeg51','S57DdSeg57','S63DdSeg63','S71DdSeg71','S73DdSeg73','S75DdSeg75','S81DdSeg81','S85DdSeg85','S117DdSeg117']

            interaction_terms =[x[0] + '_' + x[1] for x in itertools.combinations(interaction_features, 2)]
            interaction_terms2 =[]# [x[0] + '_' + x[1] + '_' + x[2] for x in itertools.combinations(interaction_features, 3)]
            interaction_terms3 =[]#x[0] + '_' + x[1] + '_' + x[2] + '_' + x[3] for x in itertools.combinations(interaction_features, 4)]
            interaction_terms = interaction_terms + interaction_terms2 + interaction_terms3
            for m in interaction_terms:
                temp = m.split('_')
                if len(temp)==2:
                    poly_df[m] = poly_df[temp[0]] * poly_df[temp[1]]
                elif len(temp)==3:
                    poly_df[m] = poly_df[temp[0]] * poly_df[temp[1]]* poly_df[temp[2]]
                elif len(temp) == 4:
                    poly_df[m] = poly_df[temp[0]] * poly_df[temp[1]] * poly_df[temp[2]]* poly_df[temp[3]]
    else:
        if CLEAR_settings.all_interactions is True:
            poly = PolynomialFeatures(interaction_only=True)

        else:
            poly = PolynomialFeatures(3)
        all_poss = poly.fit_transform(X)
        poly_names = poly.get_feature_names(X.columns)
        poly_names = [w.replace('^2', '_sqrd') for w in poly_names]
        poly_names = [w.replace(' ', '_') for w in poly_names]
        poly_df = pd.DataFrame(all_poss, columns=poly_names)

    poly_df_org_first_row = poly_df.iloc[0, :] + 0  # plus 0 is to get rid of 'negative zeros' ie python format bug
    org_poly_df = poly_df.copy(deep=True) # this is used when creating forecasts on untranformed X data (ie when no centering)


    # remove irrelevant categorical features from 'remaining' list of candidate independent features for regression
    remaining = poly_df.columns.tolist()
    temp = []
    for x in remaining:
        if x in selected:
            temp.append(x)
        elif (x[:3] in explainer.category_prefix) and (x.endswith('_sqrd')):
            temp.append(x)
        elif (x[:3] in explainer.category_prefix) and ('_' in x):
            if x[:3] == x.split("_", 1)[1][:3]:
                temp.append(x)
    remaining = [x for x in remaining if x not in temp]
    # If required, transform Y and X (to Y - y1, X-x1) so that regression goes through the data point to be explained
    # eg https://stats.stackexchange.com/questions/12484/constrained-linear-regression-through-a-specified-point
    if CLEAR_settings.regression_type == 'multiple' and CLEAR_settings.centering is True:
        Y = single_regress.neighbour_df.loc[:, 'prediction'] - single_regress.nn_forecast
        poly_df = poly_df - poly_df.iloc[0, :]
    else:
        Y = single_regress.neighbour_df.loc[:, 'prediction'].copy(deep=True)

    Y = Y.reset_index(drop=True)


    poly_df['prediction'] = pd.to_numeric(Y, errors='coerce')
    if CLEAR_settings.score_type == 'aic':
        current_score, best_new_score = 100000, 100000
    else:
        current_score, best_new_score = -1000, -1000
    remaining = sorted(remaining, key = lambda x: (len(x), x))

    warnings.simplefilter('ignore', ConvergenceWarning)
    # if len(selected ) is greater than the user specified max_predictors, then increase max_predictors so that at least
    # 1 feature are selected by the regression
    if CLEAR_settings.max_predictors <= len(selected) < 25  :
        max_predictors = len(selected) + 3
    else:
        max_predictors =  CLEAR_settings.max_predictors +1
    while remaining and current_score == best_new_score and len(selected) < max_predictors:
        scores_with_candidates = []
        candidate_formulas = []


        for candidate in remaining:
            if CLEAR_settings.no_intercept == True:
                formula = "{} ~ {}".format('prediction', ' + '.join(selected) + '-1')
            elif CLEAR_settings.regression_type == 'multiple' and CLEAR_settings.centering is True:
                formula = "{} ~ {}".format('prediction', ' + '.join(selected + [candidate]) + '-1')
            else:
                formula = "{} ~ {}".format('prediction', ' + '.join(selected + [candidate]))
            candidate_formulas.append(formula)




        try:
            if CLEAR_settings.score_type == 'aic':
                if CLEAR_settings.regression_type == 'multiple':
                    scores_with_candidates = [sm.GLS.from_formula(formula, poly_df).fit(disp=0).aic for formula in
                                          candidate_formulas]
                else:
                    # score = sm.Logit.from_formula(formula, poly_df).fit(disp=0).aic
                    if CLEAR_settings.logistic_regularise == True:
                        scores_with_candidates = [sm.Logit.from_formula(formula, poly_df).fit_regularized(method='l1',disp=0).aic for formula in
                                                  candidate_formulas]
                    else:
                        #scores_with_candidates = [sm.Logit.from_formula(formula, poly_df).fit(disp=0).aic for formula in candidate_formulas]
                        scores_with_candidates = [smf.glm(formula, data= poly_df,family=sm.families.Binomial(),freq_weights=weights).fit(disp=0).aic for formula in candidate_formulas]
                    # candidate_formulas[0],
                    # data=poly_df,
                    # family=sm.families.Binomial())
                    #     sm.Logit.from_formula(formula, poly_df).fit(disp=0).aic for formula in
                    #                               candidate_formulas]






            elif CLEAR_settings.score_type == 'prsquared':
                if CLEAR_settings.regression_type == 'multiple':
                    print("Error prsquared is not used with multiple regression")
                    exit
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        #score = sm.Logit.from_formula(formula, poly_df).fit(disp=0).prsquared
                        scores_with_candidates = [sm.Logit.from_formula(formula, poly_df).fit(disp=0).prsquared for formula in candidate_formulas]
            elif CLEAR_settings.score_type == 'adjR':
                if CLEAR_settings.regression_type == 'multiple':
                    scores_with_candidates = [sm.OLS.from_formula(formula, poly_df).fit(disp=0).rsquared_adj for formula in
                                              candidate_formulas]
                else:
                    print("Error Ajusted R-squared is not used with logistic regression")

            else:
                print('score type not correctly specified')
                exit

        except np.linalg.LinAlgError as e:
            if 'Singular matrix' in str(e):
                print('Singular matrix error')
                del selected[-1]
                continue
        except:
            print("error in step regression")


        if len(scores_with_candidates) > 0:
            if CLEAR_settings.score_type == 'aic':  # For aic lower scores are better
                best_new_score, idx = min((val, idx) for (idx, val) in enumerate(scores_with_candidates))
            else:
                best_new_score, idx = max((val, idx) for (idx, val) in enumerate(scores_with_candidates))
            best_candidate = candidate_formulas[idx].split('+ ')[-1]
            if str(best_new_score) == 'inf':
                print('CLEAR regression failed - regression score = inf returned, consider using aic')
                exit()
            if ((CLEAR_settings.score_type == 'aic') and (current_score >= best_new_score)) or \
                ((CLEAR_settings.score_type != 'aic') and (current_score <= best_new_score)):
            # if ((CLEAR_settings.score_type == 'aic') and (current_score *1.002 > best_new_score)) or \
            #     ((CLEAR_settings.score_type != 'aic') and (current_score * 1.002 < best_new_score)):

            # check that the best candidate is not an interaction term in which one of the terms is essentail NOT adding value
            # i.e. it does not change the forecast compared to just using the other term but has been selected due to small
            # random effects e.g. in generating images or in the regression dataset.
            #     if '_' in best_candidate:
            #         features = best_candidate.split(sep='_')
            #         temp = [item for item in scores_with_candidates if item[1] in features]
                    # if len(temp)>0:
                    #     best_feature = max(temp, key=lambda t: t[0])
                    #     if (best_new_score - best_feature[0])/best_new_score < 0.005:
                    #                scores_with_candidates.append((best_new_score, best_candidate))
                    #                best_new_score = best_feature[0]
                    #                best_candidate = best_feature[1]
                    #                scores_with_candidates.remove((best_new_score, best_candidate))
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                print(best_candidate + ' added to stepwise regression')
                current_score = best_new_score
            else:
                print('was a break here')# break
            # perfect separation
        else:
            print('regression failed - possibly perfect separation: file ' + str(explainer.batch))
            single_regress.perfect_separation = True
            return single_regress

    if CLEAR_settings.no_intercept == True:
        formula = "{} ~ {}".format('prediction', ' + '.join(selected) + '-1')
        selected.remove('1')
    elif CLEAR_settings.regression_type == 'multiple' and CLEAR_settings.centering is True:
        formula = "{} ~ {}".format('prediction', ' + '.join(selected) + '-1')
        selected.remove('1')
    else:
        formula = "{} ~ {}".format('prediction', ' + '.join(selected))
    try:
        if CLEAR_settings.regression_type == 'logistic':
            if CLEAR_settings.logistic_regularise == True:
                classifier = sm.Logit.from_formula(formula, poly_df).fit_regularized(method='l1', disp=0)
            else:
                classifier= smf.glm(formula, data=poly_df, family=sm.families.Binomial(),freq_weights=weights).fit(disp=0)

        else:
            classifier = sm.GLS.from_formula(formula, poly_df).fit(disp=0)
        if CLEAR_settings.score_type == 'aic':
            single_regress.prediction_score = classifier.aic
        elif CLEAR_settings.score_type == 'prsquared':
            single_regress.prediction_score = classifier.prsquared
        elif CLEAR_settings.score_type == 'adjR':
            single_regress.prediction_score = classifier.rsquared_adj
        else:
            print('incorrect score type')
        predictions = classifier.predict(poly_df)
        single_regress.features = selected
        single_regress.coeffs = classifier.params.values

        single_regress.accuracy = 'Not calculated'
        # Below code for multiclass currently assumes decision boundary at 0.5.

        if CLEAR_settings.regression_type == 'logistic' or \
                (CLEAR_settings.regression_type == 'multiple' and CLEAR_settings.centering is False):
            single_regress.intercept = classifier.params[0]
            single_regress.spreadsheet_data = []
            for i in range(len(selected)):
                selected_feature = selected[i]
                for j in range(len(classifier.params)):
                    coeff_feature = classifier.params.index[j]
                    if selected_feature == coeff_feature:
                        single_regress.spreadsheet_data.append(poly_df_org_first_row.loc[selected_feature])
            single_regress.after_center_option = classifier.predict(poly_df)
            if CLEAR_settings.no_intercept == True:
                single_regress.intercept = 0
            else:
                single_regress.intercept = classifier.params[0]
            if CLEAR_settings.regression_type == 'multiple':    # Outstanding, I need to write an adjustment for McFadden pseudo R2
                single_regress.prediction_score = Calculate_Adj_R_Squared(single_regress.neighbour_df.loc[:, 'prediction'],
                    single_regress.after_center_option, classifier, single_regress, intercept=True)

        else:
        # Note on how CLEAR uses centering to force equation through observation
        #let the observation to be explained be x0,y0
        # Let Y = y-y0, X = x-x0. The regression equation to be found is then:
        #   y-y0 = ∝ (x-x0) ie. Y= ∝ X
        # ∝ is then estimated by a regression with no intercept. This is then transformed back to
        #   y-y0 = ∝(x-x0)     y =  ∝x - ∝x0 + y0
        # This equation passes through x0,y0. The new intercept is  ∝x0 + y0 and y needs to be adjusted by y0
        # Possible Concern: This assumes linearity but the regression includes nonlinear terms eg interaction terms.
        # Reply: This is not a problem as the interaction terms are all separately included on the input data,
        # and the regression is linear relative to this expansion.

            single_regress.spreadsheet_data = []
            single_regress.intercept = single_regress.nn_forecast
            for i in range(len(selected)):
                selected_feature = selected[i]
                for j in range(len(classifier.params)):
                    coeff_feature = classifier.params.index[j]
                    if selected_feature == coeff_feature:
                        single_regress.intercept -= poly_df_org_first_row.loc[selected_feature] * classifier.params[j]
                        single_regress.spreadsheet_data.append(poly_df_org_first_row.loc[selected_feature])
            adjustment = single_regress.nn_forecast - classifier.predict(poly_df_org_first_row)
            single_regress.after_center_option= adjustment[0] + classifier.predict(org_poly_df)

            single_regress.prediction_score=Calculate_Adj_R_Squared(single_regress.neighbour_df.loc[:, 'prediction'], single_regress.after_center_option, classifier, single_regress, intercept=True)

    except:
        print(formula)

    single_regress.local_prob = single_regress.after_center_option[0]
    if len(explainer.class_labels)>2:
        single_regress.regression_class = ""  # identification of regression class requires a seperate regression for each multi class
    else:
        if single_regress.local_prob >= CLEAR_settings.binary_decision_boundary:
            single_regress.regression_class = 1
        else:
            single_regress.regression_class = 0
        if single_regress.nn_forecast >= CLEAR_settings.binary_decision_boundary:
            single_regress.nn_class = 1
        else:
            single_regress.nn_class = 0
    return single_regress


def getCounterfactualDummies(explainer, nn_forecast, data_row, observation_num,dummy_trap):
    temp_df = explainer.catSensit_df[
        (explainer.catSensit_df['observation'] == observation_num) &
        ((explainer.catSensit_df.probability>= CLEAR_settings.binary_decision_boundary)
         != (nn_forecast >= CLEAR_settings.binary_decision_boundary))].copy(deep=True)


    # get categorical features which counterfactually change observation's class
    w = temp_df.feature.to_list()
    if explainer.data_type == 'image':
        v=[]
    else:
        y = [x[:3] for x in w]
        y = list(set(y))
        z = [col for col in data_row if
             (col in explainer.cat_features) and (data_row.loc[0, col] == 1)]
        v = [x for x in z if x[:3] in y]
        # ensure that at least 1 dummy variable is excluded for each categorical feature
        if dummy_trap is True:
            for i in y:
                x1 = len([u for u in w if u.startswith(i)])  # num dummy variables selected from categorical sensitivity file
                x2 = len([u for u in explainer.cat_features if u.startswith(i)])  # dummy variables in data_row
                if x2 == x1 + 1:
                    # drop dummy from w
                    x3 = [u for u in w if u.startswith(i)][-1]
                    w.remove(x3)
    return (w + v)

def addIncludedFeatures(selected, explainer):
    temp= []
    for i in CLEAR_settings.include_features_list:
       if i in explainer.numeric_features:
           temp.append(i)
       elif i in explainer.categorical_features:
           p = explainer.categorical_features.index(i)
           q = explainer.category_prefix[p]
           r= [u for u in explainer.cat_features if u.startswith(q)]
           # drop one feature to avoid dummy trap
           if len(r)> 1:
            r= r[:-1]
           temp = temp + r
       else:
           print(i + " in input parameter 'include_feature_list' but not in dataset")
    for j in temp:
        if j in selected:
            continue
        else:
            selected.append(j)
    return selected

def Create_Synthetic_Data(X_train, model, model_name,numeric_features,categorical_features, category_prefix,
                          class_labels, neighbour_seed):
    np.random.seed(neighbour_seed)
    explainer = Create_explainer(model, model_name, class_labels,categorical_features,category_prefix,
                                 numeric_features,data_type= 'tabular')
    explainer.feature_list = X_train.columns.tolist()
    explainer.num_features = len(explainer.feature_list)
    explainer.numeric_features = numeric_features
    explainer.feature_min = X_train.quantile(.01)
    explainer.feature_max = X_train.quantile(.99)
    # creates synthetic data
    if category_prefix == []:
        explainer.master_df = pd.DataFrame(np.zeros(shape=(CLEAR_settings.num_samples,
                                                      explainer.num_features)), columns=numeric_features)
        for i in numeric_features:
            explainer.master_df.loc[:, i] = np.random.uniform(explainer.feature_min[i],
                                                         explainer.feature_max[i], CLEAR_settings.num_samples)
    else:
        explainer.master_df = pd.DataFrame(np.zeros(shape=(CLEAR_settings.num_samples,
                                                      explainer.num_features)), columns=X_train.columns.tolist())
        for prefix in category_prefix:
            cat_cols = [col for col in X_train if col.startswith(prefix)]
            t = X_train[cat_cols].sum()
            st = t.sum()
            ut = t.cumsum()
            pt = t / st
            ct = ut / st
            if len(cat_cols) > 1:
                cnt = 0
                for cat in cat_cols:
                    if cnt == 0:
                        explainer.master_df[cat] = np.random.uniform(0, 1, CLEAR_settings.num_samples)
                        explainer.master_df[cat] = np.where(explainer.master_df[cat] <= pt[cat], 1, 0)
                    elif cnt == len(cat_cols) - 1:
                        explainer.master_df[cat] = explainer.master_df[cat_cols].sum(axis=1)
                        explainer.master_df[cat] = np.where(explainer.master_df[cat] == 0, 1, 0)
                    else:
                        explainer.master_df.loc[explainer.master_df[cat_cols].sum(axis=1) == 1, cat] = 99
                        v = CLEAR_settings.num_samples - \
                            explainer.master_df[explainer.master_df[cat_cols].sum(axis=1) > 99].shape[0]
                        explainer.master_df.loc[explainer.master_df[cat_cols].sum(axis=1) == 0, cat] \
                            = np.random.uniform(0, 1, v)
                        explainer.master_df[cat] = np.where(explainer.master_df[cat] <= (pt[cat] / (1 - ct[cat] + pt[cat])),
                                                       1, 0)
                    cnt += 1
        for i in numeric_features:
            explainer.master_df.loc[:, i] = np.random.uniform(explainer.feature_min[i],
                                                         explainer.feature_max[i], CLEAR_settings.num_samples)
    for j in category_prefix:
        temp = [col for col in X_train if col.startswith(j)]
        explainer.cat_features.extend(temp)
    return explainer

def Calculate_Adj_R_Squared(Y,predictions,classifier,single_regress,intercept):
    # This is needed as the data used in the regression is weighted so as to provide soft constraints on the counterfactuals (and for images on the
    # GAN image) predicxtions. The weighting is achieved by using duplicated rows which are labelled in the target_range column with either
    # 'GAN' or 'counterf'. This function strips out these duplicated rows and then adjusted_R is calculated. It is this value that then appears
    # in CLEAR's output. For discussion of calculating R squared without intercept read:
    # see https://stats.stackexchange.com/questions/26176/removal-of-statistically-significant-intercept-term-increases-r2-in-linear-mo

    #When using multiple regression, CLEAR forces the regression to go through the observation to be explained by (i) tranforming the data and then
    # (ii) carrying out a regression with no intercept.
    index_to_regress = single_regress.neighbour_df.index[~single_regress.neighbour_df['target_range'].isin(['GAN','counterf'])]
    Y_unweighted = Y.iloc[index_to_regress]
    predictions_unweighted=predictions.iloc[index_to_regress]
    ssr = sum((Y_unweighted - predictions_unweighted) ** 2)
    if intercept ==False:
        sst = sum((Y_unweighted) ** 2)
        r2 = 1 - (ssr / sst)
        adjusted_r_squared = 1 - (1 - r2) * (len(Y_unweighted)) / (len(Y_unweighted) - len(classifier.params) - 1)
    else:
        sst = sum((Y_unweighted - Y_unweighted.mean()) ** 2)
        r2 = 1 - (ssr / sst)
        adjusted_r_squared = 1 - (1 - r2) * (len(Y_unweighted) - 1) / (len(Y_unweighted) - len(classifier.params)-1 - 1)
    return (adjusted_r_squared)



class Create_explainer(object):
# generates synthetic data
    def __init__(self, model, class_labels,categorical_features,category_prefix,numeric_features,data_type):
        self.model = model
        self.cat_features = []
        self.class_labels = class_labels
        self.categorical_features = categorical_features
        self.category_prefix = category_prefix
        self.data_type= data_type
        if len(numeric_features) != 0:
            sensitivity_file = 'numSensitivity' + '.csv'
            self.sensit_df = pd.read_csv(CLEAR_settings.CLEAR_path + sensitivity_file)
        if len(category_prefix) != 0:
            sensitivity_file = 'catSensitivity' + '.csv'
            self.catSensit_df = pd.read_csv(CLEAR_settings.CLEAR_path + sensitivity_file)


