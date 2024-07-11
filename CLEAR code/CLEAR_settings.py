from tkinter import *

""" Specifes CLEAR'S user input parameters. CLEAR sets the input parameters as global variables
whose values are NOT changed in any other module (these are CLEAR's only global variables).
Tkinter is used to provide some checks on the user's inputs. The file 
'Input Parameters for CLEAR.pdf' on Github documents the input parameters.
 """


def init():
    global case_study, random_seed, max_predictors, num_samples, regression_type, spreadsheet_file,spreadsheet_sheet,\
        score_type, test_sample, CLEAR_path,images_path,model_path,images_to_process, normalise_constants, \
        reactangle_segments, apply_counterfactual_weights, counterfactual_weight,brain_atlas_segments, \
        generate_regression_files, all_interactions, no_intercept, centering, no_polynomimals_no_complete_set_interactions, \
        use_prev_sensitivity, binary_decision_boundary,include_features, \
        include_features_list, image_infill,image_all_segments,\
        image_use_old_synthetic,image_segment_type, start_obs, finish_obs, \
        debug_mode,image_model_name,image_classes, logistic_regularise, sufficiency_threshold,\
        max_sufficient_causes, image_counterfactual_interactions, stitched_array,interactions_from_list

    case_study = 'MRI' #'Medical' or 'Synthetic' or 'MRI'
    image_model_name= 'ResNet.pth'#'Simple_valid_breakdowns_calib0.0001.pth'#'ResNet_rescale_seg1_patch_balanced_calib_valid0.0001.pth' #ResNet_rescale_seg1_patch_balanced_calib_valid0.0001.pth'
    stitched_array =  'stitched images for CLEAR.npy'#   'stitched images for CLEAR.npy'#'stitched_2NatLab.npy'#'stitched images with symbols.npy'
    reactangle_segments = "632_by_760_segments.csv"
    spreadsheet_file = "Patients_CathyExport_processed_v5.xlsx"
    spreadsheet_sheet = "No_SON_NCL"
    brain_atlas_segments = 'brain_segments1.npy'
    normalise_constants = 'ImageNet' #'ImageNet'
    start_obs = 1
    finish_obs =  1
    random_seed = 2
    max_predictors = 21# maximum number of dependent variables in stepwise regression  21
    num_samples = 2000  # number of observations to generate in Synthetic Dataset. Default 1000
    regression_type = 'logistic'  # 'multiple' 'logistic'
    logistic_regularise = False
    score_type = 'aic'  # prsquared is McFadden Pseudo R-squared. Can also be
    #                          set to aic or adjR (adjusted R-squared)
    CLEAR_path = 'C:/Users/adamp/CLEAR/'  # e.g. 'D:/CLEAR/''C:/Users/adamp/CLEAR/'
    images_path = 'C:/Users/adamp/Second Images/'
    model_path = 'C:/Users/adamp/Second Images/'
    apply_counterfactual_weights = True
    counterfactual_weight = 0.05  # weighting (as proportion of num_sample) applied to each counterfactual image in regression
    generate_regression_files = True
    binary_decision_boundary = 0.50
    # Parameters for evaluating the effects of different parts of CLEAR's regression
    no_polynomimals_no_complete_set_interactions = True
    interactions_from_list = False
    all_interactions = False
    no_intercept = False # only for multiple regression - probably delete this, it was for early version of CLEAR IMAGE - Otherwise need to change Adjusted R2
    centering = True #forces CLEAR's regression to pass through observation that is to be explained.
    # Parameters for forcing features to be included in regression
    include_features = False # Features in 'include_feature_list' will be forced into regression equation
    include_features_list = ['S86DdSeg86','S90DdSeg90', 'S08DdSeg08', 'S14DdSeg14'] #,'S52DdSeg52','S60DdSeg60','S52DdSeg52','S60DdSeg60','S04DdSeg04', 'S88DdSeg88' ]
    debug_mode = False
    sufficiency_threshold = 0.99 # for determining cases of overdetermination
    max_sufficient_causes = 2 # This should be set to 1 or 2

    image_infill ='GAN' # 'GAN'
    image_all_segments= False
    image_use_old_synthetic = False  # Only set to True when testing code
    image_counterfactual_interactions = False
    image_segment_type ='external mask'
    image_classes =['no recovery','recovery']
    check_input_parameters()
""" Check if input parameters are consistent"""


def check_input_parameters():
    def close_program():
        root.destroy()
        sys.exit()

    error_msg = ""
    if regression_type == 'logistic' and \
            (score_type != 'prsquared' and score_type != 'aic'):
        error_msg = "logistic regression and score type combination incorrectly specified"
    elif regression_type == 'multiple' and score_type == 'prsquared':
        error_msg = "McFadden Pseudo R-squared cannot be used with multiple regression"
    elif no_intercept == True and regression_type != 'multiple':
        error_msg = "no intercept requires regression type to be multiple"
    elif no_intercept == True and centering == True:
        error_msg = "centering requires no-intercept to be False"
    elif case_study not in ['Medical', 'Synthetic', 'MRI']:
        error_msg = "case study incorrectly specified"
    elif regression_type not in ['multiple', 'logistic']:
        error_msg = "Regression type misspecified"
    elif (isinstance((all_interactions & centering & no_polynomimals_no_complete_set_interactions  &
                      apply_counterfactual_weights & generate_regression_files), bool)) is False:
        error_msg = "A boolean variable has been incorrectly specified"

    if error_msg != "":
        root = Tk()
        root.title("Input Error in CLEAR_settings")
        root.geometry("350x150")

        label_1 = Label(root, text=error_msg,
                        justify=CENTER, height=4, wraplength=150)
        button_1 = Button(root, text="OK",
                          padx=5, pady=5, command=close_program)
        label_1.pack()
        button_1.pack()
        root.attributes("-topmost", True)
        root.focus_force()
        root.mainloop()


