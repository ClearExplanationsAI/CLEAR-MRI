# MRI
Scripts for the paper:

A. White, M. Saranti, A. d’Avila Garcez, T.M.H. Hope, C.J. Price, H. Bowman,
Predicting recovery following stroke: Deep learning, multimodal data and feature selection using explainable AI,
NeuroImage: Clinical (2024)

There are three folders. 

* CNN training stores the scripts for training the CNNs referenced in the paper.
* CLEAR-MRI stores the scripts for the explaianable AI method CLEAR-MRI that explains the classifications of a CNN trained on the PLORAS dataset.
* Statistical Verication stores the scripts for the statistical analyses specified in appendices A1 and A2. In addition, it contains the script for calculating AUC and F1 scores for all models.


## CNN training

Lockbox final.py is the script for training the CNNs evaluated in the paper.

Nine CNN models can be trained, as listed on page 9 of the paper.

The model to be trained is specified by the following 3 parameters at the begining of the Lockbox.py script.
* 'model_type': ResNet', 'DAFT','Early_Fusion', 'ResNet3D', 'Early_Fusion_3D','Lightweight'
* 'use_ROI_dataset'. Set to true for model to be trained with ROI dataset rather than with Stitched MRI dataset
* 'stitched_images_with_symbols'. Set to true to use Hybrid Stitched MRI dataset

The following input files are required:

*  a numpy array of 2D stitched MRI images created from the PLORAS dataset.  Each stitched image consists of sixty-four axial cross-sectional spatially normalised MRI slices in a 2D 632 x 760 image.
*  a spreadsheet specifying the patient ID, the  PLORAS tabular features specified in section 2.1 of the paper (initial severity, left hemisphere lesion size, recovery time, CAT spoken description scores), and the group number each patient ID was assigned to for training/validation of the CNNs.
*  a csv file mapping the 632 x 740 pixels in each stitched image to grey matter anatomical regions-of-interest (ROIs).The Matlab script 'ROIs_map.m' creates this csv file using Brainnetome_v1.0.2\Template\aal.nii
* For training CNNs on either the ROI dataset or the Hybrid ROI dataset, 'Create Hybrid ROI Images.py' must first be run to create a numpy array 'new training.npy' of the new images.

The CNN models are calbrated using a single parameter variant of Platt scaling. The calibration algorithm is from: Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger Proceedings of the 34th International Conference on Machine Learning, PMLR 70:1321-1330, 2017. Our script temperature.py is a modified copy of Pleiss's script from http://github.com/gpleiss/temperature_scaling.


## CLEAR-MRI

CLEAR-MRI is written in Python 3.7.9 and runs on Windows 10. The YAML file specifing the required configuration is clear_mri.yml

CLEAR-MRI requires the following input files:
*  the Pytorch CNN that classifies each 2-D MRI image. These can either be stitched images or hybrid stitched images.
*  numpy array of 2D stitched images, spreadsheet of PLORAS tabular data and file mapping stitch images to ROIs of 2D MRI images specified above.

The file CLEAR_settings.py contains the parameter variables for CLEAR-MRI. Open CLEAR_settings.py and change the value of parameter CLEAR_path to the name of the directory you have created for CLEAR e.g., CLEAR_path='D:/CLEAR/'.  
CLEAR-MRI is run by running CLEAR.py. CLEAR-MRI can be run either to explain the prediction of a single (hybrid) stitched image or a batch of such images. In the case of a single image, CLEAR-MRI will generate a HTML report explaining a single prediction if only one image is listed in the csv file. The report is entitled ‘CLEAR_Image_report_full.html'.

##  Statistical Verification

The scripts in this folder are used to:
* Run a simulation of our cross-validation procedure under the null to show the inflation of the false-positive rate for a number of statistical tests (testing_cross_validation_for_paper.m), see figure A1 of the paper.
* Calculate confidence intervals for the unbalanced test accuracy and balanced test accuracy across all patients and across language impaired patients for all the models (confidence_intervals_calculation.py).
* Calculate AUC and F1 scores for each model in the paper, along with their confidence intervals (calculate_AUC_F1_scores.py).
* Perform paired t-tests to compare the best model reported in the paper (Hybrid ROIs w/ ResNet-18) with all the rest of the models (statisticalTest_with_dfCorrection.py).

The simulation led to the development of a new method for correcting the degrees of freedom to be able to perform a t-test that does not increase the type-1 error rate. This is the procedure described in appendix A1 of the paper. The paired t-tests in statisticalTests_with_dfCorrection.py are performed after applying this degrees of freedom correction. Similarly, the calculation of confidence intervals for unbalanced and balanced accuracies as well as AUC and F1 scores all follow the method described in the appendix.

All the scripts in this folder (apart from the simulation) require the detailed spreadsheets with results that are generated after running Lockbox final.py from the CNN training folder:
* statisticalTests_with_dfCorrection.py and confidence_intervals_calculation.py require the spreadsheets saved from test_df in Lockbox final.py and
* calculate_AUC_F1_scores.py requires the spreadsheets saved from detail_df in Lockbox final.py
  
The path in the beginning of each script has to be changed to the respective folder where these spreadsheets are saved. 



