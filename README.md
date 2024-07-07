# MRI
Scripts for the paper 'Predicting recovery following stroke: Deep learning, multimodal data and feature selection using explainable AI'

There are three folders. CLEAR-MRI stores the scripts for the explaianable AI method CLEAR-MRI that explains the classifications of a CNN trained on the PLORAS dataset. 


## Prerequisites 
CLEAR-MRI is written in Python 3.7.9 and runs on Windows 10.The YAML file specifing the required configuration is CLEARMRI.yml

CLEAR-MRI requires the following input files:
A Pytorch CNN that classifies each 2-D MRI image. These can either be stitched images or hybrid images.
A numpy files of 2D MRI images.  The dimensions are (number of patients, 632, 740).
A spreadsheet specifying Patient ID and the spoken decscription scores of each patient.


## Installation 
The file CLEAR_settings.py contains the parameter variables for CLEAR MRI. Open CLEAR_settings.py and change the value of parameter CLEAR_path to the name of the directory you have created for CLEAR e.g., CLEAR_path='D:/CLEAR/'. CLEAR Image requires the user to have created both a directory containing the original image files and another directory containing the corresponding GAN images. The addresses of these two directories should be assigned in CLEAR_settings to the parameters: ‘input_images_path’ and ‘GAN_images_path’ e.g., ‘input_images_path’ = ‘D:/Diseased_Xrays/’. In order to get the original images for CheXpert, the CheXpert dataset will need to be downloaded: https://stanfordmlgroup.github.io/competitions/chexpert/ . The GAN generated images generated from CheXpert are in folder ‘CheXpert GAN’. The file ‘Medical.csv' lists the images that were used in our paper. ‘Synthetic Images .csv’ lists the synthetic images used, these are included in the folder ‘Synthetic test images’, the corresponding GAN images are in the folder ‘GAN Synthetic data’. 

## Running CLEAR Image 
Running CLEAR.py will process all the images listed in ‘Medical.csv' for CheXpert or ‘Synthetic Images.csv’ for the synthetic dataset. CLEAR will generate a HTML report explaining a single prediction if only one image is listed in the csv file. The report is entitled ‘CLEAR_Image_report_full.html'. It is expected that CLEAR Image will normally be run for single images (rather than batches of images). For each image listed in the csv file, a png file and a csv file of CLEAR’s corresponding saliency map is generated; they are saved to the directory specified by CLEAR_path, and should be transferred to a results folder that the user has created. 
Two detailed Excel spreadsheets are also created each time that CLEAR Image is run. ‘Fidelity.xlsx’ contains data for each counterfactual that CLEAR Image identifies. ‘Results_Xrays.xlsx’ contains regression data for each image, column L can be used to calculate the number of images that were classified as ‘causally overdetermined’. It is not expected that a user would normally access either of these spreadsheets. 

## Scripts that are not part of CLEAR Image 
Generating Pointing Game Scores 
