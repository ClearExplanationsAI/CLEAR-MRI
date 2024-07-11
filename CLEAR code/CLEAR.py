import numpy as np
import pandas as pd
from PIL import Image
import CLEAR_perturbations
import CLEAR_regression
import CLEAR_settings
import CLEAR_image
import torch
import torch.nn as nn
from datetime import datetime

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.fc = nn.Linear(2048, 128)
        self.fcc = nn.Linear(128, 2)


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = out.view(out.size(0), -1)
        out = self.fcc(self.fc(out))
        return out




def Run_CLEAR_MRI():
    CLEAR_settings.init()
    model = torch.load(CLEAR_settings.model_path + CLEAR_settings.image_model_name)
    c_counter_master_df = pd.read_pickle(CLEAR_settings.CLEAR_path + "c_counter_master_df.pkl")
    results_master_df = pd.read_pickle(CLEAR_settings.CLEAR_path + "results_master_df.pkl")
    np.random.seed(CLEAR_settings.random_seed)
    rectangular_images= np.load(CLEAR_settings.images_path+ 'rectangular_images.npy') # created when training neural net model. Corresponds to array 's' i.e before transform operations.                                                                        # used only for output report.
    stitched_images = np.load(CLEAR_settings.images_path  + CLEAR_settings.stitched_array)  # should be the rectangular array before resizing?
    spreadsheet_df = pd.read_excel(CLEAR_settings.images_path +  CLEAR_settings.spreadsheet_file, sheet_name=CLEAR_settings.spreadsheet_sheet)
    #the label is the CAT spoken picture description as described on pages 4 and 5 of the NeuroImage: Clinical paper
    labels_df = spreadsheet_df.loc[0:len(stitched_images)-1,['ID','SpkPicDescTOTTS']]
    labels_df['SpkPicDescTOTTS'] = (labels_df['SpkPicDescTOTTS'] > 60).astype(int)
    for Master_idx in range(CLEAR_settings.start_obs,CLEAR_settings.finish_obs+1):
        if spreadsheet_df.loc[Master_idx,'NewGroups']==5:
            continue
        try:
            t = rectangular_images[Master_idx, :, :]
            t = (t - t.min()) / (t.max() - t.min())*255
            rectangluar_image = Image.fromarray(t).convert('RGB')
            rectangluar_image.save('rectangular_image.png')
            CLEAR_image.Get_Closest(Master_idx, model,stitched_images,labels_df)
            org_img, model, top_label, top_pred, top_idx = CLEAR_image.Create_PyTorch_Model(model,Master_idx,stitched_images)
            img,segments,top_label,top_idx, model = CLEAR_image.Segment_image(org_img,model,top_label, top_idx)
            explainer=CLEAR_image.Create_image_sensitivity(img,segments,top_label,top_pred,top_idx, model)
            t = np.array(rectangluar_image).astype(float)
            explainer.rectangular_image = (t - t.min()) / (t.max() - t.min())
            explainer.observation_num = Master_idx
            explainer.file_name = 'MRI_' + str(labels_df.loc[Master_idx,'ID'])
            explainer, X_test_sample = CLEAR_image.Create_synthetic_images(explainer)
            (results_df, explainer, single_regress) = CLEAR_regression.Run_Regressions(X_test_sample, explainer)
            if single_regress.perfect_separation is True:
                print('perfect separation')
            (nncomp_df, missing_log_df) = CLEAR_perturbations.Calculate_Perturbations(explainer, results_df)
            c_counter_df=CLEAR_perturbations.Single_prediction_report(results_df, nncomp_df, single_regress, explainer)
            c_counter_master_df = c_counter_master_df.append(c_counter_df, ignore_index = True)

            results_master_df =  results_master_df.append(results_df, ignore_index=True)
            missing_log_df.to_csv(CLEAR_settings.CLEAR_path + 'Missing.csv', index=False)
            print(str(Master_idx))
        except:
            print('no explanation generated for ' + str(Master_idx))
    c_counter_master_df.to_pickle(
    CLEAR_settings.CLEAR_path + 'c_batch_df' + datetime.now().strftime("%Y%m%d-%H%M%S%f") + '.pkl')
    results_master_df.to_pickle(CLEAR_settings.CLEAR_path + 'results_batch_df.pkl')
    results_master_df.to_csv(CLEAR_settings.CLEAR_path + 'Results.csv', index=False)
    print('run completed')
    return

def Summary_report():
    CLEAR_settings.init()
    results_df=pd.read_pickle(CLEAR_settings.CLEAR_path + "results_batch_df.pkl")
    all_features = [x for xs in results_df['features'].to_list() for x in xs]
    all_features = list(set(all_features))
    feature_scores_df = pd.DataFrame(columns=all_features)
    feature_scores_df.loc[0] = 0
    feature_scores_df.loc[1] = 0
    cnt = 0
    for i,row in results_df.iterrows():
        if results_df.loc[i, 'nn_forecast'] >0.5:
            cnt+=1
        if results_df.loc[i, 'features'][0] == '1':
            results_df.loc[i, 'features'].remove('1')
            temp = results_df.loc[i, 'weights'].tolist()
            temp.pop(0)
            results_df.loc[i, 'weights'] = np.array(temp)
        features =results_df.loc[i, 'features']
        weights = results_df.loc[i, 'weights']
        idx = 0 if results_df.loc[i, 'nn_forecast']>0.5 else 1
        for j in range(len(features)):
            feature_scores_df.loc[idx, features[j]] += abs(weights[j])/(abs(weights)).max()
    feature_scores_df.to_excel(CLEAR_settings.CLEAR_path + 'MRI_global.xlsx', index=False)

    print('summary ready')
    print('number recovery is ' + str(cnt))
    print('number not recovery is ' + str(len(results_df)-cnt))
    return


if __name__ == "__main__":
    Run_CLEAR_MRI()
    Summary_report()