from __future__ import print_function, division
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms, utils
import matplotlib.pyplot as plt
import time
import copy
import math
import datetime
import daft
import torchvision
import timm
from temperature_scaling import ModelWithTemperature
from torch.utils.data import WeightedRandomSampler
from PIL import Image, ImageDraw, ImageFont
from Med3D_setting import parse_opts
from Med3D_model import generate_model
import warnings
from scipy.special import softmax

test_group = 5
exclude_background = True
input_epochs_num = 200
num_output_channels = 3
MRI_path = 'C:/Users/adamp/Second Images/'
img_size = 256
batch_size_input = 16
model_type = 'ResNet' # 'ResNet' #'DAFT' #'Early_Fusion' 'ResNet3D' 'Early_Fusion_3D','Lightweight'
num_groups = 5
class_names = ['no_recovery','recovery']
use_ROI_dataset = False
create_stitched_images = True
stitched_image_file = 'ws_images_v5_noSON_NCL.npy'
PLORAS_spreadsheet = 'Patients_CathyExport_processed_v5.xlsx'
PLORAS_spreadsheet_sheet = 'No_SON_NCL'
stitched_images_with_symbols = False
detail_analysis = True
print_messages = True

warnings.filterwarnings("ignore", category=UserWarning)

class MultiDataset(torch.utils.data.Dataset):
    def __init__(self, dependent_df, independent_df, images):
        self.dependent_df = dependent_df
        self.independent_df = independent_df
        self.images = images

    def __len__(self):
        return len(self.dependent_df)

    def __getitem__(self, idx):
        independent_vars = torch.from_numpy(self.independent_df.iloc[idx, 0:].values).float()
        labels = torch.tensor(self.dependent_df.loc[idx, 'label']).float()
        images = self.images[idx]
        return images, independent_vars, labels

class MultiModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, block, layers):
        super(MultiModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size + 512, 2)
        self.fc4 = torch.nn.Linear(self.input_size + 512, 256)
        self.fc5 = torch.nn.Linear(256, 2)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def joint_forward(self, image, tabular):
        r_x = self.conv1(image)  # 224x224
        r_x = self.bn1(r_x)
        r_x = self.relu(r_x)
        r_x = self.maxpool(r_x)  # 112x112
        r_x = self.layer1(r_x)  # 56x56
        r_x = self.layer2(r_x)  # 28x28
        r_x = self.layer3(r_x)  # 14x14
        r_x = self.layer4(r_x)  # 7x7
        r_x = self.avgpool(r_x)  # 1x1

        r_x = torch.flatten(r_x, 1)
        x = self.fc1(tabular)
        x = self.relu(x)
        x = self.fc2(x)
        out = torch.cat((r_x, x), dim=1)
        out = self.fc3(out)
        return out

    def forward(self, image, tabular):
        r_x = self.conv1(image)  # 224x224
        r_x = self.bn1(r_x)
        r_x = self.relu(r_x)
        r_x = self.maxpool(r_x)  # 112x112
        r_x = self.layer1(r_x)  # 56x56
        r_x = self.layer2(r_x)  # 28x28
        r_x = self.layer3(r_x)  # 14x14
        r_x = self.layer4(r_x)  # 7x7
        r_x = self.avgpool(r_x)  # 1x1
        r_x = torch.flatten(r_x, 1)
        x = self.fc1(tabular)
        out = torch.cat((r_x, x), dim=1)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.fc5(out)
        return out



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def multi_resnet18(num_vars):
    layers = [2, 2, 2, 2]
    model = MultiModel(num_vars, num_vars, BasicBlock, layers)
    return model


def stitch_slices(slices, size=6):
    size = 8
    stitched = np.concatenate([np.rot90(slices[i]) for i in range(size)], 1)

    for batch in range(size * 2, 65, size):
        h_layer = np.concatenate([np.rot90(slices[i]) for i in range(batch - size, batch)], 1)
        stitched = np.concatenate([stitched, h_layer], 0)
    # stitched = (stitched - stitched.mean())/stitched.std()
    return stitched

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

        self.dropout = nn.Dropout(0.6)
        self.fc = nn.Linear(2048, 128)
        self.fcc = nn.Linear(128, 2)
        self.fc1 = nn.Sigmoid()

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


def label2onehot(labels, dim):
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out

if use_ROI_dataset is True:
    model_name = 'zoom_' + model_type +'.pth'
elif  stitched_images_with_symbols is True:
    model_name = 'stitch_sym_' + model_type +'.pth'
else:
    model_name = model_type + '.pth'


s1_all = list()
s2_all = list()

if model_type in ['ResNet3D','Early_Fusion_3D','DAFT']:
    d = np.load(MRI_path + stitched_image_file)
    s = np.expand_dims(d, axis=1)
    for i in range(len(s)):
        t = s[i, :, :, :]
        t = ((t - t.min()) / (t.max() - t.min()) * 255).astype(np.float32)
        t = torch.from_numpy(t)
        s1_all.append(t)


elif use_ROI_dataset == True:
    s = np.load(MRI_path + 'new training.npy')

else:
    d = np.load(MRI_path + stitched_image_file)
    segments = np.genfromtxt(MRI_path + "segments no ventricles.csv", delimiter=',')
    if create_stitched_images is True:
        stitched = []
        for i in range(len(d)):
            stitched.append(
                stitch_slices(d[i, :, :, :]))

        s_all = np.array(stitched)
        np.save(MRI_path + 'rectangular_images.npy', s_all)
spreadsheet = pd.read_excel(MRI_path + PLORAS_spreadsheet, sheet_name=PLORAS_spreadsheet_sheet)
spreadsheet['lesion size'] = spreadsheet['New left hemisphere lesion size'].copy(deep=True)
spreadsheet['lesion size'] = spreadsheet['lesion size'].clip(upper=35000)
spreadsheet['lesion size'] = (spreadsheet['lesion size'] - spreadsheet['lesion size'].min()) / (
            spreadsheet['lesion size'].max() - spreadsheet['lesion size'].min())
spreadsheet['TPlusOneWeek SpeechScore'] = spreadsheet['TPlusOneWeek SpeechScore'].fillna(1)
ARTQ = spreadsheet['TPlusOneWeek SpeechScore'].astype(int)
spreadsheet['cat months'] = spreadsheet['Years between stroke and CAT'].copy(deep=True)
spreadsheet['cat months'] = spreadsheet['cat months'].clip(upper=5)
spreadsheet['cat months'] = (spreadsheet['cat months'] - spreadsheet['cat months'].min()) / (
            spreadsheet['cat months'].max() - spreadsheet['cat months'].min())
categorical_features = ['TPlusOneWeek SpeechScore']
category_prefix = ['sev']
independent_df = spreadsheet[['TPlusOneWeek SpeechScore', 'lesion size', 'cat months']].copy(deep=True)
independent_df = pd.get_dummies(independent_df, prefix=category_prefix, columns=categorical_features,
                                drop_first=True)


dependent_df = spreadsheet[['SpkPicDescTOTTS']].copy(deep=True)
dependent_df['label'] = (dependent_df['SpkPicDescTOTTS'] >= 60).astype('int')
test_stats = ['seed','val group', 'val_bal_acc', 'val_loss','test group', 'val_acc', 'test_acc', 'bal_acc','bal_impair0',
              'bal_impair1','bal_impair2', 'bal_impair3', 'bal_NA_impair', 'ece', 'impair0', 'impair1', 'impair2',
              'impair3', 'NA_impair', 'impair0_cnt', 'impair1_cnt', 'impair2_cnt', 'impair3_cnt',
              'NA_impair_cnt', 'impair0_acc_cnt', 'impair1_acc_cnt',
               'impair2_acc_cnt', 'impair3_acc_cnt', 'NA_impair_acc_cnt']
test_df = pd.DataFrame(columns=test_stats)
detail_stats = ['seed','ID', 'val group', 'test group','impair','prob', 'prediction', 'label']
detail_df = pd.DataFrame(columns=detail_stats)
validation_stats = ['seed','val_group', 'learning_rate','best_val_bal_acc','best_val_loss']
validation_df = pd.DataFrame(columns=validation_stats)


if model_type not in ['ResNet3D', 'Early_Fusion_3D','DAFT']:
    if model_type == 'Lightweight':
       transform_means = transform_std = (0.5,0.5,0.5)

    else:
        transform_means = IMAGENET_DEFAULT_MEAN
        transform_std = IMAGENET_DEFAULT_STD

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=num_output_channels),
        transforms.ToTensor(),
        transforms.Normalize(transform_means,transform_std)])



    if use_ROI_dataset == True:
        for i in range(len(s)):
            t = s[i, :, :]
            t = (t - t.min()) / (t.max() - t.min()) * 255
            s1_all.append(transform(t.astype(np.float32)))

    else:
        transform2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size))])

        for i in range(len(s_all)):
            tem = transform2(s_all[i, :, :].astype(np.float32))
            t = np.array(tem)
            t = (t - t.min()) / (t.max() - t.min()) * 255

            if stitched_images_with_symbols == True:
                 temp = Image.fromarray(t)  # .convert('RGB')
                 draw = ImageDraw.Draw(temp)
                 if ARTQ[i] in [1, 2, 3, 4]:
                     fnt = ImageFont.truetype(MRI_path + "wingding.ttf", 30, encoding="symb")
                     draw.multiline_text((215, 215), u'', font=fnt, fill='white')
                 elif ARTQ[i] == 5:
                     draw.regular_polygon((215, 230, 10), 3, fill="white", outline=None)
                     draw.regular_polygon((215, 230, 10), 3, fill="white", rotation=60, outline=None)
                 elif ARTQ[i] == 6:
                     draw.regular_polygon((225, 230, 10), 3, rotation=270, fill=None, outline="white")
                     draw.regular_polygon((215, 230, 10), 3, fill=None, rotation=90, outline="white")
                 elif ARTQ[i] == 7:
                     draw.ellipse((205, 220, 235, 230), fill='white')
                 else:
                     print("ARTQ error " + str(i))

                 draw.regular_polygon((175, 235, 5 + 18 * spreadsheet.loc[i, 'lesion size']), 5,
                                      fill=175, outline='white')
                 t = 5 * spreadsheet.loc[i, 'cat months']  # was 10*
                 draw.pieslice([(130, 225), (150 + t, 245 + t)], start=0, end=225, fill=150)  # was 150
                 t = np.array(temp)
            s2_all.append(t)
            s1_all.append(transform(t.astype(np.float32)))
        s2_all= np.asarray(s2_all)
    np.save(MRI_path + 'stitched images for CLEAR.npy', s2_all)





output_row=-1
detail_cnt = 0
validation_stats_cnt = -1
for seed_num in range(0,20):
    if print_messages is True:
        print('seed number is : ' + str(seed_num))
    torch.manual_seed(seed_num)
    comp_row = -1
    training_list = [1,2,3,4,5]
    training_list.remove(test_group)
    for j in training_list:
        output_row += 1
        val_group = j
        new_training_group = copy.deepcopy(training_list)
        new_training_group.remove(j)
        train_indices = spreadsheet.index[spreadsheet.NewGroups.isin(new_training_group)]
        val_indices = spreadsheet.index[spreadsheet.NewGroups.isin([val_group])]
        test_indices = spreadsheet.index[spreadsheet.NewGroups.isin([test_group])]
        ARTQ = spreadsheet['TPlusOneWeek SpeechScore'].values
        ARTQ_list = [0 if i <=1.5 else i for i in ARTQ]
        ARTQ_list = [1 if 2 <= i <= 5.5 else i for i in ARTQ_list]
        ARTQ_list = [2 if i == 6 else i for i in ARTQ_list]
        ARTQ_list = [3 if i >= 6.5 else i for i in ARTQ_list]  # 1 is normal speech
        train_dependent_df = dependent_df.iloc[train_indices, :].copy(deep=True)
        train_dependent_df.reset_index(inplace=True)
        train_independent_df = independent_df.iloc[train_indices, :].copy(deep=True)
        val_dependent_df = dependent_df.iloc[val_indices, :].copy(deep=True)
        val_dependent_df.reset_index(inplace=True)
        val_independent_df = independent_df.iloc[val_indices, :].copy(deep=True)
        test_dependent_df = dependent_df.iloc[test_indices, :].copy(deep=True)
        test_dependent_df.reset_index(inplace=True)
        test_independent_df = independent_df.iloc[test_indices, :].copy(deep=True)
        train_images = torch.stack([s1_all[i] for i in train_indices])
        val_images = torch.stack([s1_all[i] for i in val_indices])
        test_images = torch.stack([s1_all[i] for i in test_indices])

        train_dataset = MultiDataset(train_dependent_df, train_independent_df, train_images)
        val_dataset = MultiDataset(val_dependent_df, val_independent_df, val_images)
        test_dataset = MultiDataset(test_dependent_df, test_independent_df, test_images)


        class_sample_count= train_dependent_df.label.value_counts()
        weights = 1. / class_sample_count
        sample_weights = [weights[i] for i in train_dependent_df.label.values]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_input,
                                                       num_workers=0, pin_memory=True,shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_input, shuffle=False,
                                                     num_workers=0, pin_memory=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_images), shuffle=False,
                                                     num_workers=0, pin_memory=True)

        dataset_sizes = {'train': train_independent_df.shape[0], 'val': val_independent_df.shape[0]}
        num_vars = train_independent_df.shape[1]
        if print_messages is True:
            print(dataset_sizes)


        def train_model(model_type, learning_rate, num_epochs=25):
            since = time.time()
            if model_type == 'ResNet':
                # ResNet18
                # model = torchvision.models.resnet18(pretrained=False)
                model = torch.load(MRI_path + 'org_Res_Net.pth')
                num_ftrs = model.fc.in_features
                model.fc = torch.nn.Linear(num_ftrs, len(class_names))

                # model.fc = torch.nn.Sequential(
                #     torch.nn.Linear(num_ftrs, 256),
                #     torch.nn.ReLU(),
                #     torch.nn.Dropout(0.3),
                #     torch.nn.Linear(256, len(class_names))
                # )

            elif model_type == 'Early_Fusion':
                model = multi_resnet18(num_vars)

            elif model_type == 'DAFT':
                model = daft.DAFT(1, 2)

            elif model_type == 'Lightweight':
                model = CNN()  ##

            elif model_type == 'ResNet3D':
                sets = parse_opts()
                model, parameters = generate_model(sets)

            elif model_type == 'Early_Fusion_3D':
                sets = parse_opts()
                model, parameters = generate_model(sets)


            else:
                print('model specification error')

            model = model.cuda()
            with torch.no_grad():
                torch.cuda.empty_cache()
            cudnn.benchmark = True
            torch.cuda.memory_allocated()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if print_messages is True:
                print(device)


            balance_weights = torch.tensor([2, 0.5]).cuda()
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=balance_weights)
            if model_type == 'Lightweight':
                optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)  # was 0.0001
                scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            elif model_type in ['ResNet3D', 'Early_Fusion_3D']:
                params = [
                    {'params': parameters['base_parameters'], 'lr': learning_rate},
                    {'params': parameters['new_parameters'], 'lr': sets.learning_rate * 100}  #unclear what this is for

                ]
                optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-3)
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

            else:
                optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
                scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)




            best_model_wts = copy.deepcopy(model.state_dict())
            best_val_bal_acc = 0.0
            best_val_acc = 0.0
            best_val_loss = 10000
            train_acc = 0.0
            best_epoch = 0
            cumLoss = []
            cumAcc = []

            for epoch in range(num_epochs):
                if train_acc>0.96:  #was 0.98
                    break
                if print_messages is True:
                    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                    print('-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()  # Set model to training mode
                        running_loss = 0.0
                        running_corrects = 0

                        # Iterate over data.
                        cnt = 0
                        for images, inputs, labels in train_dataloader:
                            cnt += 1
                            inputs = inputs.cuda()
                            labels_oh = label2onehot(labels, dim=2)
                            labels = labels_oh.cuda()
                            images = images.cuda()

                            # zero the parameter gradients
                            optimizer.zero_grad()

                            # forward
                            # track history if only in train
                            with torch.set_grad_enabled(phase == 'train'):
                                if model_type in ['ResNet3D','ResNet','Lightweight']:
                                    outputs = model(images)
                                else:
                                    outputs = model(images, inputs)
                                _, preds = torch.max(outputs,
                                                     1)  # preds is the index (o or 1) for each prediction with the highest value
                                loss = criterion(outputs, labels)
                                loss.backward()
                                optimizer.step()

                            # statistics
                            running_loss += loss.item() * inputs.size(0)
                            running_corrects += torch.sum(preds == torch.max(labels.data, 1)[1])

                        epoch_loss = running_loss / dataset_sizes[phase]
                        epoch_acc = running_corrects.double() / dataset_sizes[phase]
                        scheduler.step(epoch_acc)

                    else:
                        model.eval()  # Set model to evaluation mode
                        running_loss = 0.0
                        running_corrects = 0
                        val_total_pos = val_true_pos = val_total_neg = val_true_neg = 0


                        for images, inputs, labels in val_dataloader:
                            images = images.cuda()
                            inputs = inputs.cuda()
                            labels_oh = label2onehot(labels, dim=2)
                            labels_org = labels
                            labels = labels_oh.cuda()

                            # zero the parameter gradients
                            optimizer.zero_grad()

                            # forward
                            # track history if only in train
                            if model_type in ['ResNet','ResNet3D','Lightweight']:
                                outputs = model(images)
                            else:
                                outputs = model(images, inputs)
                            _, preds = torch.max(outputs,1)  # preds is the index (0 or 1) for each prediction with the highest value
                            loss = criterion(outputs, labels)

                            # statistics
                            running_loss += loss.item() * inputs.size(0)
                            running_corrects += torch.sum(preds == torch.max(labels.data, 1)[1])

                            for t in range(len(labels_org)):
                                if labels_org[t] == 1:
                                    val_total_pos += 1
                                    if preds[t] == 1:
                                        val_true_pos += 1
                                elif labels_org[t] == 0:
                                    val_total_neg += 1
                                    if preds[t] == 0:
                                        val_true_neg += 1
                                else:
                                    print('val labels error')

                        epoch_val_bal_acc =  ((val_true_pos / val_total_pos) +  (val_true_neg/ val_total_neg))/2


                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / dataset_sizes[phase]

                    if phase == 'train':
                        scheduler.step(epoch_acc)
                        train_acc = epoch_acc

                    if phase == 'val':
                        cumLoss.append(epoch_loss)
                        cumAcc.append(epoch_acc)

                    if print_messages is True:
                        print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                    if (phase == 'val') and (epoch_loss <= best_val_loss):
                        best_val_loss = epoch_loss
                        best_val_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                        best_epoch = epoch
                        best_val_bal_acc = epoch_val_bal_acc
                if print_messages is True:
                    print('best epoch ' + str(best_epoch))

            time_elapsed = time.time() - since
            if print_messages is True:
                print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed

                                                                 // 60, time_elapsed % 60))
                print('Best Val Acc: {:.4f}'.format(best_val_acc))

            # Load the best model weights
            model.load_state_dict(best_model_wts)

            return model, best_epoch, best_val_acc, best_val_loss, best_val_bal_acc

        best_val_bal_acc_all_parms = 0
        for learning_rate in [1e-4, 5e-4, 1e-5]:#5e-3,
            model_ft, epoch_ft, best_val_acc, best_val_loss, best_val_bal_acc = train_model(model_type, learning_rate, num_epochs=input_epochs_num)
            validation_stats_cnt += 1
            validation_df.loc[validation_stats_cnt,'seed'] = seed_num
            validation_df.loc[validation_stats_cnt, 'val_group'] = val_group
            validation_df.loc[validation_stats_cnt, 'learning_rate'] = learning_rate
            validation_df.loc[validation_stats_cnt,'best_val_bal_acc'] = best_val_bal_acc
            validation_df.loc[validation_stats_cnt, 'best_val_loss'] = best_val_loss


            if best_val_bal_acc > best_val_bal_acc_all_parms:
                best_val_bal_acc_all_parms = best_val_bal_acc
                best_val_acc_all_parms=best_val_acc.cpu().detach().numpy().item()
                ResNet_cal = ModelWithTemperature(model_ft)
                _, ece = ResNet_cal.set_temperature(val_dataloader, model_type)
                # torch.save(ResNet_cal, MRI_path + 'calib_ResNet' + str(learning_rate) + '.pth')
                torch.save(ResNet_cal, MRI_path + model_name)


        model = torch.load(MRI_path + model_name)
        if torch.cuda.is_available():
            model.to('cuda')
        with torch.no_grad():
            model.eval()
            correct = total = total_pos= true_pos = total_neg = true_neg = 0
            total_impair0_pos = true_impair0_pos = total_impair0_neg = true_impair0_neg = 0
            total_impair1_pos = true_impair1_pos = total_impair1_neg = true_impair1_neg = 0
            total_impair2_pos = true_impair2_pos = total_impair2_neg = true_impair2_neg = 0
            total_impair3_pos = true_impair3_pos = total_impair3_neg = true_impair3_neg = 0
            total_NA_impair_pos = true_NA_impair_pos = total_NA_impair_neg = true_NA_impair_neg = 0
            bal_impair0 = bal_impair1 = bal_impair2 = bal_impair3 = bal_NA_impair= 0

            all_pred = []
            all_lab = []
            # for i, data in enumerate(test_dataloader, 0):
            impair0_cnt = impair1_cnt = impair2_cnt = impair3_cnt = NA_impair_acc_cnt = 0
            impair0 = impair1 =impair2 = impair3 = NA_impair = 0
            impair0_acc_cnt = impair1_acc_cnt = impair2_acc_cnt = impair3_acc_cnt = NA_impair_cnt = 0
            # Iterate over data.
            for images, inputs, labels in test_dataloader:
                images = images.cuda()
                inputs = inputs.cuda()
                labels= labels.cuda()

                if model_type in ['ResNet', 'ResNet3D','Lightweight']:
                    outputs = model(images)
                else:
                    outputs = model(images, inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum()

                for t in range(len(labels)):
                    if labels[t]== 1:
                        total_pos +=1
                        if preds[t]== 1:
                            true_pos +=1
                    elif labels[t]== 0:
                        total_neg +=1
                        if preds[t]== 0:
                            true_neg +=1
                    else:
                        print('labels error')

                for l in labels:
                    all_lab.append(l.cpu().detach().numpy())
                for p in preds:
                    all_pred.append(p.cpu().detach().numpy())
                impairs = [ARTQ_list[x] for x in test_indices][0:len(test_images)]
                for k in range(len(impairs)):
                    if impairs[k] == 0:
                        if labels[k] ==1:
                            total_impair0_pos += 1
                            if preds[k] == 1:
                                true_impair0_pos += 1
                        elif labels[k] == 0:
                            total_impair0_neg += 1
                            if preds[k] == 0:
                                true_impair0_neg += 1
                        impair0_cnt += 1
                        if preds[k] == labels[k]:
                            impair0_acc_cnt += 1
                    elif impairs[k] == 1:
                        if labels[k] == 1:
                            total_impair1_pos += 1
                            if preds[k] == 1:
                                true_impair1_pos += 1
                        elif labels[k] == 0:
                            total_impair1_neg += 1
                            if preds[k] == 0:
                                true_impair1_neg += 1

                        impair1_cnt += 1
                        if preds[k] == labels[k]:
                            impair1_acc_cnt += 1
                    elif impairs[k] == 2:
                        if labels[k] == 1:
                            total_impair2_pos += 1
                            if preds[k] == 1:
                                true_impair2_pos += 1
                        elif labels[k] == 0:
                            total_impair2_neg += 1
                            if preds[k] == 0:
                                true_impair2_neg += 1
                        impair2_cnt += 1
                        if preds[k] == labels[k]:
                            impair2_acc_cnt += 1
                    elif impairs[k] == 3:
                        if labels[k] == 1:
                            total_impair3_pos += 1
                            if preds[k] == 1:
                                true_impair3_pos += 1
                        elif labels[k] == 0:
                            total_impair3_neg += 1
                            if preds[k] == 0:
                                true_impair3_neg += 1
                        impair3_cnt += 1
                        if preds[k] == labels[k]:
                            if labels[k] == 1:
                                total_impair3_pos += 1
                                if preds[k] == 1:
                                    true_impair3_pos += 1
                            elif labels[k] == 0:
                                total_impair3_neg += 1
                                if preds[k] == 0:
                                    true_impair3_neg += 1
                            impair3_acc_cnt += 1
                    #isnan should be zero as currently missing is being assigned to group 1
                    elif math.isnan(impairs[k]) is True:
                        if labels[k] == 1:
                            total_NA_impair_pos += 1
                            if preds[k] == 1:
                                true_NA_impair_pos += 1
                        elif labels[k] == 0:
                            total_NA_impair_neg += 1
                            if preds[k] == 0:
                                true_NA_impair_neg += 1


                        NA_impair_cnt += 1
                        if preds[k] == labels[k]:
                            NA_impair_acc_cnt += 1
                    else:
                        print('ARTQ error')
                    if detail_analysis == True:
                        detail_df.loc[detail_cnt, 'seed'] = seed_num
                        detail_df.loc[detail_cnt, 'ID'] = k + test_indices[0]
                        detail_df.loc[detail_cnt, 'impair'] = impairs[k]
                        detail_df.loc[detail_cnt, 'val group'] = val_group
                        detail_df.loc[detail_cnt, 'test group'] = test_group
                        detail_df.loc[detail_cnt, 'prob'] = softmax(outputs[k].cpu().detach().numpy())[1]
                        detail_df.loc[detail_cnt, 'prediction'] = preds[k].cpu().detach().numpy().item()
                        detail_df.loc[detail_cnt, 'label'] = labels[k].cpu().detach().numpy().item()
                        detail_cnt +=1



            accu = correct / total
            if impair0_cnt > 0:
                impair0 = impair0_acc_cnt /impair0_cnt
            if total_impair0_pos> 0 and total_impair0_neg >0:
                bal_impair0 = ((true_impair0_pos/total_impair0_pos) + (true_impair0_neg/total_impair0_neg))/2

            if impair1_cnt > 0:
                impair1 = impair1_acc_cnt /impair1_cnt
            if total_impair1_pos> 0 and total_impair1_neg >0:
                bal_impair1 = ((true_impair1_pos/total_impair1_pos) + (true_impair1_neg/total_impair1_neg))/2
            if impair2_cnt > 0:
                impair2 = impair2_acc_cnt / impair2_cnt
            if total_impair2_pos> 0 and total_impair2_neg >0:
                bal_impair2 = ((true_impair2_pos/total_impair2_pos) + (true_impair2_neg/total_impair2_neg))/2
            if impair3_cnt > 0:
                impair3 = impair3_acc_cnt / impair3_cnt
            if total_impair3_pos> 0 and total_impair3_neg >0:
                bal_impair3 = ((true_impair3_pos/total_impair3_pos) + (true_impair3_neg/total_impair3_neg))/2
            if NA_impair_cnt > 0:
                NA_impair = NA_impair_acc_cnt /NA_impair_cnt
            if total_NA_impair_pos > 0 and total_NA_impair_neg > 0:
                bal_NA_impair = ((true_NA_impair_pos / total_NA_impair_pos) + ( true_NA_impair_neg/total_NA_impair_neg)) / 2

            test_df.loc[output_row, 'seed'] = seed_num
            test_df.loc[output_row, 'val group'] = val_group
            test_df.loc[output_row, 'test group'] = test_group
            test_df.loc[output_row, 'val_acc'] =best_val_acc_all_parms
            test_df.loc[output_row, 'val_bal_acc'] = best_val_bal_acc_all_parms
            test_df.loc[output_row,'test_acc'] =accu.cpu().detach().numpy().item()
            balanced_acc =     ((true_pos / total_pos) +  (true_neg/ total_neg))/2
            test_df.loc[output_row, 'bal_acc'] = balanced_acc
            test_df.loc[output_row, 'bal_impair0'] = bal_impair0
            test_df.loc[output_row, 'bal_impair1'] = bal_impair1
            test_df.loc[output_row, 'bal_impair2'] = bal_impair2
            test_df.loc[output_row, 'bal_impair3'] = bal_impair3
            test_df.loc[output_row, 'bal_NA_impair'] = bal_NA_impair
            test_df.loc[output_row, 'impair0'] = impair0
            test_df.loc[output_row, 'impair1'] =impair1
            test_df.loc[output_row, 'impair2'] =impair2
            test_df.loc[output_row, 'impair3'] =impair3
            test_df.loc[output_row, 'NA_impair'] =NA_impair
            test_df.loc[output_row, 'impair0_cnt'] =impair0_cnt
            test_df.loc[output_row, 'impair1_cnt'] =impair1_cnt
            test_df.loc[output_row, 'impair2_cnt'] =impair2_cnt
            test_df.loc[output_row, 'impair3_cnt'] =impair3_cnt
            test_df.loc[output_row, 'NA_impair_cnt'] =NA_impair_cnt
            test_df.loc[output_row, 'impair0_acc_cnt'] =impair0_acc_cnt
            test_df.loc[output_row, 'impair1_acc_cnt'] =impair1_acc_cnt
            test_df.loc[output_row, 'impair2_acc_cnt'] =impair2_acc_cnt
            test_df.loc[output_row, 'impair3_acc_cnt'] =impair3_acc_cnt
            test_df.loc[output_row, 'NA_impair_acc_cnt'] =NA_impair_acc_cnt

            if print_messages is True:
                print('Accuracy of the model on the test images: {} %'.format(accu))
                print('Balanced Accuracy: {} %'.format(balanced_acc))

            all_lab = [x * 1 for x in all_lab]
            all_pred =[x * 1 for x in all_pred]

if detail_analysis == True:
    now = datetime.datetime.now()
    detail_df.to_excel(MRI_path + model_name +'_detail' + str(now.hour) + str(now.minute) + str(now.second) + '.xlsx')
validation_df.to_excel(MRI_path +  model_name +'_val' + str(now.hour) + str(now.minute) + str(now.second) + '.xlsx')
test_df.to_excel(MRI_path + model_name +'_test' + str(now.hour) + str(now.minute) + str(now.second) + '.xlsx')
print('finished')
