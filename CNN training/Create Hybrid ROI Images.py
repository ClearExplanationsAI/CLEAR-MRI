import CLEAR_settings
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from PIL import Image
from skimage import measure
from rectpack import newPacker
from shapely.geometry.polygon import LinearRing
from shapely.geometry import Polygon
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

PLORAS_stitched_MRI= 'ws_images_v5_noSON_NCL.npy'
PLORAS_spreadsheet = 'Patients_CathyExport_processed_v5.xlsx'
PLORAS_spreadsheet_sheet = 'No_SON_NCL'
draw_symbols = True
image_size =256
MRI_path = 'C:/Users/adamp/Second Images/'
spreadsheet = pd.read_excel(MRI_path + PLORAS_spreadsheet,sheet_name= PLORAS_spreadsheet_sheet)
spreadsheet['TPlusOneWeek SpeechScore'] =spreadsheet['TPlusOneWeek SpeechScore'].fillna(1)
ARTQ =spreadsheet['TPlusOneWeek SpeechScore'].astype(int)

temp = spreadsheet['New left hemisphere lesion size'].copy(deep=True)
temp.replace(to_replace=0, value=np.nan, inplace=True)
spreadsheet['lesion size'] = spreadsheet['New left hemisphere lesion size'].copy(deep=True)
spreadsheet['lesion size'] = spreadsheet['lesion size'].clip(upper=35000)
spreadsheet['lesion size'] = (spreadsheet['lesion size']-spreadsheet['lesion size'].min())/(spreadsheet['lesion size'].max()-spreadsheet['lesion size'].min())
spreadsheet['cat months'] = spreadsheet['Years between stroke and CAT'].copy(deep=True)
spreadsheet['cat months'] = spreadsheet['cat months']
spreadsheet['Age At Stroke'] = (spreadsheet['Age At Stroke']-spreadsheet['Age At Stroke'].min())/(spreadsheet['Age At Stroke'].max()-spreadsheet['Age At Stroke'].min())
spreadsheet['LeftHemisphereCentered']= spreadsheet['LeftHemisphereCentered'].clip(upper=4)
spreadsheet['LeftHemisphereCentered'] = (spreadsheet['LeftHemisphereCentered']-spreadsheet['LeftHemisphereCentered'].min())/(spreadsheet['LeftHemisphereCentered'].max()- spreadsheet['LeftHemisphereCentered'].min())
spreadsheet['RightHemisphereVoels'] = (spreadsheet['RightHemisphereVoels']-spreadsheet['RightHemisphereVoels'].min())/(spreadsheet['RightHemisphereVoels'].max()- spreadsheet['RightHemisphereVoels'].min())


d= np.load(MRI_path + PLORAS_stitched_MRI)

def stitch_slices(slices, num_slices):
    size = 8
    stitched = np.concatenate([np.rot90(slices[i]) for i in range(size)], 1)
    for batch in range(6 * 2, num_slices, size):
        h_layer = np.concatenate([np.rot90(slices[i]) for i in range(batch - size, batch)], 1)
        stitched = np.concatenate([stitched, h_layer], 0)


    return stitched

CLEAR_settings.init()
rectangular_images = np.load(MRI_path + 'rectangular_images.npy')
new_training_array = np.zeros((len(rectangular_images), image_size, image_size))
top_segments = [81, 85, 13, 57, 63, 11, 29,71,83,61,7,37]
new_training_grid = np.genfromtxt("C:/Users/adamp/Dropbox/CLEAR MRI - desk/grid_for_11_12_ROIs.csv", delimiter=',')
segments = np.genfromtxt("C:/Users/adamp/Dropbox/CLEAR MRI - desk/632_by_760_segments.csv", delimiter=',')

for Master_idx in range(rectangular_images.shape[0]):
    new_training_image = np.zeros((image_size,image_size))
    rectangluar_image = rectangular_images[Master_idx, :, :]
    cnt = 0
    for z in top_segments:
        print(str(z))
        print(str(cnt))
        first_row = np.where(new_training_grid==cnt+1)[0].min()
        last_row= np.where(new_training_grid==cnt+1)[0].max()+1
        first_column = np.where(new_training_grid==cnt+1)[1].min()
        last_column = np.where(new_training_grid==cnt+1)[1].max()+1
        row_width =  last_row - first_row
        column_width = last_column - first_column

        new_square = np.zeros((row_width, column_width)).astype(int)
        bins = [(column_width,row_width), (column_width,row_width)]

        mask = np.zeros(segments.shape).astype(bool)
        new_image = np.zeros(segments.shape).astype(int)
        packer = newPacker(rotation=False)
        if type(z) is list:
            for i in z:
                mask[segments == i] = True
        else:
            mask[segments == z] = True
        new_image[mask] = rectangluar_image[mask]
        blobs= measure.label(mask, background=None, return_num=False, connectivity=2)
        regions = regionprops(blobs)
        if len(top_segments) !=1:
            rectangles= [(x.bbox[3]-x.bbox[1],x.bbox[2]-x.bbox[0],x.label-1) for x in regions]
        else:
            rectangles = [(x.bbox[3] - x.bbox[1]+5, x.bbox[2] - x.bbox[0]+5, x.label - 1) for x in regions]
        for r in rectangles:
            packer.add_rect(*r)
        for b in bins:
            packer.add_bin(*b)
        packer.pack()
        number_bins=0
        for abin in packer:
            number_bins +=1
            for rect in abin:
                if len(top_segments) !=1:
                    new_square[rect.y:rect.top,rect.x:rect.right]= new_image[regions[rect.rid].bbox[0]:regions[rect.rid].bbox[2],regions[rect.rid].bbox[1]:regions[rect.rid].bbox[3]]
                else:
                    new_square[rect.y+5:rect.top, rect.x+5:rect.right] = new_image[regions[rect.rid].bbox[0]:regions[rect.rid].bbox[2],
                                                                         regions[rect.rid].bbox[1]: regions[rect.rid].bbox[3]]

            if number_bins>1:
                print('For segment ' + str(z) + ' the number bins is ' + str(number_bins))
        new_training_image[first_row:last_row,first_column:last_column] = new_square
        cnt+=1


    temp = Image.fromarray(new_training_image)#.convert('RGB')
    draw = ImageDraw.Draw(temp)
    if draw_symbols is True:
        if  ARTQ[Master_idx] in [1,2,3,4]:
            fnt = ImageFont.truetype("C:/Windows/Fonts/wingding.ttf", 40, encoding="symb")
            draw.multiline_text((210, 210), u'', font=fnt, fill='white')

        elif ARTQ[Master_idx]== 5:
            draw.regular_polygon((215, 230, 10), 3, fill='white', outline=None)
            draw.regular_polygon((215, 230, 10), 3, fill='white', rotation=60, outline=None)
        elif ARTQ[Master_idx] == 6:
            draw.regular_polygon((225, 230, 10), 3, rotation=270, fill=None, outline='white')
            draw.regular_polygon((215, 230, 10), 3, fill=None, rotation=90, outline='white')
        elif ARTQ[Master_idx]== 7:
            draw.ellipse((205, 220, 235, 230),fill='white')

        draw.regular_polygon((235,120,5+20*spreadsheet.loc[Master_idx,'lesion size']), 5, fill = 600, outline='white')  #was 600 before that 175

        t = 10 * spreadsheet.loc[Master_idx, 'cat months']
        draw.pieslice([(215, 160), (245, 190)], start=0, end=270, fill=50 + 30 * t)
    new_training_image= np.array(temp)
    plt.imshow(new_training_image)
    plt.show()
    new_training_array[Master_idx, :, :] = new_training_image
    temp = Image.fromarray(new_training_image).convert('RGB')
    temp.save("C:/Users/adamp/New Training Data/New_training_" + str(Master_idx) + ".png")
np.save(MRI_path + 'new training.npy',new_training_array )

print('finished')

