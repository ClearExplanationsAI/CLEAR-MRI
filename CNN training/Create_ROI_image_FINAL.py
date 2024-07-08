import CLEAR_settings
import numpy as np
from skimage.measure import regionprops
from PIL import Image
from skimage import measure
from rectpack import newPacker
image_size =256
PLORAS_spreadsheet = 'Patients_CathyExport_processed_v5.xlsx'
PLORAS_spreadsheet_sheet = 'No_SON_NCL'
MRI_path = 'C:/Users/adamp/Second Images/'
d= np.load(MRI_path + 'ws_images.npy')
lesion = np.load(MRI_path + 'fuzzy_data.npy') # created in check_input.py


def stitch_slices(slices, num_slices):
    size = 8
    stitched = np.concatenate([np.rot90(slices[i]) for i in range(size)], 1)
    for batch in range(size * 2, num_slices, size):
        h_layer = np.concatenate([np.rot90(slices[i]) for i in range(batch - size, batch)], 1)
        stitched = np.concatenate([stitched, h_layer], 0)


    return stitched

rel_slices =np.zeros(68)
for i in range(len(lesion)):
    for j in range(68):
        rel_slices[j] += lesion[i, j].sum()

lesion_background = d[:,np.where(rel_slices>30000)[0].tolist() , 10:80,10:] #selects 42 images
stitched_lesion_background = []
for i in range(len(lesion)):
    stitched_lesion_background.append(stitch_slices(lesion_background[i, :, :, :],43))
stitched_lesion_background= [((x> 10).astype(np.uint8))* 0.3 for x in stitched_lesion_background]
print('h')
#

rel_lesion= lesion[:, np.where(rel_slices > 20000)[0].tolist() , 10:80, 10:]
lesion_stitched = []
for i in range(len(lesion)):
    lesion_stitched.append(stitch_slices(rel_lesion[i, :, :, :],43))


new_lesion=  [np.maximum(i[0],i[1]) for i in zip(lesion_stitched,stitched_lesion_background)]
resized_lesion = []
for i in range(len(lesion)):
    img = (new_lesion[i] * 255).astype(np.uint8)
    img = Image.fromarray(img, 'L')
    img = img.resize((image_size, 150))
    img = np.array(img)
    te= np.zeros(img.shape).astype(bool)
    te[img>120] = 1
    img[te]= 255
    resized_lesion.append(img)


CLEAR_settings.init()
top_segments = [85, 13, 57, 51,29,63,81,1,11,75,71,73,77]  #lesion 13 segs training grid.csv"
new_training_array = np.zeros((len(lesion), image_size, image_size))
new_training_grid = np.genfromtxt("C:/Users/adamp/Dropbox/CLEAR MRI - desk/lesion training grid3.csv", delimiter=',')
segments = np.genfromtxt("C:/Users/adamp/Dropbox/CLEAR MRI - desk/632_by_760_segments.csv", delimiter=',')
rectangular_images = np.load(MRI_path + 'rectangular_images.npy')
for Master_idx in range(rectangular_images.shape[0]):
    # print('idx is ' + str(Master_idx))
    new_training_image = np.zeros((image_size,image_size))
    rectangluar_image = rectangular_images[Master_idx, :, :]
    cnt = 0
    for z in top_segments:
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

    new_training_array[Master_idx, :, :] = new_training_image

    temp = Image.fromarray(new_training_array[Master_idx]).convert('RGB')
    temp.save("C:/Users/adamp/New Training Data/New_training_" + str(Master_idx) + ".png")
np.save(MRI_path + 'new training.npy',new_training_array )

print('finished')

