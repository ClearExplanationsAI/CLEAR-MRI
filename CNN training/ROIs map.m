clear all
clc
%temp =gunzip('template_MNI152.nii.gz')
%img = niftiread('D:\BrainAtlas\BrainnetomeAtlasViewer\Brainnetome_v1.0.2\Template\template_MNI152.nii')

img = niftiread('C:\Users\adamp\BrainAtlas\BrainnetomeAtlasViewer\Brainnetome_v1.0.2\Template\aal.nii');
%img = niftiread('D:\MRI files\aal Tom.nii');

im_edge = 8;
start_slice = 1;
end_slice = 64
tiles = {};
cd('C:\Users\adamp\MRI files\margarita_segments')
for i=0:116
    im = (img==i);
    tile = tile_cubes(im,im_edge,start_slice,end_slice);
   % tile = imresize(tile,[256,256],'nearest');
    imshow(tile);
    filename = strcat('AAL_',int2str(i),'.bmp');
    saveas(gcf,filename)
    tiles{i+1} = tile;
end
segments = zeros(632,760);
for i=1:length(tiles)
   tile = tiles{i};
   segments(tile==1)=i-1;
end
%%
function tile = tile_cubes(im3D,im_edge,start_slice,end_slice)
if(size(im3D,1)~=68)
    im3D=imresize3(im3D,[79,95,68]); 
end
[s1,s2,s3] = size(im3D);
tile = zeros(s1*im_edge,s2*im_edge,'single');
ind = start_slice;
for i=1:im_edge
    try
        start = ((i-1)*s1)+1;
        ending = (start-1)+s1;
        for j=1:im_edge
            startc = ((j-1)*s2)+1;
            endingc = (startc-1)+s2;  
            tile(start:ending,startc:endingc) = squeeze(im3D(:,:,ind));
            ind = ind + 1;
            disp(ind);
        end
    catch ME
        tmp = squeeze(im3D(:,:,ind));
        tmp = imresize(tmp,[91,109]);
        tile(start:ending,startc:endingc) = tmp;
        disp(ME.message)
    end
end
end      