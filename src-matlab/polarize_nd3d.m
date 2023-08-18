% segmentation models:
disp('Loading segmentation models ...')
load('C:/iris-recognition---pm-diseased-human-driven-bsif/models/SegNetWarmPostmortemDiseaseCoarse-MTPhD.mat')
modelCoarse = net;
load('C:/iris-recognition---pm-diseased-human-driven-bsif/models/SegNetWarmPostmortemDiseaseFine-MTPhD.mat')
modelFine = net;
clear net

disp(['Reading the matching pairs file ...'])
files = readcell('C:/GBIR_datasets/ND3DIris/bonafide_imagelist.txt');

DIR_SAVE_POL_IM = 'C:/GBIR_datasets/ND3DIris/polar_im_bf/';
DIR_SAVE_POL_MASK = 'C:/GBIR_datasets/ND3DIris/polar_mask_bf/';

DIR_IMAGES_TO_PROCESS = 'C:/GBIR_datasets/ND3DIris/all_bonafide_images/';

disp(['Found ' num2str(numel(files)) ' unique files; segmenting, normalizing and encoding ...'])
for i = 1:numel(files)
    
    disp([num2str(i) '/' num2str(numel(files)) ': ' char(files(i))]);
    
    % step 1: segmentation
    % (coarse for circular approximations; fine for actual occlusion mask)
    
    filePath = char(files(i));
    image = imread(fullfile(DIR_IMAGES_TO_PROCESS,filePath));
    if size(image,3) > 3
        image = image(:,:,1:3);
    end
    maskCoarse = pmIrisSegment(image, modelCoarse);
    maskFine = pmIrisSegment(image, modelFine);
    
    % step 2: circular approximation and normalization
    [pupilData, irisData, status] = pmIrisCircApprox(maskCoarse);
    
    if strcmp(status,'OK')
        [imagePol,maskPol] = pmIrisCartesianToPolar(image,maskFine,pupilData,irisData);
        imwrite(imagePol, [DIR_SAVE_POL_IM filePath])
        imwrite(maskPol, [DIR_SAVE_POL_MASK filePath])
    end
end