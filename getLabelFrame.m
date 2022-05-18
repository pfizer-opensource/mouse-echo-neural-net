function getLabelFrame(labeledDir,unlabeledDir,saveName,file_nums)
% Function to generate training data for a directory of m-mode ultrasound
% images. The function requires as inputs the location of the manually
% labeled images, the location of the unlabeled images, and a location in
% which the training data should be saved. Data is saved as ~1MB .mat files
% containing a cropped unlabeled image, a cropped label image, and the x
% and y step size (in mm) of unlabeled data. The data is given the same name as
% the original files with a .mat extension. The user can choose to select
% certain files within the directories to label by setting the "file_nums"
% variable to a list of images to include, or can use the keyword "all" to
% simply label the full directory. The user can also use the "verbose" flag
% to have the labeling results displayed as the program runs.
%
% Mary Kate Montgomery
% Mary.Montgomery@pfizer.com
% September 2020
% Pfizer, Inc

% Update user
disp(['Starting ' unlabeledDir]);

% Parse inputs
if ischar(file_nums) && strcmp(file_nums,'all')
    file_nums = 1:numel(dir(labeledDir))-2;
end

if isfile(saveName)
    [~,~,saveArray] = xlsread(saveName);
else
    saveArray = {'File name','Labeled Frame','Crop Start','Crop End'};
end

for f = 1:numel(file_nums)
    % Read stack of labeled images
    imnum = file_nums(f);
    [cropped_im,~,fname] = load_im(labeledDir,imnum);
    [~,~,~,dz] = size(cropped_im);
    
    % Find frame with most blue/cyan markings
    blue_sizes = zeros(dz,1);
    for z = 1:dz
        blue = get_blue(cropped_im,z);
        blue_sizes(z) = sum(sum(blue));
    end
    [maxBlue,labeledFrame] = max(blue_sizes);
    if maxBlue == 0
        % No labels on this image
        continue;
    end
     % Use recorded frame
    blue = get_blue(cropped_im,labeledFrame);
    
    % Close up any breaks in lines
    blue = imclose(blue,strel('disk',2,4));
    
    % Keep only four largest objects
    blue = keep_largest(blue,4);
    if isempty(blue)
        continue;
    end
    
    
    % Separate into 4 lines
    labelIm = bwlabel(blue')'; % transpose before taking bw label in order to find objects top to bottom (rather than left to right)
    % Get non-line regions of image
    nonSeg = ~blue;
    % Crop columns of frame which do not contain all 4 lines
    nonSeg(:,~min(cat(1,max(labelIm==1,[],1),max(labelIm==2,[],1),max(labelIm==3,[],1),max(labelIm==4,[],1)),[],1)) = 0;
    % Take only 5 largest objects
    nonSeg = keep_largest(nonSeg,5);
    if isempty(nonSeg)
        % If not 5 objects, exclude scan
        continue;
    end
    
    % Separate image into 5 regions
    imRegions = bwlabel(nonSeg')';
    
    % Check labels are sufficiently large
    if sum(sum(imRegions==0)) > numel(imRegions)*0.8
        continue;
    end
    
    
    % Close up holes in regions 
    % (usually left by notches in segmentation lines and overlap with words)
    for i = 1:5
        iReg = (imRegions==i);
        iReg = imclose(iReg,strel('disk',25,8));
        imRegions(iReg==1) = i;
    end
    
    % Crop any parts of segmentation with too thick of lines between
    % segmented regions (due to overlap of words)
    imRegions(:,sum(double(imRegions==0),1)>32) = 0; %allows for ~8 pixels per "line".
    
    % Ensure all regions are continuous
    imRegions2 = zeros(size(imRegions));
    for i = 1:5
        iReg = (imRegions==i);
        iReg = makeContinuous(iReg,8);
        imRegions2(iReg==1) = i;
    end
    imRegions = imRegions2; clear imRegions2
    
    xstart = find(sum(imRegions,1)>0,1,'first');
    xend = find(sum(imRegions,1)>0,1,'last');
    saveArray = cat(1, saveArray, {fname, labeledFrame,xstart,xend});
    
end
xlswrite(saveName,saveArray);
disp([saveName ' Finished!']);
end

function [im, im_inf, fname] = load_im(dirName,imnum)
% Load and crop image
d = dir([dirName '\*.dcm']);
fname = d(imnum).name;
im = dicomread(fullfile(dirName,fname));
% Crop to image part only
im = cropVevoImage(im);
% Read dicom info
im_inf = dicominfo(fullfile(dirName,fname));
end

function blue = get_blue(im,z)
% Get lines from image
[dy,dx,~,~] = size(im);
frame = im(:,:,:,z);
blue = ones([dy,dx]);
blue(frame(:,:,1)>frame(:,:,3)) = 0; % remove red lines
blue(frame(:,:,1)==frame(:,:,2)) = 0; % remove gray
blue(frame(:,:,3)<frame(:,:,2)) = 0; % remove green lines
end

function bwOut = keep_largest(bwIn,n)
% Function to remove all but n largest connected objects from bw mask
bwOut = bwIn;

% Find largest connected blue objects
C = bwconncomp(bwIn,8);

% If can't find n objects, assume image does not contain full segmentation
if C.NumObjects < n
    bwOut = [];
    return;
end

% Take n largest objects 
NumPix = cellfun(@numel,C.PixelIdxList);
[~,idx]= sort(NumPix,'descend');
for i = n+1:length(idx)
    i_del = idx(i);
    bwOut(C.PixelIdxList{i_del}) = 0;
end
end