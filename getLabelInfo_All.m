function getLabelInfo_All(labeled_dir,unlabeled_dir,saveFileName)
% Function to read labeled data for all m-mode scans within a given
% directory. The directory can contain multiple studies, with multiple
% series. The function requires as inputs the location of the manually
% labeled images, the location of the unlabeled images, and a location in
% which the training data should be saved. Data is saved as a csv 
% containing the frame number and cropping indices for each scan, labeled 
% by scan name.
%
% Mary Kate Montgomery
% Mary.Montgomery@pfizer.com
% March 2021
% Pfizer, Inc

% Check can locate directories
notDir = check_dir({unlabeled_dir,labeled_dir});
if ~isempty(notDir)
    disp(['Error: Cannot locate ' strjoin(notDir,' or ')]);
    return
end
isFmt = checkFormat(saveFileName);
if ~isFmt
    disp('Error: Please enter a .csv or .xlsx save file');
    return
end

% Get names of studies
studyNames = getSubfolders(unlabeled_dir);
for study_n = 1:numel(studyNames)
    % Set up folder names
    labStudyDir = fullfile(labeled_dir,studyNames{study_n});
    unlabStudyDir = fullfile(unlabeled_dir,studyNames{study_n});
    % Label study
    label_study(labStudyDir,unlabStudyDir,saveFileName);
    
end
end

function missingDirs = check_dir(dirList)
% Function to check if any directories contained in dirList do not exist.
% If so, a list of missing directories is returned. If not, an empty cell
% is returned.
missingDirs = {}; missingCt = 0;
for i = 1:numel(dirList)
    if ~isfolder(dirList{i})
        missingCt = missingCt + 1;
        missingDirs{missingCt} = dirList{i};
    end
end
end

function label_study(labeled_dir,unlabeled_dir,saveFileName)
% Function to read labeled data for a study. Locates all subfolders
% within the study and runs the getLabelFrame function on each
% folder.

% Get names of series within study folder
seriesNames = getSubfolders(unlabeled_dir);

% Generate training data for all series within study
for series_n = 1:numel(seriesNames)
    % Set up folders
    labSeriesDir = fullfile(labeled_dir,seriesNames{series_n});
    unlabSeriesDir = fullfile(unlabeled_dir,seriesNames{series_n});
    % Label series
    getLabelFrame(labSeriesDir,unlabSeriesDir,saveFileName,'all')
end
end


function subFolders = getSubfolders(dirName)
% Function to get names of subfolders within a folder
allfiles = dir(dirName);
dirflags = [allfiles.isdir];
subdirs = allfiles(dirflags);
if numel(subdirs) < 3
    subFolders = {''};
    return
end
subFolders = cell(numel(subdirs)-2,1);
for i = 3:numel(subdirs); subFolders{i-2} = subdirs(i).name; end
end

function isFmt = checkFormat(f)
% Function to check if file is .csv or .xlsx
tmp = strsplit(f,'.');
fmt = tmp(end);
if max(strcmp(fmt{1},{'csv','xlsx'}))
    isFmt = 1;
else
    isFmt = 0;
end
end