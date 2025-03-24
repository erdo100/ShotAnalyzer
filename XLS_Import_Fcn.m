function XLS_Import_Fcn(h, event)
global SA player

disp('Start importing visible data from XLS ...')

%% tasks to be done
% find new columns ==> append them to the SA.Table
% update values in SA.Table, only change of values, no deletion of rows
% hiding of lines or colums are done by visiblity variable during
% UpdateTable function call

%% Read all from XLS file
[pathname,filename,ext] = fileparts(SA.fullfilename);

file0 = fullfile(pathname,filename);

[FileName,PathName,FilterIndex] = uigetfile('*.xlsx','Import from XLS');
if isnumeric(FileName)
   disp('abort')
   return
end

file = [PathName, FileName];

% Read Excel file
[num, txt, raw] = xlsread(file);


%% Try to match all shot based on shotID
% Find Column where ShotID is written in XLS
new_ci_Sel = find(strcmp(raw(1,:), 'Selected'));
new_ci_ShotID = find(strcmp(raw(1,:), 'ShotID'));
new_ci_Mirrored = find(strcmp(raw(1,:), 'Mirrored'));

if isempty(new_ci_Sel) 
    disp('abort: In XLS no Column <Selected> found')
   return
end

if isempty(new_ci_ShotID) 
   disp('abort: In XLS no Column <ShotID> found')
   return
end

if isempty(new_ci_Mirrored) 
   disp('abort: In XLS no Column <Mirrored> found')
   return
end


%% Prepare data
% column names
table_names = SA.Table.Properties.VariableNames;
new_names = raw(1,:);

% Shot IDs
table_shotid = cell2mat(SA.Table.ShotID);
new_shotid = cell2mat(raw(2:end,new_ci_ShotID));

% Mirrored
table_mirrored = cell2mat(SA.Table.Mirrored);
new_mirrored = cell2mat(raw(2:end,new_ci_Mirrored));

%% find columns to be edited
err = 0;
for ci = 1:length(new_names)
    % Check whether column has name entry in XLS
    if isnan(new_names{ci})
        disp(['ERROR: Column number ', num2str(ci), ' has no name entry.'])
        err = err + 1;
    end
end

if err > 0
    disp(['Check the XLS file care fully.'])
    disp(file)
    disp('Aborting XLS import.')
    return
end

% which Columns to be shown
common_columns = intersect(new_names, table_names,'stable');

% build unique ShotIds
table_uniqueshotid = table_shotid + table_mirrored/10;
new_uniqueshotid = new_shotid + new_mirrored/10;


% common_shotid Shot IDs which are common
[common_shotid, si_new, si_table] = intersect(new_uniqueshotid, table_uniqueshotid,'stable');
% si_new = index of new_shotid for common shots ==> This is the important thing
% together with new_shotid.
if isempty(common_shotid)
    disp('No common Shots in Database found matching to shots in XLS. Aborting.')
    return
end


%% 1 Update the DataBase in SA.Table
% Loop over all columns_names
% Important: Here we cant add additional rows, this is not allowed, since
% ShotData is not edited

% Now we know how many rowswe have and which have to be updated.
% Lets find which columns we have to create/edit ==> newnames

for ni = 1:length(new_names)
    
    if ismember(new_names{ni}, table_names)
        coldata = (SA.Table.(new_names{ni}));
        newcol = 0;
        
    else
        % this column is not available therefore must be created
        coldata = cell(length(table_shotid),1);
        % we remember which position it was. In the table these variables
        % are stores at last. But we want to store then in the correct
        % position. Otherwise they are at the last position when we select
        % the columnsselector
        newcol = 1;
    end
    coldata(si_table) = (raw(1+si_new, ni));
    
    SA.Table.(new_names{ni}) = (coldata);

    if newcol
        % we sort them now
        sortind = [1:ni-1 length(SA.Table.Properties.VariableNames) ni:length(SA.Table.Properties.VariableNames)];
        SA.Table = SA.Table(:,sortind);

    end
end


%% update the GUI
SA.ShotIDsVisible = num2cell(common_shotid);
SA.ColumnsVisible = common_columns;

update_ShotList

% Ok which rows are selected and which is the shot in the DB??
hsl = findobj('Tag', 'uitable_shotlist');
identify_ShotID(1, hsl.Data, hsl.ColumnName);

% Plot selected things
player.uptodate = 0;
PlayerFunction('plotcurrent',[])


disp('Done with XLS importing')