function Read_All_GameData(~,~,mode)
% Read selected files in the Game Source folder
global SA param player

% Handle to ShotList Table
hsl = findobj('Tag', 'uitable_shotlist');

% % Update the sorting of the columns in the ShotList
% update_column_sorting(hsl)

if ~isfield(SA,'fullfilename')
    SA.fullfilename = '..\Gamedata\';
    
else
    if iscell(SA.fullfilename)
        SA.fullfilename = SA.fullfilename{1};
    end
    
end
% request the files to load
[filename, pathname, ext] = uigetfile( ...
   {'*.bbt;*.txt;*.saf', 'All Shotanalyser files (*.bbt,*.txt,*.saf)'; ...
   '*.bbt', 'BilliardBallTracker file (*.bbt)'; ...
   '*.txt', 'MyWebSport file (*.txt)'; ...
   '*.saf', 'ShotAnalyzer file (*.txt, *.saf)'}, ...
   'Pick all files to be analyzed', fullfile(SA.fullfilename), ...
   'MultiSelect', 'on');

% Check inputs
if iscell(filename)
   nfiles = size(filename,2);
   files = filename;
elseif isnumeric(filename)
   disp('abort')
   return
else
   nfiles = 1;
   files{1} = filename;
end

% assign new folder and filename
fullfilename = fullfile(pathname, filename);

%% Delete current data if not in append mode
if mode == 0
    SA = [];
end

%% Read files Depending on file-types
for fi = 1:nfiles
   disp(['Reading ', num2str(fi), '/',num2str(nfiles), ': ', files{fi}])
    [~,~,ext] = fileparts(files{fi});

   switch ext
      case '.bbt'
      % Read the new BilliardBallTracker TXT file
      SAnew = read_BBTfile([pathname,files{fi}]);
      if isempty(SAnew)
         disp('file empty')
         continue
      end
      
      case '.txt'
      % Read the new file
      SAnew = read_gamefile([pathname,files{fi}]);
      if isempty(SAnew)
         disp('file empty')
         continue
      end

      case '.saf'
      %% Read new SAF files
      content = load([pathname,files{fi}],'-mat');
      SAnew = content.SA;
      
       otherwise
         disp(['file not useful:', files{fi}])
   end
   
   %% Append to available data
   SA = append_new_shot_data(SA, SAnew);
   
end

if isempty(SA)
   disp(' empty shot list, noting to do')
   return
end

%% Update Window Title
hf = findobj('Tag','figure1');
if nfiles == 1 & mode == 0
   % read only 1 file and no append.
   % Set figure title
   set(hf, 'Name', [param.ver, pathname, files{fi}(1:end-4),'.saf'])
else
   set(hf, 'Name', [param.ver, pathname, '.saf'])
end


SA.fullfilename = fullfilename;

% Update the GUI
update_ShotList


% Ok which rows are selected and which is the shot in the DB??
identify_ShotID(1, hsl.Data, hsl.ColumnName);

% Plot selected things
player.uptodate = 0;
PlayerFunction('plotcurrent',[])

%% done message
disp(['done. Database has now ', num2str(length(SA.Shot)),' shots'])
