function ShotAnalyzer

%% Lets start with general definitions
global param player SA

%% Shot analysis base path and file name
SA.fullfilename = '..\Gamedata\';

%% Table properties
param.ver  = 'Shot Analyzer v0.43i';

param.size = [1420 2840];
param.ballR = 61.5/2;
param.ballcirc(1,:) = sin(linspace(0,2*pi,36))*param.ballR;
param.ballcirc(2,:) = cos(linspace(0,2*pi,36))*param.ballR;

param.rdiam = 7;
param.cushionwidth = 50;
param.diamdist = 97;
param.framewidth = 147;

param.colors = 'wyr';

param.BallOutOfTableDetectionLimit = 30;
param.BallCushionHitDetectionRange = 50;
param.BallProjecttoCushionLimit = 10;
param.NoDataDistanceDetectionLimit = 600;
param.MaxTravelDistancePerDt = 0.1;
param.MaxVelocity = 12000;
param.timax_appr = 5;

param.with_title = 0; % Controls whether we have to plot in table the title or not.

%% Player Setting
% delta time within the player
player.framerate = 30;
player.dt = 1/player.framerate;

%% read icons
if ~isfolder('icons')
    disp(['Message from Ersin. ',...
        'Cemal, please copy the ICONS folder ', ...
        'in the location where the *.exe is executed. Kolay gelsin'])
end

[img,map] = imread('icons\first_16.gif');
player.icon.first = ind2rgb(img,map);

[img,map] = imread('icons\fast_backward_16.gif');
player.icon.fastbackward = ind2rgb(img,map);

[img,map] = imread('icons\back_pause_16.gif');
player.icon.onebackward = ind2rgb(img,map);

[img,map] = imread('icons\play_16.gif');
player.icon.play = ind2rgb(img,map);

[img,map] = imread('icons\pause_24.gif');
player.icon.pause = ind2rgb(img,map);

[img,map] = imread('icons\forward_pause_16.gif');
player.icon.oneforward = ind2rgb(img,map);

[img,map] = imread('icons\fast_forward_16.gif');
player.icon.fastforward = ind2rgb(img,map);

[img,map] = imread('icons\last_16.gif');
player.icon.last = ind2rgb(img,map);

[img,map] = imread('icons\record_16.gif');
player.icon.record = ind2rgb(img,map);

[img,map] = imread('icons\previous_16.gif');
player.icon.previous = ind2rgb(img,map);

[img,map] = imread('icons\next_16.gif');
player.icon.next = ind2rgb(img,map);

ttImage = ones(16,16,3);
player.icon.WhiteBall = ttImage;

ttImage = zeros(16,16,3);
ttImage(:,:,1) = ones(16);
ttImage(:,:,2) = ones(16);
player.icon.YellowBall = ttImage;

ttImage = zeros(16,16,3);
ttImage(:,:,1) = ones(16);
player.icon.RedBall = ttImage;

[img,map] = imread('icons\draw_white_lines_16.gif');
player.icon.drawwhitelines = ind2rgb(img,map);
[img,map] = imread('icons\draw_yellow_lines_16.gif');
player.icon.drawyellowlines = ind2rgb(img,map);
[img,map] = imread('icons\draw_red_lines_16.gif');
player.icon.drawredlines = ind2rgb(img,map);

player.uptodate =0;


% Player display settings      
player.setting.plot_selected = false;
player.setting.plot_timediagram = false;
player.setting.plot_check_extendline = false;
player.setting.plot_only_blue_table = 'off';
player.setting.lw = ones(1,3)*7;

player.setting.ball(1).ball = true;
player.setting.ball(1).line = true;
player.setting.ball(1).initialpos = true;
player.setting.ball(1).marker = false;
player.setting.ball(1).ghostball = true;
player.setting.ball(1).diamondpos = false;
player.setting.ball(2).ball = true;
player.setting.ball(2).line = true;
player.setting.ball(2).initialpos = true;
player.setting.ball(2).marker = false;
player.setting.ball(2).ghostball = false;
player.setting.ball(2).diamondpos = false;
player.setting.ball(3).ball = true;
player.setting.ball(3).line = true;
player.setting.ball(3).initialpos = true;
player.setting.ball(3).marker = false;
player.setting.ball(3).ghostball = false;
player.setting.ball(3).diamondpos = false;

%% Figure Position
root = groot;
width = root.ScreenSize(3)*0.38;
height = width/2+177;
posx = 10;
posy = root.ScreenSize(4)-height-10;
figurepos = [posx posy width height];
hfigure = figure('Tag','figure1','Outerposition',figurepos, ...
    'Name', param.ver, 'Menubar','none', 'Toolbar','none', ...
    'Filename','', 'NumberTitle','off', ...
    'SizeChangedFcn', @figure1_SizeChangedFcn, ...
    'CloseRequestFcn',@figure1_CloseFcn);



%% Create Figure menu
figure_menu(hfigure)


%% Create Table
htable_shotlist = uitable(hfigure, 'Tag', 'uitable_shotlist', ...
    'Position', [5 5 hfigure.Position(3)-20 hfigure.Position(4)-40], ...
    'CellEditCallback' , @uitable_shotlist_CellEditCallback, ...
    'CellSelectionCallback', @uitable_shotlist_CellSelectionCallback);


% Start opening File
Read_All_GameData(0,0,0)


%
% =================================================================================================
% Window Resizing FUNCTIONS
% =================================================================================================
function figure1_SizeChangedFcn(hObject, eventdata)
hpos = get(hObject,'Position');
htable = findobj( 'Tag','uitable_shotlist');

set(htable,'Position', [6 6 hpos(3)-11 hpos(4)-20])


%
% =================================================================================================
% PLOT FUNCTIONS
% =================================================================================================
%
% --- Executes when selected cell(s) is changed in uitable_shotlist.
function uitable_shotlist_CellSelectionCallback(hObject, eventdata)
% hObject    handle to uitable_shotlist (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.CONTROL.TABLE)
%	Indices: row and column indices of the cell(s) currently selecteds
% handles    structure with handles and user data (see GUIDATA)
global player SA

% to set following 
% SA.Current_si = 1; % Index of current Shot in the Database
% SA.Current_ShotID = SA.Table.ShotID(1); % real number of the Shot
% SA.Current_ti = 1; % Index of the current Shot in the Figure
if ~isempty(eventdata.Indices)
    % Ok which rows are selected and which is the shot in the DB??
    identify_ShotID(eventdata.Indices(:,1), hObject.Data, hObject.ColumnName);
end

%plot_selection
player.uptodate = 0;
PlayerFunction('plotcurrent',[])


% --- Executes when entered data in editable cell(s) in uitable_shotlist.
function uitable_shotlist_CellEditCallback(hObject, eventdata)
% hObject    handle to uitable_shotlist (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.CONTROL.TABLE)
%	Indices: row and column indices of the cell(s) edited
%	PreviousData: previous data for the cell(s) edited
%	EditData: string(s) entered by the user
%	NewData: EditData or its converted form set on the Data property. Empty if Data was not changed
%	Error: error string when failed to convert EditData to appropriate value for Data
% handles    structure with handles and user data (see GUIDATA)
global SA player

% to set following 
% SA.Current_si = 1; % Index of current Shot in the Database
% SA.Current_ShotID = SA.Table.ShotID(1); % real number of the Shot
% SA.Current_ti = 1; % Index of the current Shot in the Figure

ti = eventdata.Indices(1);
ci = eventdata.Indices(2);
ColumnName = get(hObject,'ColumnName');
Data = get(hObject,'Data');

coli = find(strcmp(ColumnName(ci), SA.Table.Properties.VariableNames),1);

% which rows?
col_si = strcmp('ShotID', ColumnName);
col_mi = strcmp('Mirrored', ColumnName);

ShotID = cell2mat(Data(ti, col_si));
mirrored = cell2mat(Data(ti, col_mi));

[~, si, ~] = intersect(cell2mat(SA.Table.ShotID)+cell2mat(SA.Table.Mirrored)/10, ShotID+mirrored/10);

SA.Current_ti = ti;
SA.Current_ShotID = SA.Table.ShotID(si);
SA.Current_si = si;


% Assign new edited value
SA.Table.(SA.Table.Properties.VariableNames{coli})(si) = {eventdata.NewData};


%plot_selection
player.uptodate = 0;
PlayerFunction('plotcurrent',[])

