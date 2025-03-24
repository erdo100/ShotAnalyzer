function PlayerFunction(src,event)
global player SA param

fig = get(findobj('Tag','Table_figure'));


lw = player.setting.lw;

% What is currently selected
% to set following 
% SA.Current_si = 1; % Index of current Shot in the Database
% SA.Current_ShotID = SA.Table.ShotID(1); % real number of the Shot
% SA.Current_ti = 1; % Index of the current Shot in the Figure


h = findobj('Tag', 'uitable_shotlist');
shotnum = size(h.Data,1);
 
% data(:,1) = player.Shot(1).ball(1).t;
% data(:,2) = player.Shot(1).ball(1).x;
% data(:,3) = player.Shot(1).ball(1).y;
% data(:,4) = player.Shot(1).ball(2).x;
% data(:,5) = player.Shot(1).ball(2).y;
% data(:,6) = player.Shot(1).ball(3).x;
% data(:,7) = player.Shot(1).ball(3).y;
% xlswrite('BreakShot.xlsx',data);
   
if ishandle(src)
    what2do = src.Tag;
%     player.uptodate = 0;
else
    if isempty(src)
        what2do = 'plotcurrent';
    else
        what2do = src;
    end
end

if isempty(fig) | player.uptodate == 0 | (player.uptodate == 1 & ~strcmp(what2do,'plotcurrent') )
    % initiate table
    player.ax = plot_table;
    player = playerContent(player, what2do);
end


% lets go in to the while loop to plot the shot
check = 1;

switch what2do
    case 'plotcurrent'
        player.video = 0;
        % OK it only single picture, and we make special plot. So I don't
        % go in to the regular plot command within the while loop below
        check = 0;

        % delete old lines
        delete(findobj('-regexp','Tag','PlotBall*'))
        delete(findobj('Tag','myline1'));
        delete(findobj('Tag','myline2'));
        delete(findobj('Tag','myline3'));
        % plot new things
        for si = 1:length(player.Shot)
            player.ti = [length(player.Shot(si).ball(1).t) length(player.Shot(si).ball(2).t) length(player.Shot(si).ball(3).t)];
            if isfield(player.Shot(si),'hit')
                plotPlayer(player.ax, player.Shot(si).ball, player.Shot(si).hit, lw, player.ti, 'last')
            else
                plotPlayer(player.ax, player.Shot(si).ball, [], lw, player.ti, 'last')
            end
        end
        
        if length(SA.Current_si) == 1
            titletxt = [SA.Table.Filename{SA.Current_si}, ...
                ': ShotID ',sprintf('%03d',SA.Table.ShotID{SA.Current_si}),' - ',SA.Table.Player{SA.Current_si}];
            text(player.ax, 0,param.size(1)+200,titletxt,...
                'fontsize',15, 'Tag','myline3','Interpreter','none')
        end

        
    case 'first'
        player.pt_Play.CData = player.icon.play;
        player.pt_Play.State = 'off';
        player.ti = 1;
        player.video = 0;
    case 'fastbackward'
        player.pt_Play.CData = player.icon.play;
        player.pt_Play.State = 'off';
        player.ti = player.ti - 25;
        player.video = 0;
    case 'onebackward'
        player.pt_Play.CData = player.icon.play;
        player.pt_Play.State = 'off';
        player.ti = player.ti - 1;
        player.video = 0;
    case 'play'
        state = src.State;
        player.video = 0;
        
        if strcmp(state,'on')
            src.CData = player.icon.pause;
        else
            src.CData = player.icon.play;
        end
        
    case 'oneforward'
        player.pt_Play.CData = player.icon.play;
        player.pt_Play.State = 'off';
        player.ti = player.ti + 1;
        player.video = 0;
    case 'fastforward'
        player.pt_Play.CData = player.icon.play;
        player.pt_Play.State = 'off';
        player.ti = player.ti + 25;
        player.video = 0;
    case 'last'
        player.pt_Play.CData = player.icon.play;
        player.pt_Play.State = 'off';
        player.ti = [player.Timax player.Timax player.Timax];
        player.video = 0;
    case 'record'
        player.pt_Play.CData = player.icon.pause;
        player.pt_Play.State = 'on';
        player.ti = [1 1 1];
        player.video = 1;
        
        [file,path] = uiputfile('*.mp4','Where to save new animation?');
        if ~isequal(file,0) & ~isequal(path,0)
            player.videofile = [path,file];
        else
            return
        end
        
    case 'record_batch'
        player.pt_Play.CData = player.icon.pause;
        player.pt_Play.State = 'on';
        player.ti = [1 1 1];
        player.video = 1;
        
        
    case 'previous'
        player.pt_Play.CData = player.icon.play;
        player.pt_Play.State = 'off';
        player.ti = [1 1 1];
        player.video = 0;
        % current shot
        if SA.Current_ti(1) > 1
            % set  to new shot
            SA.Current_ti = max([1 SA.Current_ti(1)-1]);

            % Ok which rows are selected and which is the shot in the DB??
            identify_ShotID(SA.Current_ti, h.Data, h.ColumnName)

            player = playerContent(player, what2do);
            delete(findobj('Tag','myline3'));
            
            % Write Title without player name
            titletxt = [SA.Table.Filename{SA.Current_si}, ...
                ': ShotID ',sprintf('%03d',SA.Table.ShotID{SA.Current_si})];
            text(player.ax, 0,param.size(1)+200,titletxt,...
                'fontsize',15, 'Tag','myline3','Interpreter','none')

        end

    case 'next'
        player.pt_Play.CData = player.icon.play;
        player.pt_Play.State = 'off';
        player.ti = [1 1 1];
        player.video = 0;
        if SA.Current_ti(end) < shotnum
            % set  to new shot
            SA.Current_ti = min([shotnum SA.Current_ti(1)+1]);

            % Ok which rows are selected and which is the shot in the DB??
            identify_ShotID(SA.Current_ti, h.Data, h.ColumnName)
            
            player = playerContent(player, what2do);
            delete(findobj('Tag','myline3'));
            
            % Write Title without player name
            titletxt = [SA.Table.Filename{SA.Current_si}, ...
                ': ShotID ',sprintf('%03d',SA.Table.ShotID{SA.Current_si})];
            text(player.ax, 0,param.size(1)+200,titletxt,...
                'fontsize',15, 'Tag','myline3','Interpreter','none')
            
        end

    case 'drawwhitelines'
        player.video = 0;
        plot_MyLine('w', player.Shot(1).ball(1).x(1), player.Shot(1).ball(1).y(1))
        return

    case 'drawyellowlines'
        player.video = 0;
        plot_MyLine('y', player.Shot(1).ball(2).x(1), player.Shot(1).ball(2).y(1))
        return

    case 'drawredlines'
        player.video = 0;
        plot_MyLine('r', player.Shot(1).ball(3).x(1), player.Shot(1).ball(3).y(1))
        return

    case 'replot'
        % just do nothing and replot the current window with new settings

end


% Plot time-velocity diagram
if player.setting.plot_timediagram
    %plot_timediagram
    plot_veldiagram
end


% set current time
Titotalmax = max(player.Timax); % This is necessary to stop after last shot is completed

ti = min([max([1 player.ti]) Titotalmax]);

player.ti = [ti ti ti];

%% Initiate Video
if player.video == 1
    vidname = player.videofile;
    hv = VideoWriter(vidname, 'MPEG-4');
    hv.FrameRate = player.framerate;
    hv.Quality = 95;
    
    disp(['Start: ', vidname])
    
    open(hv);
end

%% update screen or play video
while check 
    % delete old lines
    delete(findobj('-regexp','Tag','PlotBall*'))
    delete(findobj('Tag','myline1'));
    delete(findobj('Tag','myline2'));
    % plot new things
    for si = 1:length(player.Shot)
        if isfield(player.Shot(si),'hit')
            plotPlayer(player.ax, player.Shot(si).ball, player.Shot(si).hit, lw, player.ti,'')
        else
            plotPlayer(player.ax, player.Shot(si).ball, [], lw, player.ti, '')
        end
    end

    % Store the frame
    if player.video == 1
        vid(max(player.ti)).frame = getframe(fig.Number);
    end

    % Check whether recording shall be stopped
    if strcmp(player.pt_Play.State,'on')
        check = 1;
        player.ti = player.ti+1;
        
        if player.ti > Titotalmax
            if player.video == 0 % if we dont make video, than rewind 
                player.ti = [1 1 1];
                
            else
                % we make video, so lets stop it now
                player.ti = [Titotalmax Titotalmax Titotalmax];
                check = 0;
            end
        end
    else
        check = 0;
    end
    
    drawnow
end

%% close video
if player.video == 1
    for ti = 1:Titotalmax
        writeVideo(hv,vid(ti).frame);
    end
    close(hv);
    player.video = 0;
    disp('Video saved.')
    disp('Done.')
end

player.pt_Play.CData = player.icon.play;
player.pt_Play.State = 'off';


