function AddMirroredPositions(~,~)
global SA param

disp(['Start mirroring ...'])
% What do do here?
% Mirror X
% Mirror Y
% Miroor X & Y
% Add for column to document the mirror action
hsl = findobj('Tag', 'uitable_shotlist');
sz = size(hsl.Data);

% Add SAnew.Shot(si)
% Add SAnew.Table(si)
% Add SAnew.ShotIDsVisible(si)
% Copy SAnew.ColumnsVisible
% Copy SAnew.Current_si
% Copy SAnew.Current_ShotID
% Copy SAnew.Current_ti
% Copy SAnew.fullfilename

% Inititalize shot counter for SAnew 
sinew = 0;

col_si = strcmp('ShotID', hsl.ColumnName);
col_mi = strcmp('Mirrored', hsl.ColumnName);

% Loop over all shots in the list
for ti = 1:sz(1)
    
    ShotID = cell2mat(hsl.Data(ti, col_si));
    mirrored = cell2mat(hsl.Data(ti, col_mi));
    
    [~, si, ~] = intersect(cell2mat(SA.Table.ShotID)+cell2mat(SA.Table.Mirrored)/10, ShotID+mirrored/10);
    Shot_Mirror_ID = ShotID + mirrored/10;
    
    if SA.Table.Mirrored{si} ~= 0
        % Shots are already mirrored, just copy the shot
            sinew = sinew + 1;

            % copy whole line in the Table
            SAnew.Table(sinew,:) = SA.Table(si,:);
            SAnew.ShotIDsVisible(sinew,1) = SA.ShotIDsVisible(si);

    else        
        % mirror Shots 
        
        for i = 1:4
            sinew = sinew + 1;

            % copy whole line in the Table
            SAnew.Table(sinew,:) = SA.Table(si,:);
            
            if i == 1
                % copy original data
                SAnew.Shot(sinew) = SA.Shot(si);
                SAnew.Table.Mirrored{sinew} = 1;
                
            elseif i == 2
                % Mirror LONG cushion values
                for bi = 1:3
                    SAnew.Shot(sinew).Route0(bi).t = SA.Shot(si).Route0(bi).t;
                    SAnew.Shot(sinew).Route(bi).t = SA.Shot(si).Route(bi).t;
                    SAnew.Shot(sinew).Route0(bi).x = param.size(2) - SA.Shot(si).Route0(bi).x;
                    SAnew.Shot(sinew).Route(bi).x = param.size(2) - SA.Shot(si).Route(bi).x;
                    SAnew.Shot(sinew).Route0(bi).y = SA.Shot(si).Route0(bi).y;
                    SAnew.Shot(sinew).Route(bi).y = SA.Shot(si).Route(bi).y;
                end
                SAnew.Table.Mirrored{sinew} = 2;
                
            elseif i == 3
                % Mirror SHORT cushion values
                for bi = 1:3
                    SAnew.Shot(sinew).Route0(bi).t = SA.Shot(si).Route0(bi).t;
                    SAnew.Shot(sinew).Route(bi).t = SA.Shot(si).Route(bi).t;
                    SAnew.Shot(sinew).Route0(bi).x = SA.Shot(si).Route0(bi).x;
                    SAnew.Shot(sinew).Route(bi).x = SA.Shot(si).Route(bi).x;
                    SAnew.Shot(sinew).Route0(bi).y = param.size(1) - SA.Shot(si).Route0(bi).y;
                    SAnew.Shot(sinew).Route(bi).y = param.size(1) - SA.Shot(si).Route(bi).y;
                end
                SAnew.Table.Mirrored{sinew} = 3;

            elseif i == 4
                % Mirror SHORT & LONG cushion values
                for bi = 1:3
                    SAnew.Shot(sinew).Route0(bi).t = SA.Shot(si).Route0(bi).t;
                    SAnew.Shot(sinew).Route(bi).t = SA.Shot(si).Route(bi).t;
                    SAnew.Shot(sinew).Route0(bi).x = param.size(2) - SA.Shot(si).Route0(bi).x;
                    SAnew.Shot(sinew).Route(bi).x = param.size(2) - SA.Shot(si).Route(bi).x;
                    SAnew.Shot(sinew).Route0(bi).y = param.size(1) - SA.Shot(si).Route0(bi).y;
                    SAnew.Shot(sinew).Route(bi).y = param.size(1) - SA.Shot(si).Route(bi).y;
                end
                SAnew.Table.Mirrored{sinew} = 4;
                
            end

            SAnew.ShotIDsVisible{sinew,1} = ShotID + SAnew.Table.Mirrored{sinew}/10;

        end

    end
end

% save old data to new
SAnew.ColumnsVisible = SA.ColumnsVisible;
SAnew.Current_si = SA.Current_si;
SAnew.Current_ti = SA.Current_ti;
SAnew.fullfilename = SA.fullfilename;

% Overwrite SA
 SA = SAnew;

% Update the GUI
update_ShotList

% Plot selected things
player.uptodate = 0;
PlayerFunction('plotcurrent',[])

%% done message
disp(['Mirroring done. Database has now ', num2str(length(SA.Shot)),' shots'])
