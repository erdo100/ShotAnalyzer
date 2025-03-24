function Extract_b1b2b3_position(a,b)
global SA param player

disp('Start identifying B1B2B3 Positions ...')

hsl = findobj('Tag', 'uitable_shotlist');
sz = size(hsl.Data);

for ti = 1:sz(1)
    identify_ShotID(ti, hsl.Data, hsl.ColumnName)
    si = SA.Current_si;

    if SA.Table.Interpreted{si} == 0
        
        [b1b2b3, b1i, b2i, b3i] = str2num_B1B2B3(SA.Table.B1B2B3{si});
        
        % Create fields
        for bi = 1:3
            
            % Make calculation
            % ===============EDIT THIS HERE ===============
            % First check is column already available, then update visibility
            
            %% PosX
            varname = ['B',num2str(bi),'posX'];
            ind = find(strcmp(SA.Table.Properties.VariableNames, varname));
            SA.Table.(varname){si} = SA.Shot(si).Route(b1b2b3(bi)).x(1)/param.size(2)*8;
            
            
            %% PosY
            varname = ['B',num2str(bi),'posY'];
            % First check is column already available, then update visibility
            ind = find(strcmp(SA.Table.Properties.VariableNames, varname));
            SA.Table.(varname){si} = SA.Shot(si).Route(b1b2b3(bi)).y(1)/param.size(1)*4;
            
            err.code = [];
            err.text = [];
            
            % Update the err
            % First check is column already available, then update visibility
            ind = find(strcmp(SA.Table.Properties.VariableNames, 'ErrorID'));
            SA.Table.ErrorID{si} = err.code;
            
            ind = find(strcmp(SA.Table.Properties.VariableNames, 'ErrorText'));
            SA.Table.ErrorText{si} = err.text;
            
        end
    end
end
% update GUI
update_ShotList
player.uptodate = 0;
PlayerFunction('plotcurrent',[])

disp(['done (',mfilename,')'])