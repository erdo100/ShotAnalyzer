function Extract_b1b2b3_start(~,~)

global SA player

disp(['start (',mfilename,')'])
varname = 'B1B2B3';

hsl = findobj('Tag', 'uitable_shotlist');
sz = size(hsl.Data);

for ti = 1:sz(1)
    identify_ShotID(ti, hsl.Data, hsl.ColumnName)
    si = SA.Current_si;
    if SA.Table.Interpreted{si} == 0
        % Make calculation
        [b1b2b3, err] = Extract_b1b2b3(SA.Shot(si));
        
        if ~isempty(err.code)
            SA.Table.Selected{si} = true;
            disp([num2str(si),': ',err.text]);
            
        end
        
        % Update Table
        SA.Table.(varname){si} = b1b2b3;
        
        % Update the err
        SA.Table.ErrorID{si} = err.code;
        SA.Table.ErrorText{si} = err.text;
        
    end
    
end

% update GUI
update_ShotList
player.uptodate = 0;

disp([num2str(sum(cell2mat(SA.Table.Selected))),'/',num2str(length(SA.Table.Selected)),' shots selected'])
disp(['done (',mfilename,')'])