function shotlist_menu_function(~, eventdata)
% Reads from source and applies to shots
% if source is available, then column is in source enable
% if source has no column from shot, then shot is disabled

% mcol = uimenu(hfigure,'Text','Columns Selection');
% uimenu(mcol,'Text','Remove all columns','MenuSelectedFcn',@shotlist_menu_function);
% uimenu(mcol,'Text','Show all columns','MenuSelectedFcn',@shotlist_menu_function);
% uimenu(mcol,'Text','Choose columns','MenuSelectedFcn',@shotlist_menu_function);
% uimenu(mcol,'Text','Import columns from XLS','MenuSelectedFcn',@shotlist_menu_function);

global SA

hsl = findobj('Tag', 'uitable_shotlist');
hcb = findobj('Tag', 'checkbox_plot_selected');

sld = get(hsl,'UserData');

cols = size(SA.Table,2);

switch eventdata.Source.Text
    case 'Hide all columns'
        SA.ColumnsVisible = SA.Table.Properties.VariableNames([true true false(1,cols-2)]);
        disp('done with hiding Columns')
        
    case 'Show all columns'
        SA.ColumnsVisible = SA.Table.Properties.VariableNames;
        disp('done with showing all Columns')
        
    case 'Choose columns'
        set_columnfilter
        
    case 'Delete columns'
        
    case 'Import column names to display from XLS'
        [filename, pathname, ext] = uigetfile('*.xlsx','Load columns setting (*.xlsx)');
        
        if isnumeric(filename)
            disp('abort')
            return
        end
        
        [num, txt, raw] = xlsread([pathname, filename]);
        new_names = raw(1,:);
        
        err = 0;
        warn = 0;
        % Check the content for plausibility
        for ci = 1:length(new_names)
            if isnan(new_names{ci})
                disp(['Error: Column number ', num2str(ci), ' has no name entry.'])
                err = err + 1;
            end
            
            if isempty(find(strcmp(new_names{ci},SA.Table.Properties.VariableNames),1))
                disp(['WARNING: Column', new_names{ci}, ' is not available in the database.'])
                disp('I skip empty columns and continue with import.')
                warn = warn + 1;
            end
        end

        if err > 0
            disp(['Check the XLS file care fully.'])
            disp([pathname, filename])
            disp('Import aborted.')
            return
        end

        % Columns to be edited
        [common_name, ni_new, ni_current] = intersect(new_names,SA.Table.Properties.VariableNames,'stable');
        
        SA.ColumnsVisible = common_name;
        disp('done with Loading Columns from XLS')
        if warn > 0 & err == 0
            disp([pathname, filename])
        end
        
        
end

%% Update ShotList
update_ShotList
