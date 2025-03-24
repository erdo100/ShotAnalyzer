function selection_menu_function(~, eventdata)

global SA

% uimenu(msel,'Text','Import from XLS','MenuSelectedFcn',@selection_menu_function);
% uimenu(msel,'Text','Select marked shots','MenuSelectedFcn',@selection_menu_function);
% uimenu(msel,'Text','Unselect marked shots','MenuSelectedFcn',@selection_menu_function);
% uimenu(msel,'Text','Delete selected shots','MenuSelectedFcn',@selection_menu_function);
% uimenu(msel,'Text','Select all shots','MenuSelectedFcn',@selection_menu_function);
% uimenu(msel,'Text','Unselect all shots','MenuSelectedFcn',@selection_menu_function);

h = findobj('Tag', 'uitable_shotlist');

marked = find(cell2mat(h.Data(:,1)));
nonmarked = find(cell2mat(h.Data(:,1))== false);

% Tableidcoli = find(SA.Table.Properties.VariableNames == 'ShotID');

col_si = strcmp('ShotID', h.ColumnName);
col_mi = strcmp('Mirrored', h.ColumnName);

for i = 1:length(marked)
    ShotID = cell2mat(h.Data(marked(i), col_si));
    mirrored = cell2mat(h.Data(marked(i), col_mi));
    
    [~, si, ~] = intersect(cell2mat(SA.Table.ShotID)+cell2mat(SA.Table.Mirrored)/10, ShotID+mirrored/10);
    
    markedDB(i) = si;
end

switch eventdata.Source.Text
    case 'Select marked shots'
        
        SA.Table.Selected(SA.Current_ti) = {true};
        
    case 'Unselect marked shots'
        
        SA.Table.Selected(SA.Current_ti) = {false};
        
    case 'Invert Selection'
        if ~isempty(marked)
            SA.Table.Selected(marked) = {false};
            SA.Table.Selected(nonmarked) = {true};
        else
            disp('Nothing marked, nothing to invert.')
        end
        
    case 'Unselect all shots'
        SA.Table.Selected = num2cell(false(size(SA.Table,1),1));
        
    case 'Select all shots'
        SA.Table.Selected = num2cell(true(size(SA.Table,1),1));
        
    case 'Delete selected shots'
        
        if ~isempty(marked) & ~isempty(markedDB)
            
            % delete rows which have a tick
            SA.Shot(markedDB) = [];
            SA.Table(markedDB,:) = [];
            SA.ShotIDsVisible(marked) = [];
            if length(SA.Shot) < SA.Current_si
                SA.Current_si = 1;
            end
            disp('Delete completed.')
        else
            disp('Nothing marked, nothing to hide.')
        end
        
    case 'Hide selected shots'
        
        if ~isempty(marked)
            % hide rows which have a tick
            SA.ShotIDsVisible(marked) = [];
            disp('hide completed.')
        else
            disp('Nothing marked, nothing to hide.')
        end
        
end

% update GUI
update_ShotList

% Plot selected things
PlayerFunction([],[])