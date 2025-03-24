function update_ShotList

global SA

hsl = findobj('Tag', 'uitable_shotlist');


[rowshow, roworder] = ismember(cell2mat(SA.ShotIDsVisible), cell2mat(SA.Table.ShotID)+cell2mat(SA.Table.Mirrored)/10);
[colshow, colorder] = ismember(SA.ColumnsVisible, SA.Table.Properties.VariableNames);

data = table2array(SA.Table(roworder(rowshow),colorder(colshow)));


%% Update ShotList
set(hsl,'ColumnWidth','auto', 'ColumnName', SA.ColumnsVisible, ...
    'Data', data , ...
    'ColumnEditable',true);
