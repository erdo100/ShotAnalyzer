function set_columnfilter
%%
global SA

mainfig = findobj('Tag','figure1');

mainPos = get(mainfig, 'Position');

hpos = [mainPos(1)+30 mainPos(2)-30 450 mainPos(4)+30];
hf = figure('Position', hpos, ...
   'SizeChangedFcn', @set_columnfilter_window_size, ...
   'toolbar','none', 'Menubar','none',  'NumberTitle','off', ...
   'Name','Select data to display shotlist');

% look which columns are visible
colvisible = ismember(SA.Table.Properties.VariableNames, SA.ColumnsVisible);

data = [num2cell(colvisible') SA.Table.Properties.VariableNames'];


ht = uitable(hf, 'Tag', 'Set_columns', ...
   'Position',[5 5  hpos(3)-10 hpos(4)-10], ...
   'ColumnName', {'Enable','Variable'}, ...
   'Data', data(3:end,:), ...
   'ColumnWidth',{100, 200}, ...
   'ColumnEditable', [true false], ...
   'CellEditCallback', @set_columnfilter_data_edit);


function set_columnfilter_window_size(hf,b,c)
%%

hpos = get(hf,'Position');

ht = findobj('Tag', 'Set_columns');
set(ht,'Position', [5 5  hpos(3)-10 hpos(4)-10]);


function set_columnfilter_data_edit(ht,b,c)
%%
global SA

checked = [true true cell2mat(ht.Data(:,1))'];

SA.ColumnsVisible = SA.Table.Properties.VariableNames(checked);

update_ShotList