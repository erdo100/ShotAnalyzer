function user_defined_column(h, event)
global SA param


prompt = {'Enter new column name:'};
title = 'Input';
dims = [1 35];
definput = {'UserDefinedColumn'};
answer = inputdlg(prompt,title,dims,definput);

if isempty(answer)
   disp('User abort')
   return
end

% check if name is already in use?
if find(strcmp(SA.Table.Properties.VariableNames, answer),1)
   disp('Name is alread in use. aborting')
   return
end

sz = size(SA.Table);


% Update ShotListCol
SA.Table.(answer{1}) = cell(sz(1),1);
SA.Table = SA.Table(:,[1 2 end [3:end-1]]);

SA.ColumnsDisplay(end+1) = true;
SA.ColumnsDisplay = SA.ColumnsDisplay([1 2 end [3:end-1]]);

SA.ColumnsVisible{end+1} = cell(sz(1),1);

SA.ColumnsVisible = SA.ColumnsVisible([1 2 end [3:end-1]]);

% Update the Shotlist
update_ShotList
