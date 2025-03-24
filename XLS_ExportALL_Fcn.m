function XLS_Export_Fcn(h, event)
global SA

disp('Writing all Data to XLS ...')
% Save all in to XLS file
hf = findobj('Tag','figure1');
file0 = get(hf, 'Name');

[FileName,PathName,FilterIndex] = uiputfile('*.xlsx','Export to XLS', file0(1:end-4));
if isnumeric(FileName)
   disp('abort')
   return
end

file = [PathName, FileName];

data = [SA.Table.Properties.VariableNames; ...
   table2cell(SA.Table)];

xlswrite(file, data);

disp('Done with XLS writing')