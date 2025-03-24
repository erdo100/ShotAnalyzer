function XLS_Export_Fcn(h, event)
global SA

disp('Writing visible data to XLS ...')
% Save all in to XLS file

hsl = findobj('Tag', 'uitable_shotlist');
sz = size(hsl.Data);

% assign new folder and filename
[pathname,filename,ext] = fileparts(SA.fullfilename);
file0 = fullfile(pathname,filename);

[FileName,PathName,FilterIndex] = uiputfile('*.xlsx','Export to XLS');
if isnumeric(FileName)
    disp('abort')
    return
end

file = [PathName, FileName];
data = [hsl.ColumnName'; hsl.Data];

xlswrite(file, data);

disp('Done with XLS writing')