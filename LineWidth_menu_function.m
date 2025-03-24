function LineWidth_menu_function(~, eventdata)
% Reads from source and applies to shots
% if source is available, then column is in source enable
% if source has no column from shot, then shot is disabled

% mcol = uimenu(hfigure,'Text','Columns Selection');
% uimenu(mcol,'Text','Remove all columns','MenuSelectedFcn',@shotlist_menu_function);
% uimenu(mcol,'Text','Show all columns','MenuSelectedFcn',@shotlist_menu_function);
% uimenu(mcol,'Text','Choose columns','MenuSelectedFcn',@shotlist_menu_function);
% uimenu(mcol,'Text','Import columns from XLS','MenuSelectedFcn',@shotlist_menu_function);

global player

prompt = {'White:','Yellow:', 'Red:'};
dlgtitle = 'Line width';
dims = [1 10];
definput = {num2str(player.setting.lw(1)), ...
    num2str(player.setting.lw(2)), ...
    num2str(player.setting.lw(3)) };

answer = inputdlg(prompt,dlgtitle,dims,definput);

player.setting.lw(1) = str2num(answer{1});
player.setting.lw(2) = str2num(answer{2});
player.setting.lw(3) = str2num(answer{3});



%% Update Plot
% Plot selected things
PlayerFunction([],[])
