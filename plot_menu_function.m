function plot_menu_function(h, event)

global player param

plot_only_blue_table_old = get(findobj('Tag','plot_check_bluetable'),'Checked');

% change the tick
if ~strcmp(class(h),'double')
   if h.Checked
      h.Checked = 'off';
   else
      h.Checked = 'on';
   end
end


% Plot the selected shots
player.setting.plot_selected = get(findobj('Tag','plot_selected'),'Checked');

% Plot the velocity
player.setting.plot_timediagram = get(findobj('Tag','plot_timediagram'),'Checked');

% Plot extended Shotline of B1
player.setting.plot_check_extendline = get(findobj('Tag','plot_check_extendline'),'Checked');

% Plot table only blue
player.setting.plot_only_blue_table = get(findobj('Tag','plot_check_bluetable'),'Checked');

%% Ball Settings
color = {'white','yellow','red'};
tags = {'ball', 'line', 'initialpos', 'marker', 'ghostball', 'diamondpos'};

for bi = 1:3
   %% bi Ball
   for tagi = 1:length(tags)
      player.setting.ball(bi).(tags{tagi}) = get(findobj('Tag', ['plot_check_',color{bi},'_',tags{tagi}]),'Checked');
   
   end
end

if player.setting.plot_only_blue_table ~= plot_only_blue_table_old
    close_table_figure
end

% Plot selected things
PlayerFunction([],[])
%PlayerFunction('replot',[])
