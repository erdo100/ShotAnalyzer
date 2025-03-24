function Table_menu(hf)

global player


%% Plot Menu
mplot = uimenu(hf,'Text','Plot Settings');
% Create child menu items for the menu 'Selection

uimenu(mplot,'Text','Plot Selected','Checked',player.setting.plot_selected,...
   'Tag','plot_selected', 'MenuSelectedFcn',@plot_menu_function);
uimenu(mplot,'Text','Plot time diagram','Checked',player.setting.plot_timediagram,...
   'Tag','plot_timediagram', 'MenuSelectedFcn',@plot_menu_function);
uimenu(mplot,'Text','Extend Shot Line','Checked','off', ...
   'Tag', 'plot_check_extendline', 'MenuSelectedFcn',@plot_menu_function);
uimenu(mplot,'Text','Plot only in blue','Checked',player.setting.plot_only_blue_table, ...
   'Tag', 'plot_check_bluetable', 'MenuSelectedFcn',@plot_menu_function);


% White Ball
uimenu(mplot,'Separator','on','Text','White Ball','Checked',player.setting.ball(1).ball,...
   'Tag', 'plot_check_white_ball', ...
   'MenuSelectedFcn',@plot_menu_function);
uimenu(mplot,'Text','White Ball line','Checked',player.setting.ball(1).line,...
   'Tag', 'plot_check_white_line', ...
   'MenuSelectedFcn',@plot_menu_function);
uimenu(mplot,'Text','White Initial Position','Checked',player.setting.ball(1).initialpos,...
   'Tag', 'plot_check_white_initialpos', 'MenuSelectedFcn',@plot_menu_function);
uimenu(mplot,'Text','White Ball marker','Checked',player.setting.ball(1).marker,...
   'Tag', 'plot_check_white_marker', ...
   'MenuSelectedFcn',@plot_menu_function);
uimenu(mplot,'Text','White Ghostball','Checked',player.setting.ball(1).ghostball, ...
   'Tag', 'plot_check_white_ghostball', ...
   'MenuSelectedFcn',@plot_menu_function);
uimenu(mplot,'Text','White ball Diamond position','Checked',player.setting.ball(1).diamondpos, ...
   'Tag', 'plot_check_white_diamondpos', ...
   'MenuSelectedFcn',@plot_menu_function);


% Yellow Ball
uimenu(mplot,'Separator','on','Text','Yellow Ball','Checked',player.setting.ball(2).ball,...
   'Tag', 'plot_check_yellow_ball', ...
   'MenuSelectedFcn',@plot_menu_function);
uimenu(mplot,'Text','Yellow Ball line','Checked',player.setting.ball(2).line,...
   'Tag', 'plot_check_yellow_line', ...
   'MenuSelectedFcn',@plot_menu_function);
uimenu(mplot,'Text','Yellow Initial Position','Checked',player.setting.ball(2).initialpos,...
   'Tag', 'plot_check_yellow_initialpos', 'MenuSelectedFcn',@plot_menu_function);
uimenu(mplot,'Text','Yellow Ball marker','Checked',player.setting.ball(2).marker,...
   'Tag', 'plot_check_yellow_marker', ...
   'MenuSelectedFcn',@plot_menu_function);
uimenu(mplot,'Text','Yellow Ghostball','Checked',player.setting.ball(2).ghostball, ...
   'Tag', 'plot_check_yellow_ghostball', ...
   'MenuSelectedFcn',@plot_menu_function);
uimenu(mplot,'Text','Yellow ball Diamond position','Checked',player.setting.ball(2).diamondpos, ...
   'Tag', 'plot_check_yellow_diamondpos', ...
   'MenuSelectedFcn',@plot_menu_function);

% Red Ball
uimenu(mplot,'Separator','on','Text','Red Ball','Checked',player.setting.ball(3).ball,...
   'Tag', 'plot_check_red_ball', ...
   'MenuSelectedFcn',@plot_menu_function);
uimenu(mplot,'Text','Red Ball line','Checked',player.setting.ball(3).line,...
   'Tag', 'plot_check_red_line', ...
   'MenuSelectedFcn',@plot_menu_function);
uimenu(mplot,'Text','Red Initial Position','Checked',player.setting.ball(3).initialpos,...
   'Tag', 'plot_check_red_initialpos', 'MenuSelectedFcn',@plot_menu_function);
uimenu(mplot,'Text','Red Ball marker','Checked',player.setting.ball(3).marker,...
   'Tag', 'plot_check_red_marker', ...
   'MenuSelectedFcn',@plot_menu_function);
uimenu(mplot,'Text','Red Ghostball','Checked',player.setting.ball(3).ghostball, ...
   'Tag', 'plot_check_red_ghostball', ...
   'MenuSelectedFcn',@plot_menu_function);
uimenu(mplot,'Text','Red ball Diamond position','Checked',player.setting.ball(3).diamondpos, ...
   'Tag', 'plot_check_red_diamondpos', ...
   'MenuSelectedFcn',@plot_menu_function);


% Other menu
mother = uimenu(hf,'Text','Routes Menu');

% Delete Point
uimenu(mother,'Text','Del Point','MenuSelectedFcn',@ShotEditDelete_menu_function);
% Linethickness
uimenu(mother,'Text','Set route thickness','MenuSelectedFcn',@LineWidth_menu_function);


%% Toolbar
tb = findall(hf,'Type','uitoolbar');
%tb = uitoolbar(hfigure);

% To first
player.pt_First = uipushtool(tb);
player.pt_First.Separator = 'on';
player.pt_First.CData = player.icon.first;
player.pt_First.Tooltip = 'Go to first time step';
player.pt_First.Tag = 'first';
player.pt_First.ClickedCallback = @PlayerFunction;

% big step back
player.pt_FastBackward = uipushtool(tb);
player.pt_FastBackward.CData = player.icon.fastbackward;
player.pt_FastBackward.Tooltip = '1 second back';
player.pt_FastBackward.Tag = 'fastbackward';
player.pt_FastBackward.ClickedCallback = @PlayerFunction;

% 1 step back
player.pt_OneBack = uipushtool(tb);
player.pt_OneBack.CData = player.icon.onebackward;
player.pt_OneBack.Tooltip = '1 step back';
player.pt_OneBack.Tag = 'onebackward';
player.pt_OneBack.ClickedCallback = @PlayerFunction;

% play
player.pt_Play = uitoggletool(tb);
player.pt_Play.CData = player.icon.play;
player.pt_Play.Tooltip = 'Start/Stop play';
player.pt_Play.Tag = 'play';
player.pt_Play.ClickedCallback = @PlayerFunction;

% 1 step forward
player.pt_OneForward = uipushtool(tb);
player.pt_OneForward.CData = player.icon.oneforward;
player.pt_OneForward.Tooltip = '1 step forward';
player.pt_OneForward.Tag = 'oneforward';
player.pt_OneForward.ClickedCallback = @PlayerFunction;

% big step forward
player.pt_FastForward = uipushtool(tb);
player.pt_FastForward.CData = player.icon.fastforward;
player.pt_FastForward.Tooltip = '1 second forward';
player.pt_FastForward.Tag = 'fastforward';
player.pt_FastForward.ClickedCallback = @PlayerFunction;

% to Last
player.pt_Last = uipushtool(tb);
player.pt_Last.CData = player.icon.last;
player.pt_Last.Tooltip = 'Go to end';
player.pt_Last.Tag = 'last';
player.pt_Last.ClickedCallback = @PlayerFunction;

% previous
player.pt_Previous = uipushtool(tb);
player.pt_Previous.Separator = 'on';
player.pt_Previous.CData = player.icon.previous;
player.pt_Previous.Tooltip = 'Go to previous shot';
player.pt_Previous.Tag = 'previous';
player.pt_Previous.ClickedCallback = @PlayerFunction;

% next
player.pt_Next = uipushtool(tb);
player.pt_Next.CData = player.icon.next;
player.pt_Next.Tooltip = 'Go to next shot';
player.pt_Next.Tag = 'next';
player.pt_Next.ClickedCallback = @PlayerFunction;

% record
player.pt_Record = uipushtool(tb);
player.pt_Record.CData = player.icon.record;
player.pt_Record.Tooltip = 'Record to file';
player.pt_Record.Tag = 'record';
player.pt_Record.ClickedCallback = @PlayerFunction;

% CurrentBall
ttImage = zeros(16,16,3);
ttImage(:,:,3) = ones(16);
tt.Icon = ttImage;
player.pt_CurrentBall = uipushtool(tb);
player.pt_CurrentBall.Separator = 'on';
player.pt_CurrentBall.CData = player.icon.WhiteBall;
player.pt_CurrentBall.Tooltip = 'Color of current cue ball';
player.pt_CurrentBall.Tag = 'CurrentBall';

% DrawWhiteLines
player.pt_DrawWhiteLines = uipushtool(tb);
player.pt_DrawWhiteLines.CData = player.icon.drawwhitelines;
player.pt_DrawWhiteLines.Tooltip = 'Draw annotions line for white ball';
player.pt_DrawWhiteLines.Tag = 'drawwhitelines';
player.pt_DrawWhiteLines.ClickedCallback = @PlayerFunction;

% DrawYellowLines
player.pt_DrawYellowLines = uipushtool(tb);
player.pt_DrawYellowLines.CData = player.icon.drawyellowlines;
player.pt_DrawYellowLines.Tooltip = 'Draw annotions line for white ball';
player.pt_DrawYellowLines.Tag = 'drawyellowlines';
player.pt_DrawYellowLines.ClickedCallback = @PlayerFunction;

% DrawRedLine
player.pt_DrawRedLines = uipushtool(tb);
player.pt_DrawRedLines.CData = player.icon.drawredlines;
player.pt_DrawRedLines.Tooltip = 'Draw annotions line for white ball';
player.pt_DrawRedLines.Tag = 'drawredlines';
player.pt_DrawRedLines.ClickedCallback = @PlayerFunction;


