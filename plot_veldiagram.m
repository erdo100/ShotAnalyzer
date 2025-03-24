function plot_veldiagram
global SA plot_setting param

% What is currently selected
cols='bgr';

%% create figure and axes
ax = findobj('Tag','VelPlot');
if isempty(ax)
    fig = figure('Position', param.TablePosition, ...
        'Tag','VelPlot_figure', 'PaperType','a4', 'PaperOrientation','landscape',...
        'NumberTitle','off', 'Name','Velocity Plot');
    
    addToolbarExplorationButtons(fig)
    ax = axes('Position',[0.1    0.1    0.8    0.8], 'Tag','VelPlot',  'NextPlot', 'add');
    hold(ax, 'on')
else
   delete(findobj('Tag','myline_VelPlot'));
end

%% Create and plot velocity
legtxt={};
legi=0;
for si = SA.Current_si
    [b1b2b3, b1i, b2i, b3i] = str2num_B1B2B3(SA.Table.B1B2B3{si});
    for bi =1:3
        ball(bi).vx = [[diff(SA.Shot(si).Route0(bi).x) ./ diff(SA.Shot(si).Route0(bi).t)]; 0];
        ball(bi).vy = [[diff(SA.Shot(si).Route0(bi).y) ./ diff(SA.Shot(si).Route0(bi).t)]; 0];
        ball(bi).t = SA.Shot(si).Route0(bi).t;
        if b1b2b3(bi) ~= 1
            ball(bi).t = [0; ball(bi).t(2); ball(bi).t(2:end)];
            ball(bi).vx = [0; 0; ball(bi).vx(2:end)];
            ball(bi).vy = [0; 0; ball(bi).vy(2:end)];
        end
        
        ball(bi).v = sqrt(ball(bi).vx.^2 + ball(bi).vy.^2)/1000;
    end
    
    
    for bi =1:3
        plot(ax, ball(b1b2b3(bi)).t, ball(b1b2b3(bi)).v, ['-',cols(bi)],'Tag','myline_VelPlot')
        legi=legi+1;
        legtxt{legi} = ['Shot ',num2str(si),' B',num2str(bi)];
    end
    
end
grid(ax, 'on')
title(ax, 'Balls Speeds')
xlabel(ax,'time in s')
ylabel(ax, 'velocity in m/s')
% legend(ax, legtxt);
