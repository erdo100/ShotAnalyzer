function plot_shot(ax, Route, lw)
global param plot_setting 

% plot_setting.ball(bi).all
% plot_setting.ball(bi).ball
% plot_setting.ball(bi).line
% plot_setting.ball(bi).marker
% plot_setting.ball(bi).ghostball
% plot_setting.ball(bi).diamondpos
ballcolor1 = 'wyr';
ballcolor2 = {'white', 'yellow', 'red'};
linecolor{1} = [1 1 1];
linecolor{2} = [0.8 0.8 0];
linecolor{3} = [1 0 0];


for bi =1:3
   
   % draw ball
   if plot_setting.ball(bi).ball
      patch(ax, param.ballcirc(1,:)+Route(bi).x(1),...
         param.ballcirc(2,:)+Route(bi).y(1), ballcolor2{bi}, ...
         'Tag',['PlotBall',num2str(bi)],'HitTest','off')
   end
   
   % draw line
   if plot_setting.ball(bi).line
      linesym = '-';
   else
      linesym = '';
   end
   
   % draw marker
   if plot_setting.ball(bi).marker
      markersym = 'o';
   else
      markersym = '';
   end
   
   if plot_setting.ball(bi).line | plot_setting.ball(bi).marker
      % draw routes, line
      plot(ax, Route(bi).x,Route(bi).y, ...
         [linesym, markersym], ...
         'color', [linecolor{bi}], ...
         'MarkerSize',3, ...
         'MarkerEdgeColor', ballcolor1(bi), ...
         'MarkerFaceColor', ballcolor1(bi), ...
         'linewidth', lw(bi), ...
         'Tag',['PlotBallLine',num2str(bi)]);
   end
   
end

