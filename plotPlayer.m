function plotPlayer(ax, Ball, hit, lw, ti, plotlast)
global param player SA

% player.setting.ball(bi).all
% player.setting.ball(bi).ball
% player.setting.ball(bi).line
% player.setting.ball(bi).marker
% player.setting.ball(bi).ghostball
% player.setting.ball(bi).diamondpos
ballcolor1 = 'wyr';
ballcolor2 = {'white', 'yellow', 'red'};
linecolor{1} = [1 1 1];
linecolor{2} = [0.8 0.8 0];
linecolor{3} = [1 0 0];
offsetDX = [0 param.size(2)+param.diamdist 0 -param.diamdist];
offsetDY = [-param.diamdist 0 param.size(1)+param.diamdist 0];
offsetCX = [0 param.size(2)-param.ballR 0 param.ballR];
offsetCY = [param.ballR 0 param.size(1)-param.ballR 0];
scaleD = 1+0*8/param.size(2);


for bi =1:3
%     disp('plotPlayer.m')
%     player.setting.ball(bi)
    %% draw ball
    if player.setting.ball(bi).ball & ~strcmp(plotlast,'last')
        % Ball to be plotted and we plot correct position of ball based on time
        patch(ax, param.ballcirc(1,:)+Ball(bi).x(ti(bi)),...
            param.ballcirc(2,:)+Ball(bi).y(ti(bi)), ballcolor2{bi}, ...
            'Tag',['PlotBall',num2str(bi)])
        
    elseif player.setting.ball(bi).ball & strcmp(plotlast,'last')
        % Ball to be plotted and we plot balls at initial position
        patch(ax, param.ballcirc(1,:)+Ball(bi).x(1),...
            param.ballcirc(2,:)+Ball(bi).y(1), ballcolor2{bi}, ...
            'Tag',['PlotBall',num2str(bi)])
    end

    %% draw initial position
    if player.setting.ball(bi).initialpos
        plot(ax, param.ballcirc(1,:)+Ball(bi).x(1), ...
            param.ballcirc(2,:)+Ball(bi).y(1), ballcolor2{bi}, 'Tag','myline2','HitTest','off')
    end

    %% draw line
    if player.setting.ball(bi).line
        linesym = '-';
    else
        linesym = '';
    end
    
    %% draw marker
    if player.setting.ball(bi).marker
        markersym = 'o';
    else
        markersym = '';
    end
    
    if player.setting.ball(bi).line | player.setting.ball(bi).marker
        % draw routes, line
        plot(ax, Ball(bi).x(1:ti(bi)),Ball(bi).y(1:ti(bi)), ...
            [linesym, markersym], ...
            'color', [linecolor{bi}], ...
            'MarkerSize',3, ...
            'MarkerEdgeColor', ballcolor1(bi), ...
            'MarkerFaceColor', ballcolor1(bi), ...
            'linewidth', lw(bi), ...
            'Tag',['PlotBallLine',num2str(bi)]);
    end
    
    
    % draw hits
    if isstruct(hit)
        %% draw Ghostball
        if player.setting.ball(bi).ghostball
            hinow = find(Ball(bi).t(ti(bi)) >= hit(bi).t);
            for hi = hinow
                
                if ismember(hit(bi).with(hi), ['W', 'Y', 'R']) % Plot for contact with ball
                    plot(ax, param.ballcirc(1,:) + hit(bi).XPos(hi)/scaleD, ...
                        param.ballcirc(2,:) + hit(bi).YPos(hi)/scaleD, [param.colors(bi),'-'], ...
                        'Tag','myline1','HitTest','off')
                    
                elseif ismember(hit(bi).with(hi), ['1', '2', '3', '4']) % Plot for contact on cushion
                    plot(ax, param.ballcirc(1,:) + hit(bi).XPos(hi)/scaleD, ...
                        param.ballcirc(2,:) + hit(bi).YPos(hi)/scaleD, [param.colors(bi),'-'], ...
                        'Tag','myline1','HitTest','off')
                end
                text(hit(bi).XPos(hi)/scaleD, hit(bi).YPos(hi)/scaleD, num2str(hi),...
                    'Tag','myline1','HitTest','off')
            end
        end
        
        %% draw Diamondpos or Extend Shotline
        if player.setting.ball(bi).diamondpos | (bi==1 & player.setting.plot_check_extendline)
            for hi = 1:min([3 length(hit(bi).with)])
                if ismember(hit(bi).FromC(hi), [1 3]) % Plot for contact on Long cushion 1 / 3
                    linex(1)= hit(bi).FromDPos(hi)/scaleD;
                    liney(1)= offsetDY(hit(bi).FromC(hi));
                    
                elseif ismember(hit(bi).FromC(hi), [2 4]) % Plot for contact on short cushion 2/4
                    linex(1)= offsetDX(hit(bi).FromC(hi));
                    liney(1)= hit(bi).FromDPos(hi)/scaleD;
                end
                
                if ismember(hit(bi).ToC(hi), [1 3]) % Plot for contact on Long cushion 1 / 3
                    linex(2)= hit(bi).ToDPos(hi)/scaleD;
                    liney(2)= offsetDY(hit(bi).ToC(hi));
                    
                elseif ismember(hit(bi).ToC(hi), [2 4]) % Plot for contact on short cushion 2/4
                    linex(2)= offsetDX(hit(bi).ToC(hi));
                    liney(2)= hit(bi).ToDPos(hi)/scaleD;
                    
                end
                if player.setting.ball(bi).diamondpos | (player.setting.plot_check_extendline & bi==1 & hi == 1)
                    plot(ax, linex,liney, [param.colors(bi),'o-.'], 'Tag','myline1','HitTest','off')
                end
            end
%             if player.setting.plot_check_extendline & bi==1 & hi == 1
%                 plot(ax, [0 5000], interp1(linex,liney, [0 5000],'linear','extrap'), ...
%                     [param.colors(bi),'-.'], 'Tag','myline1','HitTest','off')
%             end
        end
    end
    
end

