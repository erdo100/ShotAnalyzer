function ax = plot_table
global param player
root = groot;

fig = get(findobj('Tag','figure1'));

if ~isfield(param,'TablePosition')
    width = round(root.ScreenSize(3)*0.5/4, 0)*4; % divide by for to have even number for height.
    % Avoids warning when creating MP4: resolution must be multiple of 2
    height = round(width/(param.size(2)+param.framewidth*2)*(param.size(1)+param.framewidth*2)/2, 0)*2;
    
    if param.with_title
        height = height+44;
    end
    posx = fig.Position(1)+fig.Position(3);
    posy = fig.Position(2)+fig.Position(4)-height-50;
    
    param.TablePosition = [posx posy width height];
end


% create figure and axes
ax = findobj('Tag','Table');
if isempty(ax)
    fig = figure('Position', param.TablePosition, ...
        'Tag','Table_figure', 'PaperType','a4', 'PaperOrientation','landscape',...
        'NumberTitle','off', 'Name','Billard Table', 'ResizeFcn',@ResizeTableFigure);
    
    % add menu to table figure
    Table_menu(fig);
    
    addToolbarExplorationButtons(fig)
    plotedit({'plotedittoolbar',gcf,'toggle'})
    
    if param.with_title
        ax = axes(fig,'Position',[0 0 1 0.95],'Tag','Table','NextPlot','add',...
            'XLimMode','manual','YLimMode','manual','XtickLabel','','YtickLabel','','TickLength',[0,0], ...
            'Xcolor','none','Ycolor','none');
    else
        ax = axes(fig,'Position',[0 0 1 1],'Tag','Table','NextPlot','add',...
            'XLimMode','manual','YLimMode','manual','XtickLabel','','YtickLabel','','TickLength',[0,0], ...
            'Xcolor','none','Ycolor','none');
    end
    if strcmp(player.setting.plot_only_blue_table,'off')
        % plot cushion line
        plot(ax,[0 param.size(2) param.size(2) 0 0],[0 0 param.size(1) param.size(1) 0],'-b',...
            'HitTest','off')
        hold(ax, 'on');
        
        % plot Table frame
        patch(ax,...
            [ -param.cushionwidth  param.size(2)+param.cushionwidth  ... % (1) inside low
            param.size(2)+param.cushionwidth -param.cushionwidth ... % (1) inside top
            -param.cushionwidth -param.framewidth ... % (1) outside low left
            -param.framewidth param.size(2)+param.framewidth  ... % (1) outside top
            param.size(2)+param.framewidth -param.cushionwidth], ... % (1) outside bottom right
            [ -param.cushionwidth  -param.cushionwidth ...
            param.size(1)+param.cushionwidth param.size(1)+param.cushionwidth ...
            -param.framewidth -param.framewidth ...
            param.size(1)+param.framewidth param.size(1)+param.framewidth ...
            -param.framewidth -param.framewidth], ...
            [0.5 0.5 0], 'EdgeColor','none', 'Tag','Tableframe', ...
            'HitTest','off')
        
        
        
        % inside cushion line
        plot(ax,[param.ballR param.size(2)-param.ballR param.size(2)-param.ballR param.ballR param.ballR], ...
            [param.ballR param.ballR param.size(1)-param.ballR param.size(1)-param.ballR param.ballR], ...
            '--', 'color',[0.5 0.5 0.95],'HitTest','off')
        % diamonds line
        plot(ax,[-param.diamdist  param.size(2)+param.diamdist  param.size(2)+param.diamdist  -param.diamdist  -param.diamdist ], ...
            [-param.diamdist  -param.diamdist  param.size(1)+param.diamdist  param.size(1)+param.diamdist  -param.diamdist ], ...
            '--', 'Color',[0.45 0.45 0],'HitTest','off')
        
        % Diamonds
        for i = 1:9
            % bottom
            diamx = param.size(2)/8*(i-1);
            diamylow = -param.diamdist ;
            diamyupp = param.size(1)+param.diamdist ;
            
            % bottom
            patch(diamx+sin(linspace(0,2*pi,12))*param.rdiam, ...
                diamylow+cos(linspace(0,2*pi,12))*param.rdiam, ...
                'k', 'EdgeColor','none','HitTest','off')
            % top
            patch(diamx+sin(linspace(0,2*pi,12))*param.rdiam, ...
                diamyupp+cos(linspace(0,2*pi,12))*param.rdiam, ...
                'k', 'EdgeColor','none','HitTest','off')
        end
        for i = 1:5
            diamxleft = -param.diamdist ;
            diamxright =  param.size(2)+param.diamdist ;
            diamy = param.size(1)/4*(i-1);
            % left
            patch(diamxleft+sin(linspace(0,2*pi,12))*param.rdiam, ...
                diamy+cos(linspace(0,2*pi,12))*param.rdiam, ...
                'k', 'EdgeColor','none','HitTest','off')
            % right
            patch(diamxright+sin(linspace(0,2*pi,12))*param.rdiam, ...
                diamy+cos(linspace(0,2*pi,12))*param.rdiam, ...
                'k', 'EdgeColor','none','HitTest','off')
        end
        
        grid(ax, 'on');
        axis(ax, 'equal');
        xlim(ax,[-param.framewidth param.size(2)+param.framewidth])
        ylim(ax,[-param.framewidth param.size(1)+param.framewidth])
        set(ax,'XtickLabel','','YtickLabel','')
        
        % set ticks
        set(ax,'Xtick',linspace(0,param.size(2),9),'ytick',linspace(0,1420,5),...
            'Tag','Table', 'Color',[0 102 185]/255, ...
            'buttondownfcn',@drawmylines, 'UserData',1);
        
        plot([2 2 2 6 6 6 4]/4*param.size(1), ...
            [param.size(1)/2-182.5 param.size(1)/2 param.size(1)/2+182.5 ...
            param.size(1)/2-182.5 ...
            param.size(1)/2+182.5 param.size(1)/2 param.size(1)/2], ...
            '+k')
        
    else
        axis(ax, 'equal');
        xlim(ax,[-param.framewidth param.size(2)+param.framewidth])
        ylim(ax,[-param.framewidth param.size(1)+param.framewidth])
        set(ax, 'Color',[0 102 185]/255, 'Tag','Table', ...
            'buttondownfcn',@drawmylines, 'UserData',1);

    end
else
    fig = findobj('Tag','Table_figure');
    delete(findobj('-regexp','Tag','PlotBall*'))
    delete(findobj('Tag','myline1'));
    delete(findobj('Tag','myline2'));
    delete(findobj('Tag','myline3'));
    param.TablePosition = fig.Position;

end


function ResizeTableFigure(hf, eventdata, ~)
global param

param.TablePosition = eventdata.Source.Position;


function drawmylines(objhandle,eventdata)
button = eventdata.Button;


