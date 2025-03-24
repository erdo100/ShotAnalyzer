function plot_MyLine(color, x0,y0)

fig = findobj('Tag','Table_figure');
ax = findobj('Tag','Table');
dcm = datacursormode(fig);
dcm.UpdateFcn = [];


hline = plot(ax,x0,y0,['-',color],'Linewidth',4,'HitTest','off', 'Tag', 'myline3');
hline.UserData = length(hline.XData);

fig.WindowButtonMotionFcn = @(o,e)WBMF(o,e,ax,hline);
ax.ButtonDownFcn = @(o,e)BDF(o,e,fig,ax,hline);

end

function WBMF(this,evnt,ax,hline)

if hline.UserData < 1
    hline.XData = ax.CurrentPoint(1,1);
    hline.YData = ax.CurrentPoint(1,2);
else
    hline.XData(hline.UserData+1) = ax.CurrentPoint(1,1);
    hline.YData(hline.UserData+1) = ax.CurrentPoint(1,2);
    
end
end


function BDF(this,evnt,fig,ax,hline)
if evnt.Button == 1
    if length(hline.XData) >= 1
        hline.XData(end) = ax.CurrentPoint(1,1);
        hline.YData(end) = ax.CurrentPoint(1,2);
    end
elseif evnt.Button == 3
    if length(hline.XData) >= 3
        hline.XData(end-1:end) = [];
        hline.YData(end-1:end) = [];
    end
elseif evnt.Button == 2
    hline.XData(end) = [];
    hline.YData(end) = [];
    fig.WindowButtonMotionFcn = '';
    ax.ButtonDownFcn = '';
    hline.HitTest = 'on';
    disableDefaultInteractivity(ax);
end

hline.UserData = length(hline.XData);

end