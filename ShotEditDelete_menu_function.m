function ShotEditDelete_menu_function(~,~)

hf = findobj('Name','Billard Table');
dcm = datacursormode(hf);
dcm.Enable = 'on';
dcm.DisplayStyle = 'Window';
dcm.SnapToDataVertex = 'on';
dcm.UpdateFcn = @deleteCoordinates;
end


function txt = deleteCoordinates(~,info)
global SA

h = findobj('Tag', 'uitable_shotlist');

if sum(strcmp(SA.Table.Properties.VariableNames, 'B1B2B3')) == 0
    bind123 = [zeros(size(SA.Table,1),1)+1 zeros(size(SA.Table,1),1)+2 zeros(size(SA.Table,1),1)+3];
else
    bind123 = sort_b1b2b3(SA.Table.B1B2B3);
end
   
% What is currently selected
if length(SA.Current_si) == 1
    
    if info.Target.Color(2) == 1 % White
        bi = 1;
    elseif info.Target.Color(2) == 0.8 % Yellow
        bi = 2;
    elseif info.Target.Color(2) == 0 % Red
        bi = 3;
    end
    
    x = info.Position(1);
    y = info.Position(2);
    ind = find(info.Target.XData == x & info.Target.YData == y);
    txt = ['X=' num2str(x) ', Y=' num2str(y) ''];
    
    info.Target.XData(ind) = [];
    info.Target.YData(ind) = [];
    
    SA.Shot(SA.Current_si).Route0(bind123(SA.Current_si,bi)).t(ind) = [];
    SA.Shot(SA.Current_si).Route0(bind123(SA.Current_si,bi)).x(ind) = [];
    SA.Shot(SA.Current_si).Route0(bind123(SA.Current_si,bi)).y(ind) = [];
    
    % Reset evaluated data
    SA.Shot(SA.Current_si).Route(bind123(SA.Current_si,bi)).t = ...
        SA.Shot(SA.Current_si).Route0(bind123(SA.Current_si,bi)).t;
    SA.Shot(SA.Current_si).Route(bind123(SA.Current_si,bi)).x = ...
        SA.Shot(SA.Current_si).Route0(bind123(SA.Current_si,bi)).x;
    SA.Shot(SA.Current_si).Route(bind123(SA.Current_si,bi)).y = ...
        SA.Shot(SA.Current_si).Route0(bind123(SA.Current_si,bi)).y;
    
    % Reset Interpreted flag in Table
    SA.Table.Interpreted(SA.Current_si) = 0;
    
else
    disp('Please select only one shot to delete points')
    
end

end