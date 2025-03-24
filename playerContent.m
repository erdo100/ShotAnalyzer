function player = playerContent(player, what2do)
% Here we prepare all data for the player
% Read all selected shots
% Interpolate the shots
% save data to player
%
% This has to be read only when player is activated
global SA


% What is currently selected
toplot = SA.Current_si;

% Order b1b2b3
if sum(strcmp(SA.Table.Properties.VariableNames, 'B1B2B3')) == 0
    % There is no B1B2B3 identified
    bind123 = [zeros(size(SA.Table,1),1)+1 zeros(size(SA.Table,1),1)+2 zeros(size(SA.Table,1),1)+3];
else
    bind123 = sort_b1b2b3(SA.Table.B1B2B3);
end

for si = 1:length(toplot)
    Tmax(si) = max([max(SA.Shot(toplot(si)).Route0(1).t) max(SA.Shot(toplot(si)).Route0(2).t) max(SA.Shot(toplot(si)).Route0(3).t)]);
end

tappr = [0:player.dt:max(Tmax)]';

for si = 1:length(toplot)
    if isfield(SA.Shot(toplot(si)),'hit')
        if isfield(SA.Shot(toplot(si)).hit,'with')
            %         if ~isempty(SA.Shot(toplot(si)).hit)
            Shot(si).hit = SA.Shot(toplot(si)).hit(bind123(toplot(si),:));
        end
    end
    for bii = 1:3
        bi = bind123(toplot(si),bii);
        switch what2do
            case 'plotcurrent'
                Shot(si).ball(bii).t = SA.Shot(toplot(si)).Route0(bi).t;
                Shot(si).ball(bii).x = SA.Shot(toplot(si)).Route0(bi).x;
                Shot(si).ball(bii).y = SA.Shot(toplot(si)).Route0(bi).y;
                player.uptodate = 1;

            otherwise
                Shot(si).ball(bii).t = tappr;
                torig = [0; SA.Shot(toplot(si)).Route0(bi).t(2)-0.001; ...
                    SA.Shot(toplot(si)).Route0(bi).t(2:end); SA.Shot(toplot(si)).Route0(bi).t(end)+1];
                
                Shot(si).ball(bii).x = interp1( ...
                    torig, [SA.Shot(toplot(si)).Route0(bi).x(1); SA.Shot(toplot(si)).Route0(bi).x(1); ...
                    SA.Shot(toplot(si)).Route0(bi).x(2:end); SA.Shot(toplot(si)).Route0(bi).x(end)], ...
                    tappr,'linear','extrap');
                
                Shot(si).ball(bii).y = interp1( ...
                    torig, [SA.Shot(toplot(si)).Route0(bi).y(1); SA.Shot(toplot(si)).Route0(bi).y(1); ...
                    SA.Shot(toplot(si)).Route0(bi).y(2:end); SA.Shot(toplot(si)).Route0(bi).y(end)], ...
                    tappr,'linear','extrap');
                player.uptodate = 2;
        end
    end
end

player.Shot = Shot;
player.Timax = length(tappr);
player.ti = [1 1 1];
player.video = 0;

% % Set color of current ball
% if bind123(toplot(1),1) == 1
    player.pt_CurrentBall.CData = player.icon.WhiteBall;
    player.b1i = 1;
% elseif bind123(toplot(1),1) == 2
%     player.pt_CurrentBall.CData = player.icon.YellowBall;
%     player.b1i = 2;
% elseif bind123(toplot(1),1) == 3
%     player.pt_CurrentBall.CData = player.icon.RedBall;
%     player.b1i = 3;
% end
