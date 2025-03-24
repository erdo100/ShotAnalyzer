function Extract_DataQuality_start(~, ~)
global SA param player

disp(['start (',mfilename,')'])

% ===============EDIT THIS HERE ===============
% ===============  STOP HERE    ===============

hsl = findobj('Tag', 'uitable_shotlist');
sz = size(hsl.Data);

for ti = 1:sz(1)
    identify_ShotID(ti, hsl.Data, hsl.ColumnName)
    si = SA.Current_si;
    if SA.Table.Interpreted{si} == 0
        
        SA.Shot(si).Route = SA.Shot(si).Route0;
        
        % Make calculation
        % ===============EDIT THIS HERE ===============
        errcode = 100;
        err.code = [];
        err.text = [];
        SA.Table.Selected{si} = false;
        
        %% Check whether Points are available
        if isempty(err.code)
            
            errcode = errcode + 1;
            for bi = 1:3
                if size(SA.Shot(si).Route(bi).t,1) == 0
                    err.code = errcode;
                    err.text = ['Ball data is missing 1 (',mfilename,')'];
                    disp([num2str(si),': Ball data is missing 1 (',mfilename,')']);
                    SA.Table.Selected{si} = true;
                end
            end
            errcode = errcode + 1;
            for bi = 1:3
                if size(SA.Shot(si).Route(bi).t,1) == 1
                    err.code = errcode;
                    err.text = ['Ball data is missing 2 (',mfilename,')'];
                    disp([num2str(si),': Ball data is missing 2 (',mfilename,')']);
                    SA.Table.Selected{si} = true;
                end
            end
        end
        
        if 0
            % delete all points outside of cushion
            errcode = errcode + 1;
            if isempty(err.code)
                tol = param.BallOutOfTableDetectionLimit;
                cushion = [param.ballR-tol ...
                    param.size(2)-param.ballR+tol ...
                    param.size(2)-param.ballR+tol ...
                    param.ballR-tol];
                
                for bi = 1:3
                    inddel = false(length(SA.Shot(si).Route(bi).x),1);
                    inddel = inddel | SA.Shot(si).Route(bi).x > cushion(2);
                    inddel = inddel | SA.Shot(si).Route(bi).x < cushion(4);
                    inddel = inddel | SA.Shot(si).Route(bi).y < cushion(1);
                    inddel = inddel | SA.Shot(si).Route(bi).y > cushion(3);
                    if sum(inddel) > 0
                        SA.Shot(si).Route(bi).t(inddel) = [];
                        SA.Shot(si).Route(bi).x(inddel) = [];
                        SA.Shot(si).Route(bi).y(inddel) = [];
                    end
                end
            end
        end
        
        if isempty(err.code)
            
            errcode = errcode + 1;
            for bi = 1:3
                if size(SA.Shot(si).Route(bi).t,1) == 0
                    err.code = errcode;
                    err.text = ['Ball data is missing 1 (',mfilename,')'];
                    disp([num2str(si),': Ball data is missing 1 (',mfilename,')']);
                    SA.Table.Selected{si} = true;
                end
            end
            errcode = errcode + 1;
            for bi = 1:3
                if size(SA.Shot(si).Route(bi).t,1) == 1
                    err.code = errcode;
                    err.text = ['Ball data is missing 2 (',mfilename,')'];
                    disp([num2str(si),': Ball data is missing 2 (',mfilename,')']);
                    SA.Table.Selected{si} = true;
                end
            end
        end
        
        
        % Project Points out of cushion on the cushion
        if 1
            if isempty(err.code)
                tol = param.BallProjecttoCushionLimit;
                cushion = [param.ballR-tol ...
                    param.size(2)-param.ballR+tol ...
                    param.size(1)-param.ballR+tol ...
                    param.ballR-tol];
                oncushion = [param.ballR+0.1...
                    param.size(2)-param.ballR-0.1...
                    param.size(1)-param.ballR-0.1...
                    param.ballR+0.1];
                for bi = 1:3
                    % check ball position out of cushion 2
                    inddel = SA.Shot(si).Route(bi).x > oncushion(2);
                    if ~isempty(find(inddel,1))
                        SA.Shot(si).Route(bi).x(inddel) = oncushion(2);
                    end
                    % check ball position out of cushion 4
                    inddel = SA.Shot(si).Route(bi).x < oncushion(4);
                    if ~isempty(find(inddel,1))
                        SA.Shot(si).Route(bi).x(inddel) = oncushion(4);
                    end
                    % check ball position out of cushion 1
                    inddel = SA.Shot(si).Route(bi).y < oncushion(1);
                    if ~isempty(find(inddel,1))
                        SA.Shot(si).Route(bi).y(inddel) = oncushion(1);
                    end
                    % check ball position out of cushion 3
                    inddel = SA.Shot(si).Route(bi).y > oncushion(3);
                    if ~isempty(find(inddel,1))
                        SA.Shot(si).Route(bi).y(inddel) = oncushion(3);
                    end
                end
            end
            
            
            %% Check wheter initial balldistance is larger than ball diameter
            errcode = errcode + 1;
            if isempty(err.code)
                BB = [1 2; 1 3; 2 3];
                for bbi = 1:3
                    b1i = BB(bbi,1);
                    b2i = BB(bbi,2);
                    balldist(bbi) = sqrt((SA.Shot(si).Route(b1i).x(1) - SA.Shot(si).Route(b2i).x(1)).^2 + ...
                        (SA.Shot(si).Route(b1i).y(1) - SA.Shot(si).Route(b2i).y(1)).^2) - 2*param.ballR;
                    if balldist(bbi) < 0
                        % Balls are interfering
                        % Move them in ball center connection diretion
                        vec = [(SA.Shot(si).Route(b2i).x(1) - SA.Shot(si).Route(b1i).x(1)) ...
                            (SA.Shot(si).Route(b2i).y(1) - SA.Shot(si).Route(b1i).y(1))];
                        vec = -vec/norm(vec)*balldist(bbi);
                        
                        SA.Shot(si).Route(b1i).x(1) = SA.Shot(si).Route(b1i).x(1) - vec(1)/2;
                        SA.Shot(si).Route(b1i).y(1) = SA.Shot(si).Route(b1i).y(1) - vec(2)/2;
                        SA.Shot(si).Route(b2i).x(1) = SA.Shot(si).Route(b2i).x(1) + vec(1)/2;
                        SA.Shot(si).Route(b2i).y(1) = SA.Shot(si).Route(b2i).y(1) + vec(2)/2;
                        
                    end
                end
            end
        end
        
        
        
        %% linearity of time
        errcode = errcode + 1;
        if isempty(err.code)
            for bi = 1:3
                err.region(bi).ti = [];
                dt = diff(SA.Shot(si).Route(bi).t);
                ind = find(dt <= 0);
                delind = [];
                
                % Try to correct
                if ~isempty(ind)
                    disp([num2str(si),': Try to fix time linearity ....'])
                    t = SA.Shot(si).Route(bi).t;
                    
                    for ei = 1:length(ind)
                        delindnew = find(t(ind(ei)+1:end) <= t(ind(ei)))+ind(ei);
                        delind = [delind; delindnew];
                    end
                    
                    t(delind) = [];
                    dt = diff(t);
                    indnew = find(dt <= 0);
                    
                    if isempty(indnew)
                        disp([num2str(si),': Successfully fixed time linearity :)'])
                        SA.Shot(si).Route(bi).t(delind) = [];
                        SA.Shot(si).Route(bi).x(delind) = [];
                        SA.Shot(si).Route(bi).y(delind) = [];
                        ind = [];
                    end
                end
                
                if ~isempty(ind)
                    err.code = errcode;
                    err.text = ['no time linearity (',mfilename,')'];
                    disp([num2str(si),': no time linearity (',mfilename,')']);
                    for ei = 1:length(ind)
                        err.region(bi).ti(end+1) = ind(ei);
                    end
                    SA.Table.Selected{si} = true;
                end
            end
        end
        
        %% Check time start and end time
        if isempty(err.code)
            errcode = errcode + 1;
            
            for bi = 1:3
                if SA.Shot(si).Route(1).t(1) ~= SA.Shot(si).Route(2).t(1) | ...
                        SA.Shot(si).Route(1).t(1) ~= SA.Shot(si).Route(3).t(1)
                    err.code = errcode;
                    err.text = ['time doesnt start from 0 (',mfilename,')'];
                    disp([num2str(si),': time doesnt start from 0 (',mfilename,')']);
                    SA.Table.Selected{si} = true;
                end
            end
            
            errcode = errcode + 1;
            for bi = 1:3
                if SA.Shot(si).Route(1).t(end) ~= SA.Shot(si).Route(2).t(end) | ...
                        SA.Shot(si).Route(1).t(end) ~= SA.Shot(si).Route(3).t(end)
                    err.code = errcode;
                    err.text = ['time doesnt end equal for all balls (',mfilename,')'];
                    disp([num2str(si),': time doesnt end equal for all balls (',mfilename,')']);
                    SA.Table.Selected{si} = true;
                end
            end
        end
        
        %% check ball reflection/jumping
        if isempty(err.code)
            errcode = errcode + 1;
            % Calculate travel distance
            for bi = 1:3
                check = 1;
                i=1;
                while check
                    % to overnext
                    if i > length(SA.Shot(si).Route(bi).t)-2
                        break
                    end

                    dsx = diff(SA.Shot(si).Route(bi).x([i i+2]));
                    dsy = diff(SA.Shot(si).Route(bi).y([i i+2]));
                    dl0 = sqrt(dsx^2+dsy^2);
                    vec02 = [dsx dsy 0]';
                    
                    % to next
                    dsx = diff(SA.Shot(si).Route(bi).x([i i+1]));
                    dsy = diff(SA.Shot(si).Route(bi).y([i i+1]));
                    dl1 = sqrt(dsx^2+dsy^2);
                    vec01 = [dsx dsy 0]';
                    
                    % next to overnext
                    dsx = diff(SA.Shot(si).Route(bi).x([i+1 i+2]));
                    dsy = diff(SA.Shot(si).Route(bi).y([i+1 i+2]));
                    dl2 = sqrt(dsx^2+dsy^2);
                    vec12 = [dsx dsy 0]';
                    
                    a1 = angle2_vector(vec01,vec02);
                    a2 = angle2_vector(vec12,vec02);
                    a0 = angle2_vector(vec01,vec12);
                    if dl0 < dl1*0.5 & dl0 < dl2*0.5 & dl1 > 100
                        SA.Shot(si).Route(bi).t(i+1) = [];
                        SA.Shot(si).Route(bi).x(i+1) = [];
                        SA.Shot(si).Route(bi).y(i+1) = [];
                    end
                    i=i+1;

                end
%                 if length(SA.Shot(si).Route(bi).x) >= 4
%                     dsx = diff(SA.Shot(si).Route(bi).x);
%                     dsy = diff(SA.Shot(si).Route(bi).y);
%                     v = [dsx; dsx; zeros(length(dsx),1)]';
%                     
%                     if abs(dsx(1)+dsx(2))<5 | abs(dsy(1)+dsy(2))<5
%                         disp([num2str(si),': reflection found and corrected (',mfilename,')']);
%                         SA.Shot(si).Route(bi).t(2:3) = [];
%                         SA.Shot(si).Route(bi).x(2:3) = [];
%                         SA.Shot(si).Route(bi).y(2:3) = [];
%                     end
%                         
%                 end

            end
        end
        
        
        %% check gaps in tracking
        if isempty(err.code)
            errcode = errcode + 1;
            % Calculate travel distance
            for bi = 1:3
                ds = sqrt(diff(SA.Shot(si).Route(bi).x).^2 + diff(SA.Shot(si).Route(bi).y).^2);
                vabs = ds./diff(SA.Shot(si).Route(bi).t);
                if find(ds > param.NoDataDistanceDetectionLimit,1)
                    err.code = errcode;
                    err.text = ['gap in data is too big (',mfilename,')'];
                    disp([num2str(si),': gap in data is too big (',mfilename,')']);
                    SA.Table.Selected{si} = true;
                end
                if find(vabs > param.MaxVelocity,1)
                    err.code = errcode;
                    err.text = ['Velocity is too high (',mfilename,')'];
                    disp([num2str(si),': Velocity is too high (',mfilename,')']);
                    SA.Table.Selected{si} = true;
                end
            end
        end
        
        
        
        %% Update Route0
        for bi = 1:3
            SA.Shot(si).Route0(bi) = SA.Shot(si).Route(bi);
        end
        
        %% ===============  STOP HERE    ===============
        
        % Update the err
        SA.Table.ErrorID{si} = err.code;
        SA.Table.ErrorText{si} = err.text;
    end
end
% update GUI
update_ShotList
player.uptodate = 0;

disp([num2str(sum(cell2mat(hsl.Data(:,1)))),'/',num2str(length(hsl.Data(:,1))),' shots selected'])
disp(['done (',mfilename,')'])