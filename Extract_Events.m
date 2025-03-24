function [hit, err] = Extract_Events(si)
% Extract all ball-ball hit events and ball-Cushion hit events

global SA param

plot_flag =  strcmp(get(findobj('Tag','PlotAnalytics'),'Checked'), 'on');
ti_plot_start = 1; %

% initialize outputs
err.code = [];
err.text = '';

col = 'WYR';
ax = findobj('Tag','Table');

% Get B1,B2,B3 from ShotList
[b1b2b3, b1i, b2i, b3i] = str2num_B1B2B3(SA.Table.B1B2B3{si});

% Get the copy from original data
for bi = 1:3
    ball0(bi) = SA.Shot(si).Route0(bi);
    SA.Shot(si).Route(bi).t = SA.Shot(si).Route0(bi).t;
    SA.Shot(si).Route(bi).x = SA.Shot(si).Route0(bi).x;
    SA.Shot(si).Route(bi).y = SA.Shot(si).Route0(bi).y;
end

for bi = 1:3
    ball(bi).x = SA.Shot(si).Route0(bi).x(1);
    ball(bi).y = SA.Shot(si).Route0(bi).y(1);
    ball(bi).t = 0;
end

% Create common Time points
Tall0 = unique([ball0(1).t; ball0(2).t; ball0(3).t]);
ti=0;

% discretization o dT
tvec = linspace(0.01,1,101)';


%% Scanning all balls
hit(b1i).with = 'S';
hit(b1i).t = 0;
hit(b1i).XPos = ball0(b1i).x(1);
hit(b1i).YPos = ball0(b1i).y(1);
hit(b2i).with = '-';
hit(b2i).t = 0;
hit(b2i).XPos = ball0(b2i).x(1);
hit(b2i).YPos = ball0(b2i).y(1);
hit(b3i).with = '-';
hit(b3i).t = 0;
hit(b3i).XPos = ball0(b3i).x(1);
hit(b3i).YPos = ball0(b3i).y(1);
hit(b1i).Kiss = 0;
hit(b1i).Point = 0;
hit(b1i).Fuchs = 0;
hit(b1i).PointDist = 3000;
hit(b1i).KissDistB1 = 3000;
hit(b1i).Tpoint = 3000;
hit(b1i).Tkiss = 3000;
hit(b1i).Tready = 3000;
hit(b1i).TB2hit = 3000;
hit(b1i).Tfailure = 3000;

%% Targets
% - Make valid Ball routes which resolve every event exactly
% - Smoothen the trajectories, accelerations

%% Idea
% make simulation/extrapolation for next time steps

% But what is the velicity of the balls. Very initially, B1 is moving, B2,B3 are not moving.
% How big is V1 velocity? Where can we find indications?

%% Investigate forward
% integrate from current position/velocity and check distance

%% Investigate backward
% use velocities after the collision and estimate previous condition
% check energy/momentum of sum of balls

% Case 1: No collision in ti=2
%  - Ball distance is big in first time steps
%  - Ball distance is small in first time steps

% Case 2: Collision in ti=2
%  - Ball distance is big in first time steps
%  - Ball distance is small in first time steps

% B2 velocity is 0 in ti=2

% ball velocities
for bi = 1:3
    ball0(bi).dt = diff(ball0(bi).t);
    ball0(bi).vx = [diff(ball0(bi).x)./ball0(bi).dt; 0];
    ball0(bi).vy = [diff(ball0(bi).y)./ball0(bi).dt; 0];
    ball0(bi).v  = [sqrt(ball0(bi).vx.^2 + ball0(bi).vy.^2); 0];
    ball0(bi).dt = [ball0(bi).dt; ball0(bi).dt(end)];
end
% Set initial ball speeds for B2 and B3 = 0
ball0(b2i).vx(1) = 0;
ball0(b2i).vy(1) = 0;
ball0(b2i).v(1)  = 0;
ball0(b3i).vx(1) = 0;
ball0(b3i).vy(1) = 0;
ball0(b3i).v(1)  = 0;

if plot_flag
    ax = plot_table;
    lw = ones(1,3)*0.5;
    plot_shot(ax, ball0, lw)
end
%% Lets start

do_scan = 1;
bi_list = [1 2 3];

% ball0 shall contain only remaining future data
% ball1 shall contain only proccessed past data


while do_scan
    ti=ti+1;
    
    %% Approximate Position of Ball at next time step
    dT = diff(Tall0(1:2));
    tappr = Tall0(1) + dT*tvec;
    for bi = 1:3
        
        % Check if it is last index
        if length(ball0(bi).t) >= 2
            
            %Travel distance in original data
            if length(ball(bi).x) >=2 
                ds0 = sqrt((ball(bi).x(2)-ball(bi).x(1))^2 + (ball(bi).y(2)-ball(bi).y(1))^2);
                v0 = ds0/dT;
            end
            % current Speed
            b(bi).vt1 = ball0(bi).v(1);
            
            % was last hit in last time step?
            iprev = find(ball(bi).t >= hit(bi).t(end), param.timax_appr, 'last');
            
            % Speed Components in the next ti
            if strcmp(hit(bi).with, '-')
                % Ball was not moving previously
                % hit(bi).with has only 1 element = '-'
                % Ball 2 and 3 with ti = 1, ball is not moving ==> v=0
                b(bi).v1 = [0 0];
                b(bi).v2 = [ball0(bi).vx(2) ball0(bi).vy(2)];
                
            elseif length(iprev) == 1
                % when last hit  was on this ball
                % current X -Y Velocity components
                b(bi).v1 = [ball0(bi).vx(1) ball0(bi).vy(1)];
                b(bi).v2 = [ball0(bi).vx(2) ball0(bi).vy(2)];
                
            else
                % extrapolate from last points
                CoefVX = [ball(bi).t(iprev) ones(length(iprev),1)]\ball(bi).x(iprev);
                CoefVY = [ball(bi).t(iprev) ones(length(iprev),1)]\ball(bi).y(iprev);
                
                b(bi).v1 = [CoefVX(1) CoefVY(1)];
                b(bi).v2 = [ball0(bi).vx(2) ball0(bi).vy(2)];
            end
            
            % Speed in data for next ti
            b(bi).vt1 = norm(b(bi).v1);
            b(bi).vt2 = norm(b(bi).v2);
            
            vtnext = max([b(bi).vt1 ball0(bi).v(1)]);
            
            if norm(b(bi).v1) > 0
                vnext = b(bi).v1 / norm(b(bi).v1)*vtnext;
            else
                vnext = [0 0];
            end
            
            % Approximation of next ti
%             b(bi).xa = ball(bi).x(end) + b(bi).v1(1)*dT*tvec;
%             b(bi).ya = ball(bi).y(end) + b(bi).v1(2)*dT*tvec;
            b(bi).xa = ball(bi).x(end) + vnext(1)*dT*tvec;
            b(bi).ya = ball(bi).y(end) + vnext(2)*dT*tvec;
            
        end
        
        %% Plot
        if plot_flag & ti >= ti_plot_start
            %          plot(ax, ball(bi).x(end), ball(bi).y(end), 'ok', 'tag', 'hlast')
            plot(ax, ball(bi).x, ball(bi).y, '-or', 'tag', 'hlast')
            plot(ax, b(bi).xa([1 end]), b(bi).ya([1 end]), '--or', 'tag', 'hlast',...
                'Markersize',7, 'Linewidth',2','MarkerFaceColor','r')
            plot(ax, ball(bi).x(end)+param.ballcirc(1,:),ball(bi).y(end)+param.ballcirc(2,:),'-k', 'tag', 'hlast')
            plot(ax, b(bi).xa(end)+param.ballcirc(1,:),b(bi).ya(end)+param.ballcirc(2,:),'-r', 'tag', 'hlast')
            
            drawnow
        end
        
    end
    
    %% Calculate Ball trajectory angle change
    for bi = 1:3
        % Angle of trajectory in data for next ti
        b(bi).a12 = angle_vector(b(bi).v1, b(bi).v2);
    end
    
    %% Calculate the Ball Ball Distance
    % List of Ball-Ball collisions
    BB = [1 2; 1 3; 2 3];
    
    % Get B1,B2,B3 from ShotList
    [b1b2b3, b1i, b2i, b3i] = str2num_B1B2B3(SA.Table.B1B2B3{si});

    for bbi = 1:3
        bx1 = b1b2b3(BB(bbi,1));
        bx2 = b1b2b3(BB(bbi,2));
        
        d(bbi).BB = sqrt((b(bx1).xa - b(bx2).xa).^2 + ...
            (b(bx1).ya - b(bx2).ya).^2) - 2*param.ballR;
    end
    
    %% Calculate Cushion distance
    for bi = 1:3
        b(bi).cd(:,1) = b(bi).ya - param.ballR;
        b(bi).cd(:,2) = param.size(2)-param.ballR - b(bi).xa;
        b(bi).cd(:,3) = param.size(1)-param.ballR - b(bi).ya ;
        b(bi).cd(:,4) = b(bi).xa - param.ballR;
    end
    
    hitlist = [];
    lasthitball = 0;
    
    %% Evaluate cushion hit
    % Criteria for cushion hit:
    %  - Ball was moving in Cushion direction
    %  - Ball coordinate is close to cushion
    %  - Ball is changing direction
    for bi = 1:3
        for cii = 1:4
            
            checkdist = sum(min(b(bi).cd(:,cii)) <= 0) > 0;
            checkangle = b(bi).a12 > 1 | b(bi).a12 == -1;
            velx = b(bi).v1(1);
            vely = b(bi).v1(2);
            
            checkcush = 0;
            tc = 0;
            if checkdist & checkangle & vely < 0 & cii==1 & b(bi).v1(2)~=0 % bottom cushion
                checkcush = 1;
                tc = interp1(b(bi).cd(:,cii), tappr, 0, 'linear','extrap'); % Exact time of contact
                cushx = interp1(tappr, b(bi).xa, tc, 'linear','extrap');
                cushy = param.ballR;
            elseif checkdist & checkangle & velx > 0 & cii == 2  & b(bi).v1(1)~=0 % right cushions
                checkcush = 1;
                tc = interp1(b(bi).cd(:,cii), tappr, 0, 'linear','extrap'); % Exact time of contact
                cushx = param.size(2)-param.ballR;
                cushy =interp1(tappr, b(bi).ya, tc, 'linear','extrap');
            elseif checkdist & checkangle & vely > 0 & cii == 3  & b(bi).v1(2)~=0 % top Cushion
                checkcush = 1;
                tc = interp1(b(bi).cd(:,cii), tappr, 0, 'linear','extrap'); % Exact time of contact
                cushy = param.size(1)-param.ballR;
                cushx = interp1(tappr, b(bi).xa, tc, 'linear','extrap');
            elseif checkdist & checkangle & velx < 0 & cii == 4  & b(bi).v1(1)~=0 % left Cushion
                checkcush = 1;
                tc = interp1(b(bi).cd(:,cii), tappr, 0, 'linear','extrap'); % Exact time of contact
                cushx = param.ballR;
                cushy = interp1(tappr, b(bi).ya, tc, 'linear','extrap');
            end
            if tc < 0
                disp('')
            end
            if checkcush
                
                hitlist(end+1, 1) = tc; % What Time
                hitlist(end, 2) = bi; % Which Ball
                hitlist(end, 3) = 2;   % Contact with 1=Ball, 2=cushion
                hitlist(end, 4) = cii; % Contact with ID
                hitlist(end, 5) = cushx; % Contact location
                hitlist(end, 6) = cushy; % Contact location
            end
            
        end
    end
    
    
    %% Evaluate Ball-Ball hit
    for bbi = 1:3
        bx1 = b1b2b3(BB(bbi,1));
        bx2 = b1b2b3(BB(bbi,2));
        
        % Check whether distance has values smaller and bigger 0
        % But take care, B1 can go through B2. Therefore use transition
        % positiv gap ==> negative gap.
        checkdist = 0;
        if sum(d(bbi).BB <= 0) > 0 & sum(d(bbi).BB > 0) > 0 & ...
                ((d(bbi).BB(1) >= 0 & d(bbi).BB(end) < 0) | ...
                (d(bbi).BB(1) < 0 & d(bbi).BB(end) >= 0))
            % Balls are going to be in contact or are already in contact. Previous contact not detected
            
            checkdist = 1;
            tc = interp1(d(bbi).BB, tappr, 0, 'linear','extrap');
            
        elseif sum(d(bbi).BB <= 0) > 0 & sum(d(bbi).BB > 0) > 0
            % here the ball-ball contact is going through the balls
            checkdist = 1;
            ind = diff(d(bbi).BB) <= 0;
            tc = interp1(d(bbi).BB(ind), tappr(ind), 0, 'linear','extrap');        
        end

        checkangleb1 = b(bx1).a12 > 10 | b(bx1).a12 == -1;
        checkangleb2 = b(bx2).a12 > 10 | b(bx2).a12 == -1;
        
        if abs(b(bx1).vt1) > eps | abs(b(bx1).vt2) > eps
            checkvelb1 = abs(b(bx1).vt2 - b(bx1).vt1) > 50;
        else
            checkvelb1 = 0;
        end
        
        if abs(b(bx2).vt1) > eps | abs(b(bx2).vt2) > eps
            checkvelb2 = abs(b(bx2).vt2 - b(bx2).vt1) > 50;
        else
            checkvelb2 = 0;
        end
        
        if tc >= hit(bx1).t(end)+0.01 & tc >= hit(bx2).t(end)+0.01
            checkdouble = 1;
        else
            checkdouble = 0;
        end
        
        
        
        
        if checkdouble==1 & checkdist==1 %& (checkangleb1 ==1 & checkangleb2==1) & checkvelb1==1 & checkvelb2==1
            % disp('Contact.....Contact.....Contact.....Contact.....Contact.....')
            hi = size(hitlist,1) + 1;
            
            hitlist(hi,1) = tc; % Contact Time
            hitlist(hi,2) = bx1; % Ball 1
            hitlist(hi,3) = 1; % 1= Ball-Ball, 2= Cushion
            hitlist(hi,4) = bx2; % Ball 2
            hitlist(hi,5) = interp1(tappr, b(bx1).xa, tc, 'linear','extrap');
            hitlist(hi,6) = interp1(tappr, b(bx1).ya, tc, 'linear','extrap');
            
            hi = size(hitlist,1) + 1;
            hitlist(hi,1) = tc; % Contact Time
            hitlist(hi,2) = bx2; % Ball 1
            hitlist(hi,3) = 1; % 1= Ball-Ball, 2= Cushion
            hitlist(hi,4) = bx1; % Ball 2
            hitlist(hi,5) = interp1(tappr, b(bx2).xa, tc, 'linear','extrap');% Contact location x
            hitlist(hi,6) = interp1(tappr, b(bx2).ya, tc, 'linear','extrap');% Contact location y
        end
    end
    
    %% When Just before the Ball-Ball hit velocity is too small, then hit can be missed
    % therefore we check whether B2 is moving without hit
    % now only for first hit
    % then we have to calculate the past hit
    % Move back B1 so that B1 is touching B2
    
    if ti > 1 & length(hit(b2i).t) == 1 & isempty(hitlist) & ball0(b1i).t(2) >= ball0(b2i).t(2) & ...
            (ball0(b2i).x(1) ~= ball0(b2i).x(2) | ball0(b2i).y(1) ~= ball0(b2i).y(2))

        % which bbi is b1i hits b2i
        % first is B1, second is B2
%         bbi = find(BB(:,1) == b1i & BB(:,2) == b2i);
        bbi = find(BB(:,1) == 1 & BB(:,2) == 2);
        
        tc = interp1(d(bbi).BB, tappr, 0, 'linear','extrap');
        if tc < 0
            [~,ti] = min(d(bbi).BB);
            tc = tappr(ti);
%             tc = tappr(end);
        end

        hi = size(hitlist,1) + 1;
        hitlist(hi,1) = tc; % Contact Time
        hitlist(hi,2) = b1i; % Ball 1
        hitlist(hi,3) = 1; % 1= Ball-Ball, 2= Cushion
        hitlist(hi,4) = b2i; % Ball 2
        hitlist(hi,5) = interp1(tappr, b(b1i).xa, tc, 'linear','extrap');
        hitlist(hi,6) = interp1(tappr, b(b1i).ya, tc, 'linear','extrap');
        
        hi = size(hitlist,1) + 1;
        hitlist(hi,1) = tc; % Contact Time
        hitlist(hi,2) = b2i; % Ball 1
        hitlist(hi,3) = 1; % 1= Ball-Ball, 2= Cushion
        hitlist(hi,4) = b1i; % Ball 2
        hitlist(hi,5) = ball0(b2i).x(1);% Contact location x
        hitlist(hi,6) = ball0(b2i).y(1);% Contact location y

    end

    %% Check first Time step for Ball-Ball hit
    % if Balls are too close, so that the direction change is not visible,
    % then the algorithm cant detect the Bal Ball hit. Therefore:
    % Check whether B2 is moving with without hit
    % If yes, then use the direction of B2, calculate the perpendicular
    % direction and assign to B1
    if ti == 1 & isempty(hitlist) & ball0(b1i).t(2) == ball0(b2i).t(2) & ...
            (ball0(b2i).x(1) ~= ball0(b2i).x(2) | ball0(b2i).y(1) ~= ball0(b2i).y(2))
        
        vec_b2dir = [ball0(b2i).x(2)-ball0(b2i).x(1) ...
            ball0(b2i).y(2)-ball0(b2i).y(1)]; % Direction vector of B1, this is reference

        % contact position of B1. This is in tangential direction of B2
        % movement direction
        b1pos2 = [ball0(b2i).x(1) ball0(b2i).y(1)] -vec_b2dir/norm(vec_b2dir)*param.ballR*2;
        
        hi = size(hitlist,1) + 1;
        
        hitlist(hi,1) = ball0(b1i).t(2)/2; % Contact Time
        hitlist(hi,2) = b1i; % Ball 1
        hitlist(hi,3) = 1; % 1= Ball-Ball, 2= Cushion
        hitlist(hi,4) = b2i; % Ball 2
        hitlist(hi,5) = b1pos2(1);
        hitlist(hi,6) = b1pos2(2);
        
        hi = size(hitlist,1) + 1;
        hitlist(hi,1) = ball0(b1i).t(2)/2;; % Contact Time
        hitlist(hi,2) = b2i; % Ball 2
        hitlist(hi,3) = 1; % 1= Ball-Ball, 2= Cushion
        hitlist(hi,4) = b1i; % Ball 1
        hitlist(hi,5) = ball0(b2i).x(1);% Contact location x
        hitlist(hi,6) = ball0(b2i).y(1);% Contact location y

    end
    
    
    
    
    
    %% Assign new hit event or next timestep in to the ball route history
    % if hit: replace current point with hit,
    %     - add current point of ball0 to ball
    %     - replace current point with new hit point
    %
    % otherwise
    %     - add current point of ball0 to ball
    %     - delete current point in ball0
    
    bi_list = [1:3]; % To store which ball didn't have hit, so must set new time manually
    if ~isempty(hitlist) & Tall0(2) >= tc
        
        % get the first time whith hit is detected
        tc = min(hitlist(:,1));
        if  Tall0(2) < tc
            % hit time if after next time step, skip
            disp('warning: hit is after next time step')
        end
        
        % check whether
        % - the time at this event is already in the ball history, for example this time
        %   point will be also available in the next time step
        % - Time is not before time step of other ball. This can happen, since each ball has own time
        % discretization. And then for other ball the next time step can be be ealier than the
        % detected new contact point
        
        % Check whether for current ball more than 1 event has happend
        for bi = 1:3
            hits = sum(hitlist(:,1) == tc & hitlist(:,2) == bi);
            if hits > 1
                error(['ERROR: Ball ',num2str(bi), ' has ',num2str(hits),' at same time.'])
            end
        end
        
        % Just in case that several independent contact partners have exactly at same time hit event
        % e.g Ball-Ball collision has 2 entries at same time for B1 and B2
        % check whether the time at this event is already in the ball history, for example this time
        % point will be also available in the next time step
        
        %% assign new data for hits
        for hi = find(hitlist(:,1) == tc)'
            
            bi = hitlist(hi,2);
            bi_list(bi_list==bi)=[];
            lasthit_ball = bi;
            lasthit_t = tc;
            
            % Check whether time is already in Ball0.
            % If not, add new data.
            % If yes, then overwrite existing data
            
            % Check Distance of current point to next point
            % if distance is too small delete the next point and the time
            % in Tall0
            % checkdist = true if dist > 1mm
            checkdist = 0;
            if length(ball0(bi).x) > 1
                checkdist = sqrt((ball0(bi).x(2)-hitlist(hi,5))^2 + ...
                    (ball0(bi).y(2)-hitlist(hi,6))^2) < 5;
            end
            
            % Why I a was deleteing the second when it was too close??
%             if checkdist & length(ball0(bi).t) > 2
%                 Tall0(2) = []; % Time
%                 ball0(bi).t(2) = [];
%                 ball0(bi).x(2) = [];
%                 ball0(bi).y(2) = [];
%                 
%             end
            
            Tall0(1) = tc; % Time
            
            ball0(bi).t(1) = tc;
            ball0(bi).x(1) = hitlist(hi,5); % x
            ball0(bi).y(1) = hitlist(hi,6); % y
            
            ball(bi).t(end+1,1) = tc;
            ball(bi).x(end+1,1) = hitlist(hi,5); % x
            ball(bi).y(end+1,1) = hitlist(hi,6); % y
            
            %% Hit data
            hit(bi).t(end+1) = hitlist(hi,1);
            if hitlist(hi,3)== 1
                % Ball
                hit(bi).with(end+1) = col(hitlist(hi,4));
            elseif hitlist(hi,3)== 2
                % Cushion
                hit(bi).with(end+1) = num2str(hitlist(hi,4));
            end
            hit(bi).XPos(end+1) = hitlist(hi,5); % hit position X
            hit(bi).YPos(end+1) = hitlist(hi,6); % hit position Y
            
            %% Plot
            if plot_flag & ti >= ti_plot_start
                % Ball Position at hit
                plot(ax, ball(bi).x(end), ball(bi).y(end),'oc', 'tag', 'hlast2')
                plot(ax, ball(bi).x(end)+param.ballcirc(1,:), ...
                    ball(bi).y(end)+param.ballcirc(2,:),'-w', 'tag', 'hlast2')
                drawnow
            end
            
        end
        
        
        %% Assign time to ball without hit event
        % and here we add the time points for the balls without any action in this time
        for bi = bi_list
            % is ball.dt < Total.dT
            ball(bi).t(end+1,1) = Tall0(1);
            ball(bi).x(end+1,1) = interp1(ball0(bi).t, ball0(bi).x, Tall0(1)); % x
            ball(bi).y(end+1,1) = interp1(ball0(bi).t, ball0(bi).y, Tall0(1)); % y
            
            if  b(bi).vt1 > 0
                ball0(bi).t(1) = Tall0(1);
                ball0(bi).x(1) = ball(bi).x(end,1);
                ball0(bi).y(1) = ball(bi).y(end,1);
            end
            
            %% Plot
            if plot_flag & ti >= ti_plot_start
                % Ball Position at hit
                plot(ax, ball(bi).x(end), ball(bi).y(end),'oc', 'tag', 'hlast2')
                drawnow
            end
        end
        
        
    else
        
        %% Assign new data for no Ball hit
        % Assign time to all balls without hit event
        % and here we add the time points for the balls without any action in this time
        Tall0(1) = []; % Time
        
        for bi = 1:3

            ball(bi).t(end+1,1) = Tall0(1);
            if  b(bi).vt1 > 0
                ball(bi).x(end+1,1) = interp1(ball0(bi).t, ball0(bi).x, Tall0(1)); % x
                ball(bi).y(end+1,1) = interp1(ball0(bi).t, ball0(bi).y, Tall0(1)); % y
                
            else
                ball(bi).x(end+1,1) = ball(bi).x(end);
                ball(bi).y(end+1,1) = ball(bi).y(end);
            end
            
            ind = find(ball0(bi).t < Tall0(1)) ;
            if ball0(bi).t(2) <= Tall0(1)
                ball0(bi).x(ind) = [];
                ball0(bi).y(ind) = [];
                ball0(bi).t(ind) = [];
            else
                ball0(bi).t(1) = Tall0(1);
                ball0(bi).x(1) = ball(bi).x(end);
                ball0(bi).y(1) = ball(bi).y(end);
            end
        end
                
    end
    
    %% Update derivatives
    for bi=1:3
        if hit(bi).with(end) ~= '-'
            % First Sort by time
            [~,ind] = sort(ball0(bi).t);
            ball0(bi).t = ball0(bi).t(ind);
            ball0(bi).x = ball0(bi).x(ind);
            ball0(bi).y = ball0(bi).y(ind);
            
            ball0(bi).dt = diff(ball0(bi).t);
            ball0(bi).vx = [diff(ball0(bi).x)./ball0(bi).dt; 0];
            ball0(bi).vy = [diff(ball0(bi).y)./ball0(bi).dt; 0];
            ball0(bi).v  = [sqrt(ball0(bi).vx.^2 + ball0(bi).vy.^2); 0];
        end
    end
    
    for bi=1:3
        for iii = 1:length(hit(bi).t)
            if isempty(find(hit(bi).t(iii)==ball(bi).t))
                disp('no hit time found time ball data')
            end
        end
        
        ind = find(ball0(bi).t < Tall0(1));
        if ~isempty(ind)
%             disp([num2str(bi),':',num2str(ind)])
        end
        
    end
    
     
    %% Plot
    if plot_flag & ti >= ti_plot_start
        for bi = 1:3
            plot(ax, ball(bi).x, ball(bi).y, 'cs', 'Markersize',8, 'tag', 'hlast')
        end
        
        hlast = findobj('tag', 'hlast');
        delete(hlast)
    end
    
    % Check whether time is over
    do_scan = length(Tall0) >= 3;
end

%% Plot
if plot_flag
    hlast = findobj('tag', 'hlast2');
    delete(hlast)
end

% Assign back the shot
for bi = 1:3
    
    SA.Shot(si).Route(bi).t = ball(bi).t;
    SA.Shot(si).Route(bi).x = ball(bi).x;
    SA.Shot(si).Route(bi).y = ball(bi).y;
    
end


