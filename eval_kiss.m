function hit = eval_kiss(hit, b1i, b2i, b3i)
% Eval for Point
%     Eval for Cusions numbers
%     Eval closest B1-B2 passage for kiss risk
%     Eval closest B1-B3 passage for kiss risk
%     Eval center/Center Gap of point
%     Eval Fuchs

% Eval for Miss
%     Eval Kisses
%     Eval closest B1-B3 Center/Center passage for point

% Point time
% Time when Hit B2 and hit 3C

ballcolor = 'WYR';

% First Hit B1 to B3
b1b3i = find(hit(b1i).with == ballcolor(b3i));

% All Hits B1 to B2
b1b2i = find(hit(b1i).with == ballcolor(b2i));

% All Hits B2 to B3
b2b3i = find(hit(b2i).with == ballcolor(b3i));

% All B1 cushion hits
b1cushi = find(hit(b1i).with == '1' | hit(b1i).with == '2' | ...
    hit(b1i).with == '3' | hit(b1i).with == '4');


% initialization done in Extract_Events.m
% hit(b1i).Point = 0;
% hit(b1i).Kiss = 0;
% hit(b1i).Fuchs = 0;
% hit(b1i).Tpoint = 1000;
% hit(b1i).Tkiss = 1000;
% hit(b1i).Tready = 1000;
% hit(b1i).TB2hit = 1000;
% hit(b1i).Tfailure = 1000;

failure = 0;
pointtime = 1000;
failtime = 1000;
kisstime = 1000;
b1b23C_time = 1000;
b1b2time = 1000;
b1b3time = 1000;

%% Check for Point
% Criteria: 'W' hits first 'Y' 3C and finally 'R'
if ~isempty(b1b3i) & ~isempty(b1b2i) & length(b1cushi) >= 3
    
    if hit(b1i).t(b1b3i(1)) > hit(b1i).t(b1b2i(1)) & ...
            hit(b1i).t(b1b3i(1)) > hit(b1i).t(b1cushi(3))
        % B1B3 hit after B1B2 hit and after 3xB1Cushion
        % Point
        hit(b1i).Point = 1;
        pointtime = hit(b1i).t(b1b3i(1));
    end
end

%% Check for failure, B1 hits B3 too early
% Criteria: 'W' hits 'Y' and 'R' but with less than 3C in between
if ~isempty(b1b3i) & ~isempty(b1b2i) & length(b1cushi) >= 3
    
    if hit(b1i).t(b1b3i(1)) > hit(b1i).t(b1b2i(1)) & ...
            hit(b1i).t(b1b3i(1)) < hit(b1i).t(b1cushi(3))
        % B1B3 hit after B1B2 hit and after 3xB1Cushion
        % Point
        failure = 1;
        failtime = hit(b1i).t(b1b3i(1));
    end
end

%% Check for Kisses
% Kiss detection method: 
%   Point: then check whether before point finished contact was between
%          kiss partner
%   no Point: Then check all hit before before B1 B3?
% first kiss that happened is relevant
kisstime = 1000;
 
if hit(b1i).Point == 1
    if ~isempty(b2b3i) % Check for B2 B3 kiss
       if hit(b2i).t(b2b3i) < pointtime % b2b3 hit happen before Point
           if kisstime > hit(b2i).t(b2b3i) % this happen before latest kisstime
               hit(b1i).Kiss = 3;
               hit(b1i).Fuchs = 1;
               kisstime = hit(b2i).t(b2b3i);
           end
       end
    end
    if length(b1b2i) >= 2 % Check for B2 B1 kiss
       if hit(b1i).t(b1b2i(2)) < pointtime % b2b1 hit happen before Point
           if kisstime > hit(b1i).t(b1b2i(2)) % this happen before latest kisstime
               hit(b1i).Kiss = 1;
               hit(b1i).Fuchs = 1;
               kisstime = hit(b1i).t(b1b2i(2));
           end
       end
    end
else
    if failure == 0
        if ~isempty(b2b3i) % Check for B2 B3 kiss
            if kisstime > hit(b2i).t(b2b3i) % this happen before latest kisstime
                hit(b1i).Kiss = 3;
                kisstime = hit(b2i).t(b2b3i);
            end
        end
        if length(b1b2i) >= 2 % Check for B2 B1 kiss
            if kisstime > hit(b1i).t(b1b2i(2)) % this happen before latest kisstime
                hit(b1i).Kiss = 1;
                kisstime = hit(b1i).t(b1b2i(2));
            end
        end
    end 
    
end


% Time when B1B2 hit and B13C finished
% Criteria: 'W' hits first 'Y' 3C and finally 'R'
if ~isempty(b1b2i) & length(b1cushi) >= 3
    b1b23C_time = max([hit(b1i).t(b1b2i(1)) hit(b1i).t(b1cushi(3))]);
end

if ~isempty(b1b2i)
    b1b2time = hit(b1i).t(b1b2i(1));
end

hit(b1i).Tpoint = pointtime;
hit(b1i).Tkiss = kisstime;
hit(b1i).Tready = b1b23C_time;
hit(b1i).TB2hit = b1b2time;
hit(b1i).Tfailure = failtime;
