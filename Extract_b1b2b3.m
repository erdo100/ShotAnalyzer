function [b1b2b3, err] = Extract_b1b2b3(shot)

% initialize outputs
b1b2b3num = [];
err.code = [];
err.text = '';

col = 'WYR';

if length(shot.Route(1).t)<2 | length(shot.Route(2).t)<2 | length(shot.Route(3).t)<2
    % if ~isempty([tb2i2 tb2i3]) & ~isempty([tb2i2 tb2i3])
    % B1, B2 , B3 moved
    b1b2b3 = 'WYR';
    err.code = 2;
    err.text = ['Empty Data (',mfilename,'Extract_b1b2b3.m)'];
    
    return
    
end

[t, b1b2b3num] = sort([shot.Route(1).t(2), shot.Route(2).t(2), shot.Route(3).t(2)]);

% Temporary assumed B1B2B3 order
b1it = b1b2b3num(1);
b2it = b1b2b3num(2);
b3it = b1b2b3num(3);

if length(shot.Route(b1it).t) >= 3
    
    % find balls moving until 3rd timestep of first moving ball
    tb2i2 = find(shot.Route(b2it).t(2:end) <= shot.Route(b1it).t(2),1,'last');
    tb3i2 = find(shot.Route(b3it).t(2:end) <= shot.Route(b1it).t(2),1,'last');
    tb2i3 = find(shot.Route(b2it).t(2:end) <= shot.Route(b1it).t(3),1,'last');
    tb3i3 = find(shot.Route(b3it).t(2:end) <= shot.Route(b1it).t(3),1,'last');
    
    if isempty(tb2i2) & isempty(tb3i2) & isempty(tb2i3) & isempty(tb3i3)
        % only B1 moved in first time step for sure
        b1b2b3 = col(b1b2b3num);
        return
    end
    
else
    % No ball moved
    b1b2b3 = col;
    return
end


if (~isempty(tb2i2) | ~isempty(tb2i3)) & (isempty(tb3i2) & isempty(tb3i3))
    % if ~isempty([tb2i2 tb2i3]) & isempty([tb3i2 tb3i3])
    % B1 and B2 moved
    % B1-B2 vector:
    vec_b1b2 = [shot.Route(b2it).x(1)-shot.Route(b1it).x(1) ...
        shot.Route(b2it).y(1)-shot.Route(b1it).y(1)];
    % B1 direction
    vec_b1dir = [shot.Route(b1it).x(2)-shot.Route(b1it).x(1) ...
        shot.Route(b1it).y(2)-shot.Route(b1it).y(1)];
    % B2 direction
    vec_b2dir = [shot.Route(b2it).x(2)-shot.Route(b2it).x(1) ...
        shot.Route(b2it).y(2)-shot.Route(b2it).y(1)];
    
    angle_b1 = angle_vector(vec_b1b2,vec_b1dir);
    angle_b2 = angle_vector(vec_b1b2,vec_b2dir);
    
    if angle_b2 > 90
        b1b2b3num = [2 1 3];
    end
    
end

if (isempty(tb2i2) & isempty(tb2i3)) & (~isempty(tb3i2) | ~isempty(tb3i3))
    % if isempty([tb2i2 tb2i3]) & ~isempty([tb3i2 tb3i3])
    % B1 and B3 moved
    vec_b1b3 = [shot.Route(b3it).x(1)-shot.Route(b1it).x(1) ...
        shot.Route(b3it).y(1)-shot.Route(b1it).y(1)];
    % B1 direction
    vec_b1dir = [shot.Route(b1it).x(2)-shot.Route(b1it).x(1) ...
        shot.Route(b1it).y(2)-shot.Route(b1it).y(2)];
    % B2 direction
    vec_b3dir = [shot.Route(b3it).x(2)-shot.Route(b3it).x(1) ...
        shot.Route(b3it).y(2)-shot.Route(b3it).y(2)];
    
    angle_b1 = angle_vector(vec_b1b3,vec_b1dir);
    angle_b3 = angle_vector(vec_b1b3,vec_b3dir);
    
    if angle_b3 > 90
        b1b2b3num = [2 3 1];
    end
    
end

b1b2b3 = col(b1b2b3num);


if (~isempty(tb2i2) | ~isempty(tb2i3)) & (~isempty(tb3i2) | ~isempty(tb3i3))
    % if ~isempty([tb2i2 tb2i3]) & ~isempty([tb2i2 tb2i3])
    % B1, B2 , B3 moved
    err.code = 2;
    err.text = ['all balls moved at same time (',mfilename,'Extract_b1b2b3.m)'];
    
    return
    
end

