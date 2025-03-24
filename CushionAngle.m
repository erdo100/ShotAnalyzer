function angle = CushionAngle(ci, v1, v2)

v1 = [v1 0]';
v2 = [v2 0]';

if ci == 1
    % 1st cushion
    b = [0 1 0]';
elseif ci == 2
    % 2nd cushion
    b = [-1 0 0]';
elseif ci == 3
    % 3rd cushion
    b = [0 -1 0]';
elseif ci == 4
    % 4th cushion
    b = [1 0 0]';
end
    
angle(1) = angle_vector(v1,-b);
angle(2) = angle_vector(v2,b);

% Check whether we have negative angle
% Check tangent speed is in same direction
if ci == 1
    % 1st cushion
    if sign(v1(1)) ~=  sign(v2(1)) 
        angle(1) = -angle(1);
    end
elseif ci == 2
    % 2nd cushion
    if sign(v1(2)) ~=  sign(v2(2)) 
        angle(1) = -angle(1);
    end
elseif ci == 3
    % 3rd cushion
    if sign(v1(1)) ~=  sign(v1(1)) 
        angle(1) = -angle(1);
    end
elseif ci == 4
    % 4th cushion
    if sign(v1(2)) ~=  sign(v2(2)) 
        angle(1) = -angle(1);
    end
end

    