function Shot = correct_velocity(Shot)

disp(Shot)

% we try to correct noise and velocity jumps
% sweep over several points and look for velocity changes
% if velocity change drop -X % is detected activate
% if in next 5 timesteps a velocity
%     increase +XX% & decrease to old value is with tolerance
% is detected, then reposition position values based on new velocities
tsteps = length(Shot.t);
dx = diff(Shot.x);
dy = diff(Shot.y);
dt = diff(Shot.t);

ds = sqrt((Shot.x(2:end)-Shot.x(1:end-1)).^2 + (Shot.y(2:end) - Shot.y(1:end-1)).^2)/1000;
vel = ds./dt;

ind = find(dt > mean(dt));

figure
plot( Shot.t, Shot.x,'-')
hold on
grid on
plot( Shot.t, Shot.y,'-')

plot( Shot.t(ind), Shot.y(ind),'o')

check = 1;
while check
    
    %% Identify single extremes
    dev = vel_mvmean - vel;
    ind = find(dev < -0.5,1,'first');
    
    if ~isempty(ind)
        subplot(3,1,1)
        plot( Shot.t(ind), dx(ind), 'o')
        plot( Shot.t(ind), dy(ind), 'o')
        
        subplot(3,1,2)
        plot( Shot.t(ind), vel(ind), 'o')
        
        subplot(3,1,3)
        plot( Shot.t(ind), Shot.x(ind), 'o')
        plot( Shot.t(ind), Shot.y(ind), 'o')
        
        if abs((dx(ind+1)-dx(ind-1))/dx(ind-1)) < 0.1 | abs((dy(ind+1)-dy(ind-1))/dy(ind-1)) < 0.1
            dx1 = (dx(ind+1)+dx(ind-1))/2;
            dy1 = (dy(ind+1)+dy(ind-1))/2;
            Shot.x(ind) = Shot.x(ind-1) + dx1;
            Shot.y(ind) = Shot.y(ind-1) + dy1;
            
            dx = diff(Shot.x);
            dy = diff(Shot.y);
            
            ds = sqrt((Shot.x(2:end)-Shot.x(1:end-1)).^2 + (Shot.y(2:end) - Shot.y(1:end-1)).^2)/1000;
            dt = diff(Shot.t);
            vel = ds./dt;
            
            subplot(3,1,1)
            plot( Shot.t(ind), dx(ind), 'x')
            plot( Shot.t(ind), dy(ind), 'x')
            
            subplot(3,1,2)
            plot( Shot.t(ind), vel(ind), 'x')
            
            subplot(3,1,3)
            plot( Shot.t(ind), Shot.x(ind), 'x')
            plot( Shot.t(ind), Shot.y(ind), 'x')
            
            
            figure
            plot( Shot.t, Shot.y, '-x')
        end
    end
    
end

