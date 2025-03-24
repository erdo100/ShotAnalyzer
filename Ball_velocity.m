function [vt, v1, v2, alpha, offset] = Ball_velocity(ball, hit, ei)
% Calculates angles and velocities
% time just before Hit, or from beginning at first
% Time just before next hit, or to the end at last
   imax = 10;

if hit.with(ei) ~= '-'
   ti = interp1(ball.t, [1:length(ball.t)], hit.t(ei),'nearest');
   
   if ei > 1
      ti_before = interp1(ball.t, [1:length(ball.t)], hit.t(ei-1),'nearest');
   else
      ti_before = 1;
   end
   
   if ei+1 <= length(hit.t)
      ti_after = interp1(ball.t, [1:length(ball.t)], hit.t(ei+1),'nearest');
   else
      ti_after = length(ball.t);
   end
   
   
   % Before condition
   it = min(ti - ti_before, imax);
   if it >= 1 | ei == 1
      if ei == 1
         ind = [2 4];
      else
         ind = [ti-it ti];
      end
      
      % Calculate v1 = v before
      dx = sqrt(diff(ball.x(ind)).^2 + diff(ball.y(ind)).^2);
      dt = diff(ball.t(ind));
      
      vt0 = dx ./ dt;
      vx = diff(ball.x(ind)) ./dt;
      vy = diff(ball.y(ind)) ./dt;
      
      if (vt0) > 0
         vt(1) = (vt0/1000);
         v1 = [(vx/1000) (vy/1000)];
         alpha(1) = atan2(vx, vy)*180/pi;
         
      else
         vt(1) = (vt0/1000);
         v1 = [0 0];
         alpha(1) = NaN;
         
      end
      
   else
      vt(1) = 0;
      v1 = [0 0];
      alpha(1) = NaN;
   end
   
   % Calculate v2 = v fter
   it = min(ti_after - ti, imax);
   if it >= 1
      ind = [ti_after-it ti_after];
      dx = sqrt(diff(ball.x(ind)).^2 + diff(ball.y(ind)).^2);
      dt = diff(ball.t(ind));
      
      vt0 = dx ./ dt;
      vx = diff(ball.x(ind)) ./dt;
      vy = diff(ball.y(ind)) ./dt;
      
      if (vt0) > 0
         vt(2) = (vt0/1000);
         v2 = [(vx/1000) (vy/1000)];
         % Calculate direction
         alpha(2) = (atan2(vx, vy))*180/pi;
         
      else
         vt(2) = (vt0/1000);
         v2 = [0 0];
         alpha(2) = NaN;
      end
   else
      vt(2) = 0;
      v2 = [0 0];
      alpha(2) = NaN;
   end
   
   % Calculate offset
   p1 = [ball.x(ti) ball.y(ti)];
   p2 = [ball.x(ti_after) ball.y(ti_after)];
   
   % Point is location of moving ball B1 p1
   % vector of B1 direction v1
   % |(p2-p1) x v1| /|v1|
   % Output
   if norm(v1) > 0
      offset = norm(cross([p2 0]-[p1 0],[v2 0]))/norm([v2 0]);
   else
      offset = NaN;
   end
else
   vt(2) = 0;
   vt(1) = 0;
   alpha(1) = 0;
   alpha(2) = 0;
   v1 = [0 0];
   v2 = [0 0];
   
   
end