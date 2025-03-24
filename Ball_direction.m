function direction = Ball_direction(Route, hit, ei)

global param

direction = zeros(2, 6); % fromCushion# PosOn PosThrough toCushion# PosOn PosThrough


% get all points just before current hit and select the points at the end
imax = 10; % max node number to be considered
if ei == 1 & length(hit.t) > 1
   t1i = find(Route.t >= hit.t(ei) & Route.t <= hit.t(ei+1),imax, 'first' );
   t2i = find(Route.t >= hit.t(ei) & Route.t <= hit.t(ei+1),imax, 'last' );
elseif ei < length(hit.t)
   t1i = find(Route.t >= hit.t(ei-1) & Route.t <= hit.t(ei),imax, 'last' );
   t2i = find(Route.t >= hit.t(ei) & Route.t <= hit.t(ei+1),imax, 'last' );
else
   t1i = find(Route.t >= hit.t(ei),imax, 'first' );
   t2i = find(Route.t >= hit.t(ei),imax, 'last' );
end


% Points to be used for Extrapolate to cushions
% Initial direction directly after hit event
pstart(1,:) = [Route.x(t1i(1)) Route.y(t1i(1))]; % X/Y at start
pend(1,:) = [Route.x(t1i(end)) Route.y(t1i(end))]; % X/Y at just after start

% direction just before next event
pstart(2,:) = [Route.x(t2i(1)) Route.y(t2i(1))]; % X/Y just before end
pend(2,:) = [Route.x(t2i(end)) Route.y(t2i(end))]; % X/Y at end


for i = 1:2
   % i=1:  Direction at start
   % i=2:  Direction at end
   
   p1 = pstart(i,:); 
   p2 = pend(i,:);   
   
   
   % Projection on cushion
   if sum(isnan([p1 p2])) == 0
      % Catch error when both Interpolation ranges are equal
      if p1(2) ~= p2(2)
         xon1 = interp1( [p1(2) p2(2)], [p1(1) p2(1)], param.ballR, 'linear','extrap'); % cushion1
         xthrough1 = interp1( [p1(2) p2(2)], [p1(1) p2(1)], -param.diamdist, 'linear','extrap'); % cushion1
         xon3 = interp1( [p1(2) p2(2)], [p1(1) p2(1)], param.size(1) - param.ballR, 'linear','extrap'); % cushion3
         xthrough3 = interp1( [p1(2) p2(2)], [p1(1) p2(1)], param.size(1) + param.diamdist, 'linear','extrap'); % cushion1
      else
         xon1 = p1(1);
         xon3 = p1(1);
         xthrough1 = p1(1);
         xthrough3 = p1(1);
         
      end
      
      % Catch error when both Interpolation ranges are equal
      if p1(1) ~= p2(1)
         yon2 = interp1( [p1(1) p2(1)], [p1(2) p2(2)], param.size(2) - param.ballR, 'linear','extrap'); % cushion2
         yon4 = interp1( [p1(1) p2(1)], [p1(2) p2(2)], param.ballR, 'linear','extrap'); % cushion4
         ythrough2 = interp1( [p1(1) p2(1)], [p1(2) p2(2)], param.size(2) + param.diamdist, 'linear','extrap'); % cushion2
         ythrough4 = interp1( [p1(1) p2(1)], [p1(2) p2(2)], -param.diamdist, 'linear','extrap'); % cushion4
      else
         yon2 = p1(2);
         yon4 = p1(2);
         ythrough2 = p1(2);
         ythrough4 = p1(2);
      end
      
%       hl = plot([p1(1) p2(1)],[p1(2) p2(2)],'ok-');
%       delete(hl);
%       
      % is on cushion?
      % on cushion 1
      if xon1 >= param.ballR & xon1 <= param.size(2) - param.ballR
         % decide is it from or to?
         if p2(2) < p1(2)
            % moving from p1 to p2 in dir to 1
            direction(i,4) = 1;
            direction(i,5) = xon1;
            direction(i,6) = xthrough1;
         else
            % moving from p1 to p2 in dir away from1
            direction(i,1) = 1;
            direction(i,2) = xon1;
            direction(i,3) = xthrough1;
         end
      end
      
      % on cushion 3
      if xon3 >= param.ballR & xon3 <= param.size(2) - param.ballR
         % decide is it from or to?
         if p1(2) < p2(2)
            % moving from p1 to p2 in dir to 3
            direction(i,4) = 3;
            direction(i,5) = xon3;
            direction(i,6) = xthrough3;
            
         else
            % moving from p1 to p2 in dir from 3
            direction(i,1) = 3;
            direction(i,2) = xon3;
            direction(i,3) = xthrough3;
            
         end
      end
      
      % on cushion 2
      if yon2 >= param.ballR & yon2 <= param.size(1) - param.ballR
         % decide is it from or to?
         if p1(1) < p2(1)
            % moving from p1 to p2 in dir to 2
            direction(i,4) = 2;
            direction(i,5) = yon2;
            direction(i,6) = ythrough2;
            
         else
            % moving from p1 to p2 in dir from 2
            direction(i,1) = 2;
            direction(i,2) = yon2;
            direction(i,3) = ythrough2;
            
         end
      end
      
      % on cushion 4
      if yon4 >= param.ballR & yon4 <= param.size(1) - param.ballR
         % decide is it from or to?
         if p2(1) < p1(1)
            % moving from p1 to p2 in dir to 4
            direction(i,4) = 4;
            direction(i,5) = yon4;
            direction(i,6) = ythrough4;
            
         else
            % moving from p1 to p2 in dir from 4
            direction(i,1) = 4;
            direction(i,2) = yon4;
            direction(i,3) = ythrough4;
            
         end
      end
      
   else
      disp('')
      
   end
end

%direction(:,[2 3 5 6]) = direction(:,[2 3 5 6]);%*8/param.size(2);

