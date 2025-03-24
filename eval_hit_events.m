function hit = eval_hit_events(hit, si, b1b2b3)

global SA param

% Check of potential events
% These are all detected events for each ball stored in hit(bi).with at times hit(bi).t

% investigate for each ball
b1i = b1b2b3(1);

%% Output other variables
hit(b1i).AimOffsetLL = [];
hit(b1i).AimOffsetSS = [];
hit(b1i).B1B2OffsetLL = [];
hit(b1i).B1B2OffsetSS = [];
hit(b1i).B1B3OffsetLL = [];
hit(b1i).B1B3OffsetSS = [];

for bi = 1:3
   for hi = 1:8
      hit(bi).Type(hi) = NaN;
      hit(bi).FromC(hi) = NaN;
      hit(bi).ToC(hi) = NaN;
      hit(bi).V1(hi) = NaN;
      hit(bi).V2(hi) = NaN;
      hit(bi).Offset(hi) = NaN;
      hit(bi).Fraction(hi) = NaN;
      hit(bi).DefAngle(hi) = NaN;
      hit(bi).CutAngle(hi) = NaN;
      hit(bi).CInAngle(hi) = NaN;
      hit(bi).COutAngle(hi) = NaN;
      hit(bi).FromCPos(hi) = NaN;
      hit(bi).ToCPos(hi) = NaN;
      hit(bi).FromDPos(hi) = NaN;
      hit(bi).ToDPos(hi) = NaN;
   end
end

%% Evaluate all others
for bi = b1b2b3
   % Go through all hit of this ball
   for hi = 1:length(hit(bi).with)
      % disp(['bi=',num2str(bi),'-hi=',num2str(hi),'-with=',hit(bi).with(hi)])
      if hit(bi).with(hi) == '-'
         % skipping first entries with '-'
         continue
      end
      
      [v, v1, v2, alpha, offset] = Ball_velocity(SA.Shot(si).Route(bi), hit(bi), hi) ;
      
      % Output
      hit(bi).V1(hi) = v(1); % Output
      hit(bi).V2(hi) = v(2); % Output
      hit(bi).Offset(hi) = offset;
            
      if v(1) > 0 | v(2) > 0
         direction = Ball_direction(SA.Shot(si).Route(bi), hit(bi), hi);
      else
         direction = NaN(2,6);
      end
      % Output
      hit(bi).FromC(hi) = direction(2,1);
      hit(bi).FromCPos(hi) = direction(2,2);
      hit(bi).FromDPos(hi) = direction(2,3);
      hit(bi).ToC(hi) = direction(2,4);
      hit(bi).ToCPos(hi) = direction(2,5);
      hit(bi).ToDPos(hi) = direction(2,6);
      
      hit(bi).Type(hi) = 0; % 0=Start 1= Ball, 2=cushion
      
      % Did we hit cushion?
      c2i = find(hit(bi).with(hi) == '1234',1);      
      if ~isempty(c2i)
          % Ball-Cushion hit
         % Output
         hit(bi).Type(hi) = 2; % 1= Ball, 2=cushion
         % angles based on velocities
         Angle = CushionAngle(str2num(hit(bi).with(hi)), v1, v2);
%         CushionAngle = TranslateCushionAngle(alpha, c2i, v1, v2);
         % Output
         hit(bi).CInAngle(hi) = Angle(1);
         hit(bi).COutAngle(hi) = Angle(2);
      end
      
      % then check in case of Ball-Ball hits, whether this is the moving ball
      % Which Ball was hit
      b2i = find(hit(bi).with(hi) == 'WYR',1);
      % Ball-Ball hit
      if ~isempty(b2i)
         % Output
         hit(bi).Type(hi) = 1; % 1= Ball, 2=cushion
      end
      
      % Ball Hit
      if ~isempty(b2i) & v(1) > 0
         % Calculation of hitfraction based on Point / vector distance
         % Point is location of not moving ball B2 p2
         tb10i = find(hit(bi).t(hi) == SA.Shot(si).Route(bi).t,1);
         tb11i = find(hit(bi).t(hi) == SA.Shot(si).Route(b2i).t,1);
         
         if isempty(tb10i)
            disp('No time point found matching to hit event')
         end
         
         pb1 = [SA.Shot(si).Route(bi).x(tb10i) SA.Shot(si).Route(bi).y(tb10i)];
         pb2 = [SA.Shot(si).Route(b2i).x(tb11i) SA.Shot(si).Route(b2i).y(tb11i)];
         
         hib2 = find(hit(b2i).t == hit(bi).t(hi),1);
         [~, v1b2, ~, ~, ~] = Ball_velocity(SA.Shot(si).Route(b2i), hit(b2i), hib2) ;
         
         % Point is location of moving ball B1 p1
         % vector of B1 direction v1
         % |(p2-p1) x v1| /|v1|
         
         % relative velocity before contact current Ball to contact partner
         % ball
         velrel = [v1-v1b2 0];
         
         % Output
         try
            hit(bi).Fraction(hi) = 1-norm(cross([pb2 0]-[pb1 0],velrel))/norm(velrel)/param.ballR/2;
            hit(bi).DefAngle(hi) = acos(sum(v1.*v2)/(norm(v1)*norm(v2)))*180/pi;
         catch
            hit(bi).Fraction(hi) =0;
            hit(bi).DefAngle(hi) = 0;
         end

         % Calculation of Objectball cut angle
         
         [~, ~, b2v2, ~, ~] = Ball_velocity(SA.Shot(si).Route(b2i), hit(b2i), 2) ;

         tb20i = find(hit(bi).t(hi) == SA.Shot(si).Route(bi).t,1);
         tb21i = find(hit(bi).t(hi) == SA.Shot(si).Route(b2i).t,1);
         hit(bi).CutAngle(hi) = acos(sum(v1.*b2v2)/(norm(v1)*norm(b2v2)))*180/pi;
         
      end
   end
end

%% Calculate the shot offset angle direction depending
% calculation based on Ball ball position
% calculation based on ball-Cushionhit direction
b1i = b1b2b3(1);
b2i = b1b2b3(2);
b3i = b1b2b3(3);

% Aimdirection Offset LL
if length(hit(b1i).YPos) >= 2
    if abs(hit(b1i).YPos(1)-hit(b1i).YPos(2)) > eps
        % Angle of Shot direction in diamonds offset read on long cushion
        hit(b1i).AimOffsetLL = abs((hit(b1i).XPos(2)-hit(b1i).XPos(1))/(hit(b1i).YPos(2)-hit(b1i).YPos(1))*...
        (param.size(1)+2*param.diamdist))*4/param.size(1);
    end
end

% Aimdirection Offset SS
if length(hit(b1i).XPos) >= 2
    if abs(hit(b1i).XPos(1)-hit(b1i).XPos(2)) > eps
        hit(b1i).AimOffsetSS = abs((hit(b1i).YPos(2)-hit(b1i).YPos(1))/(hit(b1i).XPos(2)-hit(b1i).XPos(1))*...
            (param.size(2)+2*param.diamdist))*4/param.size(1);
    end
end

if length(hit(b1i).with) >= 2
    if ismember(hit(b1i).with(2), 'WYR')
        % if we hit first ball, then offset value has a sign depending on
        % the B2 movement direction
        directL = sign(hit(b2i).XPos(1)-hit(b1i).XPos(2))*sign(hit(b1i).XPos(1)-hit(b1i).XPos(2));
        directS = sign(hit(b2i).YPos(1)-hit(b1i).YPos(2))*sign(hit(b1i).YPos(1)-hit(b1i).YPos(2));
        hit(b1i).AimOffsetLL = hit(b1i).AimOffsetLL*directL;
        hit(b1i).AimOffsetSS = hit(b1i).AimOffsetSS*directS;
    end
end
if length(hit(b1i).with) >= 3
    if ismember(hit(b1i).with(2), '1234')
        % if we hit first a cushion, then negative is when have given
        % counter effet
        directS = sign(( hit(b1i).YPos(2)-hit(b1i).YPos(1) ) - (hit(b1i).YPos(2)-hit(b1i).YPos(3)));
        directL = sign(( hit(b1i).XPos(2)-hit(b1i).XPos(1) ) - (hit(b1i).XPos(2)-hit(b1i).XPos(3)));
        hit(b1i).AimOffsetLL = hit(b1i).AimOffsetLL*directL;
        hit(b1i).AimOffsetSS = hit(b1i).AimOffsetSS*directS;
    end
end


% Position Offset LL B1-B2
if abs(hit(b2i).YPos(1)-hit(b1i).YPos(1)) > eps
    hit(b1i).B1B2OffsetLL = abs((hit(b2i).XPos(1)-hit(b1i).XPos(1))/(hit(b2i).YPos(1)-hit(b1i).YPos(1))*...
        (param.size(1)+2*param.diamdist))*4/param.size(1);
else
    hit(b1i).B1B2OffsetLL = 99;
end

% Position Offset SS B1-B2
if abs(hit(b2i).XPos(1)-hit(b1i).XPos(1)) > eps
hit(b1i).B1B2OffsetSS = abs((hit(b2i).YPos(1)-hit(b1i).YPos(1))/(hit(b2i).XPos(1)-hit(b1i).XPos(1))*...
    (param.size(2)+2*param.diamdist))*4/param.size(1);
else
    hit(b1i).B1B2OffsetSS = 99;
end

% Position Offset LL B1-B3
if abs(hit(b3i).YPos(1)-hit(b1i).YPos(1)) > eps
hit(b1i).B1B3OffsetLL = abs((hit(b3i).XPos(1)-hit(b1i).XPos(1))/(hit(b3i).YPos(1)-hit(b1i).YPos(1))*...
    (param.size(1)+2*param.diamdist))*4/param.size(1); 
else
    hit(b1i).B1B3OffsetLL = 99;
end

% Position Offset SS B1-B3
if abs(hit(b3i).XPos(1)-hit(b1i).XPos(1)) > eps
hit(b1i).B1B3OffsetSS = abs((hit(b3i).YPos(1)-hit(b1i).YPos(1))/(hit(b3i).XPos(1)-hit(b1i).XPos(1))*...
    (param.size(2)+2*param.diamdist))*4/param.size(1);
else
    hit(b1i).B1B3OffsetSS = 99;
end


%% Calculate Inside/Outside shot
% calculated only when 
%   - first hit is to ball
%   - second hit is cushion
% Calculate vectors
% v1 = P1:B1-Position to P3:first cushion hit
% v2 = P2: B1-B2-Hit Position to P3:B1-cushion hit position
% v3 = P3: cushion hit position to P4:next hit position B1 direction after cushion hit

if length(hit(b1i).Type) >= 2
    if hit(b1i).Type(2) == 2
        % Shot was cushion first
        hit(b1i).with(1) = 'B';
        
    elseif length(hit(b1i).t) >= 4 & hit(b1i).Type(2) == 1 & hit(b1i).Type(3) == 2
        P1 = [hit(b1i).XPos(1) hit(b1i).YPos(1) 0]';
        P2 = [hit(b1i).XPos(2) hit(b1i).YPos(2) 0]';
        P3 = [hit(b1i).XPos(3) hit(b1i).YPos(3) 0]';
        P4 = [hit(b1i).XPos(4) hit(b1i).YPos(4) 0]';
        
        v1 = P1-P3;
        v2 = P2-P3;
        v3 = P4-P3;
        
        a1 = acos(sum(v1.*v3)/(norm(v1)*norm(v3)))*180/pi;
        a2 = acos(sum(v2.*v3)/(norm(v2)*norm(v3)))*180/pi;
        
        if a1 <= a2
            % Shot is internal
            hit(b1i).with(1) = 'I';
        else
            % Shot is internal
            hit(b1i).with(1) = 'E';
        end
    end
end
