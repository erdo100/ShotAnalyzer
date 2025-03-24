function hit = eval_Point_and_Kiss_Control(si, hit)

global SA param
cols = 'WYR';

% Get B1,B2,B3 from ShotList
[b1b2b3, b1i, b2i, b3i] = str2num_B1B2B3(SA.Table.B1B2B3{si});

% ball(1) is B1
for bi = 1:3
    ball(bi) = SA.Shot(si).Route0(b1b2b3(bi));
end

%% evaluate point, kiss, and timing of those
hit = eval_kiss(hit, b1i, b2i, b3i);
% hit(1).Point = pointtime;
% hit(1).Kiss = kisstime;
% hit(1).Tready = b1b23C_time;

%% Calculate the Ball Ball Distance
% List of Ball-Ball collisions
BB = [1 2; 1 3; 2 3];

Tall = unique([ball(1).t; ball(2).t; ball(3).t]);


for bbi = 1:3
    bx1 = BB(bbi,1);
    bx2 = BB(bbi,2);
    
    B1x = interp1(ball(bx1).t, ball(bx1).x, Tall);
    B1y = interp1(ball(bx1).t, ball(bx1).y, Tall);
    B2x = interp1(ball(bx2).t, ball(bx2).x, Tall);
    B2y = interp1(ball(bx2).t, ball(bx2).y, Tall);
    
    
    CC(bbi).dist = sqrt((B2x - B1x).^2 + (B2y - B1y).^2);
end

% evaluate kiss thickness
if hit(b1i).Kiss == 1
    ei = find(hit(b1i).t == hit(b1i).Tkiss,1);
    hit(b1i).KissDistB1 = hit(b1i).Fraction(ei)*param.ballR*2;
else
    % No B1-B2 kiss, evaluate closest passage
    
    % first time index after B1-B2 hit
    ind1 = find(Tall > hit(b1i).TB2hit,1);
    % first maximium after b1-B2 hit. 
    ind2 = ind1+find(diff(CC(1).dist(ind1:end)) < 0,1);
    % If there was Point, then that is the limitation of time
    ind3 = min([find(Tall >= hit(b1i).Tkiss,1) ...
        find(Tall >= hit(b1i).Tpoint,1) ...
        find(Tall >= hit(b1i).Tfailure,1) ...
        length(Tall)]);
    
    % Now look for closest approach B1 to B2
    hit(b1i).KissDistB1 = min(CC(1).dist(ind2:ind3));

%     CenterCenterVector = [(B2x - B1x) (B2y - B1y) zeros(length(Tall),1)];
%     
%     VelocityVectors= [(diff(B2x)-diff(B1x))./diff(Tall) ...
%         (diff(B2y)-diff(B1y))./diff(Tall) ...
%         zeros(length(Tall)-1,1); 0 0 0];
%     
%     alpha = acos(dot(CenterCenterVector,VelocityVectors,2)./ ...
%         (vecnorm(CenterCenterVector').*vecnorm(VelocityVectors'))')*180/pi;
%     
%     CCdist = sin(alpha*pi/180)*param.ballR*2;
%     
end


%% Point accuracy
% new with sign:
% positive when B3 is right and B1 is passing from left
% negative when B3 is left and B1 is passing from right

% CALCULATE VECTOR PRODUCT:
% if new vector looking up, z=positive ==> passing from left
% if new vector looking up, z=negative ==> passing from right

if hit(b1i).Point
%    hit(b1i).PointDist = (1-hit(b1i).Fraction(hi))*param.ballR*2;

    hitsign = 0;
    % With Point ==> Evaluate hit fraction
    hi1 = find(hit(b1i).with == SA.Table.B1B2B3{si}(3),1,'first');
    hi3 = find(hit(b3i).with == SA.Table.B1B2B3{si}(1),1,'first');
    
% first check whether 100% hit, hit fraction exactly = 1
    if hit(b1i).Fraction(hi1) == 1
        hitsign = 0;
    else
        
        t1i = find(ball(b3i).t  < hit(b1i).t(hi1),1,'last');
        t3i = find(ball(b3i).t  < hit(b1i).t(hi1),1,'last');
    
        v1 = [hit(b1i).XPos(hi1)-ball(1).x(t1i) ...
            hit(b1i).YPos(hi1)-ball(1).y(t1i) ...
            0];
        
        v2 = [hit(b3i).XPos(hi3)-hit(b1i).XPos(hi1) ...
            hit(b3i).YPos(hi3)-hit(b1i).YPos(hi1) ...
            0];
        
        v3 = cross(v2,v1);
        if v3(3) > 0
            hitsign = 1;
        else
            hitsign = -1;
        end
    end
    hit(b1i).PointDist = hitsign * (1-hit(b1i).Fraction(hi1))*param.ballR*2;
    
else
    % No Point
    % closest B1-Center/B3-Center distance after B2 hit & 3rd cushion
    ind = find(Tall > hit(b1i).Tready);
    if sum(ind) > 0
        [PointDist, imin] = min(CC(2).dist(ind));
        
        t1i = find(ball(1).t  <= Tall(ind(imin)),1,'last');
        t3i = find(ball(3).t  <= Tall(ind(imin)),1,'last');
        
        v1 = [ball(1).x(t1i)-ball(1).x(t1i-1) ...
            ball(1).y(t1i)-ball(1).y(t1i-1) ...
            0];
        
        v2 = [ball(3).x(t3i)-ball(1).x(t1i) ...
            ball(3).y(t3i)-ball(1).y(t1i) ...
            0];
        
        v3 = cross(v2,v1);
        if v3(3) > 0
            hitsign = 1;
        else
            hitsign = -1;
        end
    
        hit(b1i).PointDist = hitsign * PointDist;
        
    else
        hit(b1i).PointDist = 3000;
    end
    

end




