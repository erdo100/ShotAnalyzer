dirs = dir("E:\DaVinciResolveProjects\20221208_OffsetSystem\20230118_2346\*.bbt");
results = [];
results{1} = 'Filename';
results{2} = 'Offset';
results{3} = 'B2 Pos X';
results{4} = 'B2 Pos Y';
results{5} = 'Hit Thickness';
results{6} = 'B1_V';
results{8} = 'B1 pos1 X';
results{9} = 'B1 Pos1 Y';
results{10} = 'B1 pos2 X';
results{11} = 'B1 Pos2 Y';
results{12} = 'B1 pos3 X';
results{13} = 'B1 Pos3 Y';
results{14} = 'B1 pos4 X';
results{15} = 'B1 Pos4 Y';
results{16} = 'B1 Angle 1';
results{17} = 'B1 Angle 2';
results{18} = 'B1 Offset 2';
results{19} = 'B2 Angle';
results{20} = 'B2 CutAngle';
results{21} = 'B1 pos4 X';

for fi = 1:length(dirs)
    if fi == 38
        disp('1')
    end
    filename = [dirs(fi).folder,'\',dirs(fi).name];
    disp(dirs(fi).name)
    
    % Read the file csv format
    data = readmatrix(filename, 'FileType', 'text');
    
    
    B1hit = 'Y';
    B1pos = [];
    
    % Find when B2 is moving:
    iB2_startmove = find(abs(diff(data(3:end,4)))> 5,1,'first')+2;
    
    
    % Shots shall always go with rising B1 X and Y
    % identify shot direction
    dirx = data(iB2_startmove,2)-data(1,2);
    if dirx < 0
        data(:,[2,4,6]) = 2840-data(:,[2,4,6]);
    end
    diry = data(iB2_startmove,3)-data(1,3);
    if diry < 0
        data(:,[3,5,7]) = 1420-data(:,[3,5,7]);
    end
    
    
    %% Calculate with this the velocity of B1
    B1vel = sqrt((data(iB2_startmove,2)-data(1,2))^2 + (data(iB2_startmove,3)-data(1,3))^2)/(data(iB2_startmove,1)-data(1,1));
    disp(['B1vel=',num2str(B1vel),'m/s'])
    
    %% Calculate direction B1
    B1angle = atan2(data(iB2_startmove,3)-data(1,3), data(iB2_startmove,2)-data(1,2))/pi*180;
    disp(['B1angle=',num2str(B1angle),'°'])
    
    %% Calculate B2 Position
    B2pos = [mean(data(1:iB2_startmove,4)), mean(data(1:iB2_startmove,5))];
    
    %% Calculate Hit Fraction
    dirvec = [data(1,2) data(1,3) 0 ] - [data(iB2_startmove-3,2) data(iB2_startmove-3,3) 0];
    hitThickness = 1 - norm(cross([B2pos 0]-[data(1,2) data(1,3) 0], dirvec))/norm(dirvec) / 61.5;
    
    %% Calculate B1 1st Cushion hit
    ii = iB2_startmove-1+find(data(iB2_startmove:end,3) > 1350,1,'first');
    [~, ij] = min(data(ii:ii+10,4));
    ib1_c1 = ii-1+ij;
    B1_Lcus1 = [data(ib1_c1,2) data(ib1_c1,3)];
    B1hit(end+1) = '1';
    
    % [B1_Lcus1(2),ib1_c1] = max(data(i1:i1+10,3));
    % B1_Lcus1(1) = data(i1+ib1_c1-1,2);
    
    disp(['B1_Lcus1=',num2str(B1_Lcus1(1)),'mm'])
    
    %% Calculate B2 direction
    % Calculate B2 Cushion hit
    % What hits first Long or short cushion??
    % Check hit for short cushion first
    ii = iB2_startmove-1+find(data(iB2_startmove:end,4) > 2780,1,'first');
    [~, ib2_2S] = max(data(ii:ii+10,4));
    ib2_2S = ii-1+ib2_2S;

    % Now check if there is ymax value bigger than initial position
    ii = iB2_startmove-1+find(data(iB2_startmove:ib2_2S,5) > 1360,1,'first');
    [b2ymax, ij] = max(data(ii:ib2_2S,5));
    ib2_2L = ii-1+ij;
    
    % Calculate B2 direction
    if ~isempty(ib2_2L)
        B2pos2 = [data(ib2_2L,4),data(ib2_2L,5)];
    else
        B2pos2 = [data(ib2_2S,4),data(ib2_2S,5)];
    end
    
    B2angle = atan2(B2pos2(2)-B2pos(2), B2pos2(1)-B2pos(1))/pi*180;
    disp(['B2angle=',num2str(B2angle),'°'])
    
    %% Identify B1 cushion hits
    % where it hits first the next cushion?
    % L - S - L
    % L - L - S
    % L - L - L
    ib1_c1L1 =[];
    ib1_c2S2 =[];
    ib1_c2L3 = [];
    ib1_c2 = [];

    
    % Ball 1 Cushion 2
    % Check Short cushion 2 hit 2
    ii = ib1_c1-1+find(data(ib1_c1:end,2) > 2780,1,'first');
    [~, ib1_c2S2] = max(data(ii:ii+10,2));
    ib1_c2S2 = ii-1+ib1_c2S2;
    % Check Long cushion 3 hit 2
    ii = ib1_c1-1+find(data(ib1_c1:end,3) < 60,1,'first');
    [~, ib1_c2L3] = min(data(ii:min(ii+20,size(data,1)),3));
    ib1_c2L3 = ii-1+ib1_c2L3;
    
    
    [ib1_c2, ind] = sort([ib1_c2S2, ib1_c2L3]);
    ib1_c2 = ib1_c2(1);
    hit = '23';
    B1hit(end+1) = hit(ind(1));
    
    
    ib1_c3L3 = 1e6;
    ib1_c3S2 = 1e6;
    ib1_c3L1 = 1e6;
    
    % Ball 1 Cushion 3
    % Check Long cushion 3
    if B1hit(end) ~= '3'
        ii = ib1_c2-1+find(data(ib1_c2:end,3) < 40,1,'first');
        [~, ij] = min(data(ii:ii+10,3));
        if ~isempty(ij)
            ib1_c3L3 = ii-1+ij;
        end
    end
    
    % Check short cushion 2
    if B1hit(end) ~= '2'
        ii = ib1_c2-1+find(data(ib1_c2:end,2) > 2800,1,'first');
        [~, ij] = max(data(ii:ii+10,2));
        if ~isempty(ij)
            ib1_c3S2 = ii-1+ij;
        end
    end
    
    % Check Long cushion 1
    ii = ib1_c2-1+find(data(ib1_c2:end,3) > 1380,1,'first');
    [~, ij] = min(data(ii:ii+10,3));
    if ~isempty(ij)
        ib1_c3L1 = ii-1+ij;
    end
    
    [ib1_c3, ind] = sort([ib1_c3L3, ib1_c3S2, ib1_c3L1]);
    ib1_c3 = ib1_c3(1);
    hit = '321';
    B1hit(end+1) = hit(ind(1));
    ib1 = [1 ib1_c1 ib1_c2 ib1_c3];
    
    disp(B1hit)
    B1pos = [data(ib1,2),data(ib1,3)];
    
    %% B1 angle before C2
    B1angle(2) = -atan2(data(ib1(3),3)-data(ib1(3)-30,3), data(ib1(3),2)-data(ib1(3)-30,2))/pi*180;
    disp(['B1angle2=',num2str(B1angle(2)),'°'])
    
    
    %% Output
    % filename
    col = 1;
    results{fi+1,col} = dirs(fi).name;
    % Offset
    col = col+1;
    results{fi+1,col} = 1590/tan(B1angle(1)/180*pi)/2840*8;
    
    % Ball2 Position
    col = col+1;
    results{fi+1,col} = 8 - B2pos(1)/2840*8;
    
    col = col+1;
    results{fi+1,col} = 4 - B2pos(2)/2840*8;
    
    % Ball 2 hit thickness
    col = col+1;
    results{fi+1,col} = hitThickness;
    
    % Ball1 velocity
    col = col+1;
    results{fi+1,col} = B1vel;
    
    % Ball hit
    col = col+1;
    results{fi+1,col} = B1hit;
    
    % Ball 1 positions
    for i = 1:4
        for j = 1:2
            col = col+1;
            results{fi+1,col} = B1pos(i,j)/2840*8;
        end
    end
    
    % Ball 1 angles
    for i = 1:length(B1angle)
        col = col+1;
        results{fi+1,col} = B1angle(i);
    end
    
    % Ball 1 offset before 2nd cushion
    col = col+1;
    results{fi+1,col} = (1420-61.5)/tan(B1angle(2)/180*pi)/2840*8;
    
    % Ball2 angle
    col = col+1;
    results{fi+1,col} = B2angle;
    
    % Ball2 Cutangle
    col = col+1;
    results{fi+1,col} = B1angle(1)-B2angle;
    disp(' ')

    % Ball 1 Position at 3rd cushion Cutangle
    col = col+1;
    results{fi+1,col} = 8 - B1pos(4,1)/2840*8;
    
end

writecell(results,['E:\DaVinciResolveProjects\20221208_OffsetSystem\ShotAnalyzer_Data_23a.xlsx'])