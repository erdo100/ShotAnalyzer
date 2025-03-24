function SA = read_BBTfile(filepath)

% Read the file csv format
data = readmatrix(filepath, 'FileType', 'text');

% extract filename
[~,filename,~] = fileparts(filepath);

si = 0;
SA.Shot = [];
check = 0;

si = si + 1;
Selected{si,1} = false;
Filename{si,1} = filename;
GameType{si,1} = '3';
ShotID{si,1} = str2num(char(datetime('now','Format','yyMMddHHmmssSSS')));
Player{si,1} = 0;

ErrorID{si,1} = -1;
ErrorText{si,1} = 'only read in';
Interpreted{si,1} = 0;
Mirrored{si,1} = 0;

for bi = 1:3
    SA.Shot(si).Route(bi).t = data(:,1)/1000;
    SA.Shot(si).Route(bi).x = data(:,(bi-1)*2 + 2);
    SA.Shot(si).Route(bi).y = data(:,(bi-1)*2 + 3);
    % Calculate movement distance
    dL = sqrt((SA.Shot(si).Route(bi).x-SA.Shot(si).Route(bi).x(1)).^2 + (SA.Shot(si).Route(bi).y-SA.Shot(si).Route(bi).y(1)).^2);

    ind = find(dL > 5);
    if isempty(ind)
        % Ball didn't move. Only t1 end Tlast is given
        SA.Shot(si).Route(bi).t(2:end-1) = [];
        SA.Shot(si).Route(bi).x(2:end-1) = [];
        SA.Shot(si).Route(bi).y(2:end-1) = [];
    else
        SA.Shot(si).Route(bi).t([2:ind(1)]) = [];
        SA.Shot(si).Route(bi).x([2:ind(1)]) = [];
        SA.Shot(si).Route(bi).y([2:ind(1)]) = [];
    end
    
%    SA.Shot(si).Route(bi) = correct_velocity(SA.Shot(si).Route(bi));

    SA.Shot(si).Route0(bi) = SA.Shot(si).Route(bi);
    check = 1;
end
SA.Shot(si).hit=0;

if check
    SA.Table = table(Selected, ShotID, Mirrored, Filename, GameType, Interpreted, Player, ErrorID, ErrorText);
else
    SA = [];
end

