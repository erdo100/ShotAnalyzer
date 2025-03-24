function SA = read_gamefile(filepath)

% Read the file in json format
json = jsondecode(fileread(filepath));
% Scaling factors
tableX = 2840;
tableY = 1420;

% extract filename
[~,filename,~] = fileparts(filepath);

si = 0;
SA.Shot = [];
check = 0;

for seti = 1:length(json.Match.Sets)
    if iscell(json.Match.Sets(seti).Entries)
        json.Match.Sets(seti).Entries_old = json.Match.Sets(seti).Entries;
        json.Match.Sets(seti).Entries = [];
        for entryi = 1:length(json.Match.Sets(seti).Entries_old)
            json.Match.Sets(seti).Entries(1).IsMarked = json.Match.Sets(seti).Entries_old{1}.IsMarked;
            json.Match.Sets(seti).Entries(1).PathTracking = json.Match.Sets(seti).Entries_old{1}.PathTracking;
            json.Match.Sets(seti).Entries(1).PathTrackingId = json.Match.Sets(seti).Entries_old{1}.PathTrackingId;
            json.Match.Sets(seti).Entries(1).PathTrackingOverview = json.Match.Sets(seti).Entries_old{1}.PathTrackingOverview;
            json.Match.Sets(seti).Entries(1).Scoring = json.Match.Sets(seti).Entries_old{1}.Scoring;
            json.Match.Sets(seti).Entries(1).StartPosition = json.Match.Sets(seti).Entries_old{1}.StartPosition;
            json.Match.Sets(seti).Entries(1).UniqueId = json.Match.Sets(seti).Entries_old{1}.UniqueId;
        end
    end
    for entryi = 1:length(json.Match.Sets(seti).Entries)
        if json.Match.Sets(seti).Entries(entryi).PathTrackingId > 0
            si = si + 1;
            Selected{si,1} = false;
            Filename{si,1} = filename;
            GameType{si,1} = json.Match.GameType;
            Player1{si,1} = json.Player1;
            Player2{si,1} = json.Player2;
            Set{si,1} = seti;
            ShotID{si,1} = json.Match.Sets(seti).Entries(entryi).PathTrackingId;
            
            CurrentInning{si,1} = json.Match.Sets(seti).Entries(entryi).Scoring.CurrentInning;
            CurrentSeries{si,1} = json.Match.Sets(seti).Entries(entryi).Scoring.CurrentSeries;
            CurrentTotalPoints{si,1} = json.Match.Sets(seti).Entries(entryi).Scoring.CurrentTotalPoints;
            Point{si,1} = json.Match.Sets(seti).Entries(entryi).Scoring.EntryType;
            
            playernum = json.Match.Sets(seti).Entries(entryi).Scoring.Player;
            if playernum == 1
                Player{si,1} = Player1{si,1};
            elseif playernum == 2
                Player{si,1} = Player2{si,1};
            end
            ErrorID{si,1} = -1;
            ErrorText{si,1} = 'only read in';
            Interpreted{si,1} = 0;
            Mirrored{si,1} = 0;
            
            for bi = 1:3
                clen = length(json.Match.Sets(seti).Entries(entryi).PathTracking.DataSets(bi).Coords);
                SA.Shot(si).Route(bi).t = zeros(clen,1);
                SA.Shot(si).Route(bi).x = zeros(clen,1);
                SA.Shot(si).Route(bi).y = zeros(clen,1);
                for ci = 1:clen
                    SA.Shot(si).Route(bi).t(ci) = json.Match.Sets(seti).Entries(entryi).PathTracking.DataSets(bi).Coords(ci).DeltaT_500us*0.0005;
                    SA.Shot(si).Route(bi).x(ci) = json.Match.Sets(seti).Entries(entryi).PathTracking.DataSets(bi).Coords(ci).X*tableX;
                    SA.Shot(si).Route(bi).y(ci) = json.Match.Sets(seti).Entries(entryi).PathTracking.DataSets(bi).Coords(ci).Y*tableY;
                    check = 1;
                end
                
                SA.Shot(si).Route0(bi) = SA.Shot(si).Route(bi);
                SA.Shot(si).hit=0;
            end
            
            
        end
    end
end

%
% for si = 1:length(SA.Shot)
%     for bi = 1:3
%         if length(SA.Shot(si,1).Route(bi).t) == 1
%             SA.Shot(si,1).Route(bi).t(2,1) = SA.Shot(si,1).Route(bi).t(1);
%             SA.Shot(si,1).Route(bi).x(2,1) = SA.Shot(si,1).Route(bi).x(1);
%             SA.Shot(si,1).Route(bi).y(2,1) = SA.Shot(si,1).Route(bi).y(1);
%         end
%     end
% end


if check
    SA.Table = table(Selected, ShotID, Mirrored, Filename, GameType, Interpreted, Player, ErrorID, ErrorText, ...
        Set, CurrentInning, CurrentSeries, CurrentTotalPoints, Point);
else
    SA = [];
end