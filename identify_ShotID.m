function identify_ShotID(ti, Data, ColumnName)
global SA

col_si = strcmp('ShotID', ColumnName);
col_mi = strcmp('Mirrored', ColumnName);

ShotID = cell2mat(Data(ti, col_si));
mirrored = cell2mat(Data(ti, col_mi));

[~, si, ~] = intersect(cell2mat(SA.Table.ShotID)+cell2mat(SA.Table.Mirrored)/10, ShotID+mirrored/10);

SA.Current_ti = ti;
SA.Current_ShotID = SA.Table.ShotID(si);
SA.Current_si = si;

