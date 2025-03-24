function SA = append_new_shot_data(current, new)
%% Append new database to current
% All columns must be available

% If columns are not available in current data, then Add column with empty data
% If columns are not available in new data, then add column with empty data

% then append both data sets

% Clear double available Shot IDs but keep available calculated data of rows
% What to do if values are different? Ak the user!!

% use enable of first database
% for new columns, use enable = true


% When old SA is empty, then just give the new
if isempty(current)
    SA = new;
    
    %% Adjust Visibility for NEW data
    % Check availability of shotID visibility for new
    if isfield(new,'ShotIDsVisible')
        SA.ShotIDsVisible = new.ShotIDsVisible;
    else
        SA.ShotIDsVisible = num2cell(cell2mat(new.Table.ShotID) + cell2mat(new.Table.Mirrored)/10);
    end
    
    % Check availability of Columns visibility for new
    if isfield(new,'ColumnsVisible')
        SA.ColumnsVisible = new.ColumnsVisible;
    else
        SA.ColumnsVisible = SA.Table.Properties.VariableNames;
    end
    
    
else
    
    % does new has field ColumnVisible?
    if ~isfield(new,'ColumnsVisible')
        new.ColumnsVisible = [];
    end
    % does new has field ShotIDsVisible?
    if ~isfield(new,'ShotIDsVisible')
        new.ShotIDsVisible = new.Table.ShotID;
    end

    
    [cur_varlen, cur_shotlen] = size(current.Table);
    [new_varlen, new_shotlen] = size(new.Table);
    
    [names, ci, ni] = unique([current.Table.Properties.VariableNames new.Table.Properties.VariableNames],'stable');
    
    % In names we have now the new columns.
    % The first <cur_varlen> columns are from current shot
    for ni = 1:length(names)
        if isempty(find(strcmp(names{ni}, new.Table.Properties.VariableNames),1))
            % name is not avaiable in new
            % Create variable in new
            new.Table.(names{ni}) = cell(new_varlen,1);
        elseif isempty(find(strcmp(names{ni}, current.Table.Properties.VariableNames),1))
            % name is not available in current
            current.Table.(names{ni}) = cell(cur_varlen,1);
        end
    end
    % Now both table have same variables. So we can append new data to current
    
    % But before appending them, we have to delete doubled ShotIds
    % we keep the current data and delete them from new
    [doubleID, doubleIDicur, doubleIDinew] = intersect(cell2mat(current.Table.ShotID), cell2mat(new.Table.ShotID));
    
    if ~isempty(doubleID)
        new.Table(doubleIDinew,:) =[];
        new.Shot(doubleIDinew) = [];
        
        % Also clear the visibility for the doubled shots
        [~,~,ind] = intersect(doubleID,cell2mat(new.ShotIDsVisible));
        if ~isempty(ind)
            new.ShotIDsVisible(ind) = [];
        end
        
    end
    
    %% Assign the outputs
    SA.Table = [current.Table; new.Table];
    SA.Shot = [current.Shot new.Shot];
    SA.ColumnsVisible = [current.ColumnsVisible new.ColumnsVisible];
    SA.ShotIDsVisible = [current.ShotIDsVisible; new.ShotIDsVisible];
    
end

