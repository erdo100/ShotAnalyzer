function Extract2_Events_start(~,~)

% Extract all ball-ball hit events and ball-Cushion hit events

global SA 

for si = 1:length(SA.Shot)
    SA.Table.Interpreted{si} = 0;
end

        
Extract_Events_start