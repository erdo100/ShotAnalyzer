function Extract_Events_start(~,~)

% Extract all ball-ball hit events and ball-Cushion hit events

global SA player

if isempty(find(strcmp(SA.Table.Properties.VariableNames,'B1B2B3'),1))
    disp(['B1B2B3 is not identified yet (',mfilename,')'])
    return
end

err = 0;
err_shots = [];

shotlength = length(SA.Shot);
for si = 1:shotlength
    if SA.Table.Interpreted{si} == 0 
        disp(['Shot ', num2str(si),'/',num2str(shotlength)])
        if length(SA.Table.B1B2B3{si}) == 3
            % when there is cushion contact in next time step
            %       table_fig = findobj('Tag','Table_figure');
            %       delete(table_fig)
            %
            %       SAD.selectedrows = si;
            %       htable_shotlist = findobj('Tag', 'uitable_shotlist');
            %       set(htable_shotlist, 'Userdata', SAD);
            %       plot_selection
            
            try
                [b1b2b3, b1i, b2i, b3i] = str2num_B1B2B3(SA.Table.B1B2B3{si});
                
                % extract all events
                [hit, ~] = Extract_Events(si);
                
                % collect the events
                hit = eval_hit_events(hit,si,b1b2b3);

                % This is now done in create_varname.m
%                 % replace b1b2b3 codes
%                 hit = replace_hitcode_b1b2b3(hit, b1b2b3, b1i, b2i, b3i);

                % evaluate hit accuracy and kiss control
                hit = eval_Point_and_Kiss_Control(si, hit);
                
                % create the SA.Table
                SA = create_varname(SA, hit, si) ;
                
                % copy the hit data to SA
                SA.Shot(si).hit = hit;
                
                % Set intrepeted flag
                SA.Table.Interpreted{si} = 1;
                
                % reset flags and delete Route data
                SA.Table.ErrorID{si} = [];
                SA.Table.ErrorText{si} = [];
                SA.Table.Selected{si} = false;
                SA.Shot(si).Route = [];
                
                
            catch
                SA.Table.ErrorID{si} = 100;
                SA.Table.ErrorText{si} = 'Check diagram, correct or delete.';
                SA.Table.Selected{si} = true;
                
                disp('Some error occured, probably ball routes are not continous. Check diagram, correct or delete')
                err = err + 1;
                err_shots(err) = si;
            end
        else
            disp(['B1B2B3 has not 3 letters, skipping Shot ',num2str(si)])
        end
    end
    
end

update_ShotList

disp('These Shots are not interpreted:')
for i=1:err
    disp(num2str(err_shots(i)))
end
% update GUI
update_ShotList
player.uptodate = 0;
PlayerFunction('plotcurrent',[])

disp(['done (',mfilename,')'])
