function SA = create_varname(SA, hit, si)
global param

varparts = fieldnames(hit);

[b1b2b3, b1i, b2i, b3i] = str2num_B1B2B3(SA.Table.B1B2B3{si});

% write BX_with
for balli = 1:3
    varname = ['B',num2str(balli),'_with'];
    % First check is column already available, then update visibility
    ind = find(strcmp(SA.Table.Properties.VariableNames, varname),1);
    if isempty(ind)
            SA.ColumnsVisible{end+1} = varname;
    end
    
    with_new = replace_colors_b1b2b3(hit(b1b2b3(balli)).with, b1b2b3);
    SA.Table.(varname){si} = with_new;
end

SA.Table.('Point0'){si} = hit(b1i).Point;

SA.Table.('Kiss'){si} = hit(b1i).Kiss;

SA.Table.('Fuchs'){si} = hit(b1i).Fuchs;

SA.Table.('PointDist'){si} = hit(b1i).PointDist;

SA.Table.('KissDistB1'){si} = hit(b1i).KissDistB1;

SA.Table.('AimOffsetSS'){si} = hit(b1i).AimOffsetSS;
SA.Table.('AimOffsetLL'){si} = hit(b1i).AimOffsetLL;

SA.Table.('B1B2OffsetSS'){si} = hit(b1i).B1B2OffsetSS;
SA.Table.('B1B2OffsetLL'){si} = hit(b1i).B1B2OffsetLL;
SA.Table.('B1B3OffsetSS'){si} = hit(b1i).B1B3OffsetSS;
SA.Table.('B1B3OffsetLL'){si} = hit(b1i).B1B3OffsetLL;

delname = {'Fraction','DefAngle','CutAngle','CInAngle','COutAngle'};

himax = [8 4 0];

for bi = 1:3
    % write other results
    for hi = 1 : himax(bi) %min([length(hit(b1b2b3(bi)).t) 8])
        for vi = 21:length(varparts)
            varname = ['B',num2str(bi),'_',num2str(hi),'_',varparts{vi}];
            
            if hi == 1 & find(strcmp(varparts{vi}, delname),1)
            else
                if length(hit(b1b2b3(bi)).(varparts{vi})) >= hi
                    % Now check whether we have cushion position, which we want
                    % to store as Diamonds values
                    if contains(varparts{vi},'Pos')
                        scale = param.size(2)/8;
                    else
                        scale = 1;
                    end
                    
                    % Now add the column
                    SA.Table.(varname){si} = hit(b1b2b3(bi)).(varparts{vi})(hi)/scale;
                end
            end
        end
    end
end

