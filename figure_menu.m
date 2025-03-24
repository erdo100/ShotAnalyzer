function figure_menu(hfigure)

% Files Menu
mfile = uimenu(hfigure,'Text','File');
% Create child menu items for the menu 'Selection
uimenu(mfile,'Text','Load new file', 'MenuSelectedFcn', {@Read_All_GameData, 0});
uimenu(mfile,'Text','Load & Append file', 'MenuSelectedFcn', {@Read_All_GameData, 1});
uimenu(mfile,'Text','Save', 'MenuSelectedFcn', {@SaveFcn, 0});
uimenu(mfile,'Text','Save all to new file', 'MenuSelectedFcn', {@SaveFcn, 1});
uimenu(mfile,'Text','Save selected to new file', 'MenuSelectedFcn', {@SaveFcn, 2});
uimenu(mfile,'Text','Export selected shots to PNG', 'MenuSelectedFcn', @Export_PDF_Fcn);
uimenu(mfile,'Text','Export selected shots to MP4', 'MenuSelectedFcn', @Export_MP4_Fcn);
uimenu(mfile,'Text','Export selected shots to BBS', 'MenuSelectedFcn', @Export_BBS_Fcn);
% Create child menu items for the menu 'Selection
uimenu(mfile,'Separator','on', 'Text','Export full shots table to XLS', 'MenuSelectedFcn', @XLS_ExportALL_Fcn);
uimenu(mfile,'Text','Export visible columns to XLS', 'MenuSelectedFcn', @XLS_Export_Fcn);
uimenu(mfile,'Text','Import all shots table from XLS', 'MenuSelectedFcn', @XLS_Import_Fcn);

% Interpreter Menu
mextract = uimenu(hfigure,'Text','Interpreter');
% Create child menu items for the menu 'Selection
uimenu(mextract,'Text','Run All Interpreters','MenuSelectedFcn',@Extract_AllAtOnce);
uimenu(mextract,'Text','Analyze data quality','MenuSelectedFcn',@Extract_DataQuality_start);
uimenu(mextract,'Text','Add mirrored positions','MenuSelectedFcn',@AddMirroredPositions);
uimenu(mextract,'Text','Identify B1B2B3','MenuSelectedFcn',@Extract_b1b2b3_start);
uimenu(mextract,'Text','Identify B1B2B3 Position','MenuSelectedFcn',@Extract_b1b2b3_position);
uimenu(mextract,'Text','Identify Events','MenuSelectedFcn',@Extract_Events_start);
uimenu(mextract,'Text','ReIdentify Events','MenuSelectedFcn',@Extract2_Events_start);
uimenu(mextract,'Separator','on','Text','Plot analytics','Checked','off',...
   'Tag', 'PlotAnalytics', 'MenuSelectedFcn',@menu_change_function);

% Columns Selection Menu
mcol = uimenu(hfigure,'Text','Columns');
uimenu(mcol,'Text','Add blank column','MenuSelectedFcn',@user_defined_column);
uimenu(mcol,'Text','Hide all columns','MenuSelectedFcn',@shotlist_menu_function);
uimenu(mcol,'Text','Show all columns','MenuSelectedFcn',@shotlist_menu_function);
uimenu(mcol,'Text','Choose columns','MenuSelectedFcn',@shotlist_menu_function);
% uimenu(mcol,'Text','Delete columns','MenuSelectedFcn',@shotlist_menu_function);
uimenu(mcol,'Text','Import column names to display from XLS','MenuSelectedFcn',@shotlist_menu_function);

% Selection Menu
msel = uimenu(hfigure,'Text','Shots Selection');
% Create child menu items for the menu 'Selection
uimenu(msel,'Text','Select marked shots','MenuSelectedFcn',@selection_menu_function);
uimenu(msel,'Text','Unselect marked shots','MenuSelectedFcn',@selection_menu_function);
uimenu(msel,'Text','Invert Selection','MenuSelectedFcn',@selection_menu_function);
uimenu(msel,'Text','Select all shots','MenuSelectedFcn',@selection_menu_function);
uimenu(msel,'Text','Unselect all shots','MenuSelectedFcn',@selection_menu_function);
uimenu(msel,'Text','Hide selected shots','MenuSelectedFcn',@selection_menu_function);
uimenu(msel,'Text','Delete selected shots','MenuSelectedFcn',@selection_menu_function);



function menu_change_function(h, event)

if ~strcmp(class(h),'double')
   if strcmp(h.Checked,'off')
      h.Checked = 'on';
   else
      h.Checked = 'off';
   end
end

