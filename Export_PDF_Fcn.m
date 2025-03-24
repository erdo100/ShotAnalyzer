function Export_PDF_Fcn(eventdata, data)

global SA param player
[pathname, filename, ext] = fileparts(SA.fullfilename);
exportfolder = 'PNG_export';

exportfolder = fullfile(pathname,exportfolder);
if ~isfolder(exportfolder)
    mkdir(exportfolder);
end

si_Sel = find(cell2mat(SA.Table.Selected) == true);

for si = si_Sel'
    
    SA.Current_si = si;
    
    player.uptodate = 0;
    
    PlayerFunction('plotcurrent',[])
    
    fig = findobj('Tag','Table_figure');
    ax = findobj('Tag','Table');
    frame_h = findobj('Tag','Tableframe');
    
    %set background to none
    set(ax, 'Color','none');
    % set Frame color
    set(frame_h, 'FaceColor', 'none')
    
    filename = fullfile(exportfolder, ...
        [SA.Table.Filename{si},'_',sprintf('%03d',SA.Table.ShotID{si}),'_',num2str(SA.Table.Mirrored{si}),'.png']);
    print(fig,filename,'-dpng')%,'-fillpage')
    disp(['Exported to ',filename])
end

close_table_figure

PlayerFunction('plotcurrent',[])
