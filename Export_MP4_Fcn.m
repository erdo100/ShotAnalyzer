function Export_MP4_Fcn(eventdata, data)

global SA param player
[pathname, filename, ext] = fileparts(SA.fullfilename);
exportfolder = 'MP4_export';

%exportfolder = fullfile(pathname,exportfolder);
if ~isfolder(exportfolder)
    mkdir(exportfolder);
end

si_Sel = find(cell2mat(SA.Table.Selected) == true);

for si = si_Sel'
    
    SA.Current_si = si;
    
    player.uptodate = 0;
    
    
    
    file = [SA.Table.Filename{si},'_',sprintf('%03d',SA.Table.ShotID{si}),'_',num2str(SA.Table.Mirrored{si}),'.mp4'];
    player.videofile = [exportfolder,'\',file];
    
    PlayerFunction('record_batch',[])

end

close_table_figure

PlayerFunction('plotcurrent',[])
