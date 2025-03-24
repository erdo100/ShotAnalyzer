function Export_BBS_Fcn(eventdata, data)

global SA param player


% Read original BBS file
% Find corrosponding line
% write new file


% Read Original file

% request the files to load
[filename, pathname] = uigetfile('*.bbs', ...
   'Pick file to be extracted', fileparts(SA.fullfilename), 'MultiSelect', 'on')

% Check inputs
if iscell(filename)
   nfiles = size(filename,2);
   files = filename;
elseif isnumeric(filename)
   disp('abort')
   return
else
   nfiles = 1;
   files{1} = filename;
end

for fi = 1:length(files)
    BBStxt = readlines(fullfile(pathname, filename{fi}));
    
    % Now search the content for selected lines
    
    TF=[];
    TF(1) = 1;
    i=1;
    for si = 1:length(SA.Table.Selected)
        if SA.Table.Selected{si} & ~isempty(find(contains(BBStxt, SA.Table.Filename{si})))
            i = i+1;
            TF(i) = find(contains(BBStxt, SA.Table.Filename{si}));
        end
    end
    
    copyfile(fullfile(pathname, filename{fi}), [fullfile(pathname, filename{fi}),'_orig'])
    
    fid = fopen(fullfile(pathname, filename{fi}),'w');
    fprintf(fid, "%s\n", BBStxt{TF})
    fclose(fid)
    
end