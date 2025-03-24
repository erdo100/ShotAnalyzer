function SaveFcn(h, event, mode)
global SA


file0 = '';%fullfile(SA.fullfilename);

switch mode
   case 0
      if exist(file0, 'file') ~= 2
         [FileName,PathName,FilterIndex] = uiputfile('*.saf','Save Shot Analyzer File', file0);
         if isnumeric(FileName)
            disp('abort')
            return
         end
         file = [PathName, FileName];
         
      else
         
         file = file0;
      end
      
      SA1 = SA;

   case 1
      % Save all in to new file
      
      [FileName,PathName,FilterIndex] = uiputfile('*.saf','Save Shot Analyzer File', file0);
      if isnumeric(FileName)
         disp('abort')
         return
      end
      
      file = [PathName, FileName];
      SA1 = SA;
      
      %for i = 1:length(SA.Shot)
      %    SA1.Shot(i) = rmfield(SA1.Shot,'Route');
      %end

    case 2
      % save selected in to new file
      [FileName,PathName,FilterIndex] = uiputfile('*.saf','Save Shot Analyzer File', file0);
      if isnumeric(FileName)
         disp('abort')
         return
      end
      
      file = [PathName, FileName];
      
      ind = cell2mat(SA.Table.Selected);

      SA1.Shot = SA.Shot(ind);
      SA1.Table = SA.Table(ind,:);
   
end

disp('Start saving ...')
save_shot(file, SA1);

SA.fullfilename = file;

disp('done with save')

function save_shot(file, SA)
save(file, 'SA','-v7.3');
