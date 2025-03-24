function ind = sort_b1b2b3(b1b2b3)
ind = zeros(length(b1b2b3),3);
for i = 1:length(b1b2b3)
   [b123, ind(i,:)] = sort([strfind(b1b2b3{i},'W') strfind(b1b2b3{i},'Y') strfind(b1b2b3{i},'R')]);
end