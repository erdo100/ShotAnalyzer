function [b1b2b3, b1i, b2i, b3i]=str2num_B1B2B3(B1B2B3)

b1b2b3 = [strfind('WYR',B1B2B3(1)) ...
   strfind('WYR',B1B2B3(2)) ...
   strfind('WYR',B1B2B3(3))];
b1i = b1b2b3(1);
b2i = b1b2b3(2);
b3i = b1b2b3(3);
