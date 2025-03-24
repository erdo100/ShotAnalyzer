function angle = angle_vector(a,b)

if norm(a) > 0 & norm(b) > 0
   angle = acos(sum(a.*b)/(norm(a)*norm(b)))*180/pi;
elseif norm(a) > 0 | norm(b) > 0
   angle = -1;
else
   angle = -2;
end