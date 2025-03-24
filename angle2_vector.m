function angle = angle2_vector(a,b)

angle = acos(sum(a.*b)/(norm(a)*norm(b)))*180/pi;
