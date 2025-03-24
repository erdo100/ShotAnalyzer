function with_new = replace_colors_b1b2b3(with, b1b2b3)


with_new = strrep(strrep(strrep(with,'W','V'),'Y','Z'),'R','T');

if b1b2b3(1) == 1 % B1 is white ==> A is B1
    with_new = strrep(with_new,'V', 'W');
end
if b1b2b3(2) == 1 % B2 is white ==> A is B2
    with_new = strrep(with_new,'V', 'Y');
end
if b1b2b3(3) == 1 % B3 is white ==> A is B3
    with_new = strrep(with_new,'V', 'R');
end

if b1b2b3(1) == 2 % B1 is yellow ==> B is B1
    with_new = strrep(with_new,'Z', 'W');
end
if b1b2b3(2) == 2
    with_new = strrep(with_new,'Z', 'Y');
end
if b1b2b3(3) == 2
    with_new = strrep(with_new,'Z', 'R');
end

if b1b2b3(1) == 3
    with_new = strrep(with_new,'T', 'W');
end
if b1b2b3(2) == 3
    with_new = strrep(with_new,'T', 'Y');
end
if b1b2b3(3) == 3
    with_new = strrep(with_new,'T', 'R');
end

