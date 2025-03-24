function close_table_figure

global param

fig = findobj('Tag','Table_figure');
param.TablePosition = fig.Position;
close(fig)