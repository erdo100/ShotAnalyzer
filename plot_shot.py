from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
import numpy as np

class plot_shot:
    def __init__(self, param, master=None):
        self.param = param
        self.master = master or tk.Toplevel()
        self.master.title("Shot Visualization")
        
        # Create figure and canvas
        self.fig = Figure(figsize=(7, 12))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Setup axes and menu
        self._setup_menu()
        self.ax = self.fig.add_subplot(111)
        self._initialize_balls()
        self._setup_axes()
        
        # Close handling
        self.master.protocol("WM_DELETE_WINDOW", self.close)

    def _setup_axes(self):
        
        self.ax.set_xlim(0, self.param['size'][0])
        self.ax.set_ylim(0, self.param['size'][1])
        self.ax.set_xlabel('X-axis (m)')
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xticks(np.linspace(0, self.param['size'][0], 5))
        self.ax.set_yticks(np.linspace(0, self.param['size'][1], 9))
        self.ax.grid(True, linestyle='--', linewidth=0.8, color='gray')
        self.ax.set_facecolor((0.4, 0.4, 1.0))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.tick_params(axis='both', which='both', length=0)

    def _setup_menu(self):
        menubar = tk.Menu(self.master)
        
        # White ball menu
        white_menu = tk.Menu(menubar, tearoff=0)
        for option in ['ball line', 'markers', 'ghost ball', 'start position', 'current position']:
            white_menu.add_command(label=option, 
                                 command=lambda o=option: self._menu_handler('white', o))
        
        # Yellow ball menu
        yellow_menu = tk.Menu(menubar, tearoff=0)
        for option in ['ball line', 'markers', 'ghost ball', 'start position', 'current position']:
            yellow_menu.add_command(label=option, 
                                  command=lambda o=option: self._menu_handler('yellow', o))
        
        # Red ball menu
        red_menu = tk.Menu(menubar, tearoff=0)
        for option in ['ball line', 'markers', 'ghost ball', 'start position', 'current position']:
            red_menu.add_command(label=option, 
                               command=lambda o=option: self._menu_handler('red', o))

        menubar.add_cascade(label="Plot White", menu=white_menu)
        menubar.add_cascade(label="Plot Yellow", menu=yellow_menu)
        menubar.add_cascade(label="Plot Red", menu=red_menu)

        self.master.config(menu=menubar)

    def _menu_handler(self, color, option):
        print(f"Menu selection: {color} - {option}")

    def _initialize_balls(self):
        self.ball_line = {}
        self.ball_line[0], = self.ax.plot([], [], 'w-', label='Ball 0', marker='o', markersize=5)
        self.ball_line[1], = self.ax.plot([], [], 'y-', label='Ball 1')
        self.ball_line[2], = self.ax.plot([], [], 'r-', label='Ball 2')

        self.ball_circ = {}
        self.ball_circ[0] = plt.Circle((0.200, 0.220), self.param['ballR'], 
                                     color='w', linewidth=2, fill=True)
        self.ball_circ[1] = plt.Circle((0.100, 0.500), self.param['ballR'], 
                                     color='y', linewidth=2, fill=True)
        self.ball_circ[2] = plt.Circle((0.800, 1.000), self.param['ballR'], 
                                     color='r', linewidth=2, fill=True)

        
        # plot table rectangle
        self.ax.add_patch(plt.Rectangle((0, 0), self.param['size'][0], self.param['size'][1],
                                      edgecolor='black', facecolor=(0.4, 0.4, 1.0), lw=2))
        
        # plot inner limits of ball center
        self.ax.add_patch(plt.Rectangle((self.param['ballR'], self.param['ballR']),
                                        self.param['size'][0] - self.param['ballR']*2, 
                                        self.param['size'][1] - self.param['ballR']*2, 
                                        edgecolor='black', facecolor=(0.4, 0.4, 1.0), lw=1, linestyle='--'))
        for circ in self.ball_circ.values():
            self.ax.add_patch(circ)
        
    def plot(self, ball):
        # Update existing plot elements
        self.ball_line[0].set_data(ball[0]['x'], ball[0]['y'])
        self.ball_line[1].set_data(ball[1]['x'], ball[1]['y'])
        self.ball_line[2].set_data(ball[2]['x'], ball[2]['y'])

        self.ball_circ[0].center = (ball[0]['x'][0], ball[0]['y'][0])
        self.ball_circ[1].center = (ball[1]['x'][0], ball[1]['y'][0])
        self.ball_circ[2].center = (ball[2]['x'][0], ball[2]['y'][0])
        self.canvas.draw_idle()

    def update(self):
        self.canvas.draw_idle()


    def close(self):
        self.master.destroy()