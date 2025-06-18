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
        self.ball_line[1], = self.ax.plot([], [], 'y-', label='Ball 1', marker='o', markersize=5)
        self.ball_line[2], = self.ax.plot([], [], 'r-', label='Ball 2', marker='o', markersize=5)

        self.ball_circ = {}
        self.ball_circ[0] = plt.Circle((0.200, 0.220), self.param['ballR'], 
                                     color='w', linewidth=2, fill=True)
        self.ball_circ[1] = plt.Circle((0.100, 0.500), self.param['ballR'], 
                                     color='y', linewidth=2, fill=True)
        self.ball_circ[2] = plt.Circle((0.800, 1.000), self.param['ballR'], 
                                     color='r', linewidth=2, fill=True)        # Initialize hit event circles
        self.hit_events = {}
        for bi in range(3):
            self.hit_events[bi] = []  # List to store circle patches

        
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
        
    def plot(self, ball, hit=None):
        # Update existing plot elements
        self.ball_line[0].set_data(ball[0]['x'], ball[0]['y'])
        self.ball_line[1].set_data(ball[1]['x'], ball[1]['y'])
        self.ball_line[2].set_data(ball[2]['x'], ball[2]['y'])

        self.ball_circ[0].center = (ball[0]['x'][0], ball[0]['y'][0])
        self.ball_circ[1].center = (ball[1]['x'][0], ball[1]['y'][0])
        self.ball_circ[2].center = (ball[2]['x'][0], ball[2]['y'][0])
          # Update hit event circles if hit data is available and valid
        # First, remove existing hit event circles
        for bi in range(3):
            for circle in self.hit_events[bi]:
                circle.remove()
            self.hit_events[bi].clear()
        
        # Only plot hit events if evaluations have been done and events are available
        if hit is not None and self._are_hit_events_valid(hit):
            for bi in range(3):
                if bi in hit and 'XPos' in hit[bi] and 'YPos' in hit[bi]:
                    # Check if there are actual hit events (more than just the starting position)
                    if len(hit[bi]['XPos']) > 1:
                        # Extract hit positions, skipping the first position (starting position)
                        for i in range(1, len(hit[bi]['XPos'])):  # Skip index 0 (starting position)
                            x_pos = hit[bi]['XPos'][i]
                            y_pos = hit[bi]['YPos'][i]
                            
                            # Handle case where positions might be lists or scalars
                            x_positions = x_pos if isinstance(x_pos, list) else [x_pos]
                            y_positions = y_pos if isinstance(y_pos, list) else [y_pos]
                            
                            # Create circles for each hit position
                            for x, y in zip(x_positions, y_positions):
                                circle = plt.Circle((x, y), self.param['ballR'], 
                                                  color='black', linewidth=1, fill=False)
                                self.ax.add_patch(circle)
                                self.hit_events[bi].append(circle)        
        self.canvas.draw_idle()

    def _are_hit_events_valid(self, hit):
        """
        Check if hit events have been properly extracted and are valid for plotting.
        Returns True only if the hit data contains actual events (not just initial state).
        """
        if not hit:
            return False
        
        # Check if any ball has hit events beyond the initial position
        for bi in range(3):
            if (bi in hit and 
                'XPos' in hit[bi] and 
                'YPos' in hit[bi] and 
                'with' in hit[bi]):
                
                # Check if there are hit events beyond the starting position
                if len(hit[bi]['XPos']) > 1:
                    # Check if there are actual hit types recorded (not just initial '-')
                    hit_types = hit[bi]['with']
                    if len(hit_types) > 1 and any(h != '-' for h in hit_types[1:]):
                        return True
        
        return False

    def update(self):
        self.canvas.draw_idle()


    def close(self):
        self.master.destroy()