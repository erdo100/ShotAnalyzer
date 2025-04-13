import numpy as np
from tkinter import filedialog, messagebox
import subprocess
import sys
import tkinter as tk
from tkinter import ttk
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from read_gamefile import read_gamefile  # Assuming this is your custom module
from extract_shotdata_start import extract_shotdata_start  # Assuming this is your custom module
from extract_events_start import plot_debug
from plot_shot import plot_shot

class DataFrameViewer:
    def __init__(self, root):
        self.root = root
            # Table properties (adjusted for Python syntax)
        self.param = {
            "ver": "Shot Analyzer v0.43i_Python",
            # Assuming [height, width] based on typical usage, but MATLAB code suggested [Y, X] -> [1420, 2840]
            # Let's stick to MATLAB's apparent [height, width] based on index usage (size(1)=y, size(2)=x)
            "size": [1420, 2840],
            "ballR": 61.5 / 2,
            # 'ballcirc' not directly used in translated functions, kept for reference
            "ballcirc": {
                "x": np.sin(np.linspace(0, 2 * np.pi, 36)) * (61.5 / 2),
                "y": np.cos(np.linspace(0, 2 * np.pi, 36)) * (61.5 / 2),
            },
            "rdiam": 7,
            "cushionwidth": 50,
            "diamdist": 97,
            "framewidth": 147,
            "colors": "wyr",
            "BallOutOfTableDetectionLimit": 30, # Used in commented MATLAB code
            "BallCushionHitDetectionRange": 50, # Not used in translated code
            "BallProjecttoCushionLimit": 10,   # Not used in translated code (clipping used instead)
            "NoDataDistanceDetectionLimit": 600, # Used
            "MaxTravelDistancePerDt": 0.1,      # Not used in translated code
            "MaxVelocity": 12000,               # Used
            "timax_appr": 5,                    # Not used in translated code
        }
        self.column_structure = ['Selected', 'ShotID', 'Mirrored', 'Filename', 
                            'GameType', 'Interpreted', 'Player', 'ErrorID', 'ErrorText', 'Set', 
                            'CurrentInning', 'CurrentSeries', 'CurrentTotalPoints', 'Point']
        self.df = pd.DataFrame(columns=self.column_structure)
        self.SA = None
        self.tree = None
        self.setup_ui()
        self.setup_menu()
        
        # Create plot window and visualization
        self.plot_window = tk.Toplevel(root)
        self.ps = plot_shot(self.param, master=self.plot_window)
        
        # Event bindings
        self.tree.bind('<<TreeviewSelect>>', self.on_tree_select)
        self.tree.bind('<Up>', self.handle_keypress)
        self.tree.bind('<Down>', self.handle_keypress)
        self.current_selection = None

    def setup_menu(self):
        menubar = tk.Menu(self.root)
        
        # New File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Gamefile", command=self.load_gamefile)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Existing Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Delete Selected Rows", command=self.delete_selected_rows)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        
        self.root.config(menu=menubar)

    def load_gamefile(self):
        filepath = filedialog.askopenfilename(
            title="Select Gamefile",
            filetypes=[("Game files", "*.txt *.csv"), ("All files", "*.*")]
        )
        
        if not filepath:
            return

        try:
            # Call the function directly
            self.SA = read_gamefile(filepath)
            #extract_shotdata_start(filepath)  # Store the entire SA structure
            
            # Extract the table from SA
            new_df = self.SA["Table"]
            
            # Validate columns
            if list(new_df.columns) != self.column_structure:
                raise ValueError("Loaded data columns don't match expected structure")
            
            # Update and refresh
            self.df = new_df
            self.refresh_table()
            
        except KeyError:
            messagebox.showerror("Error", "SA structure missing 'Table' key")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load gamefile:\n{str(e)}")
    def refresh_table(self):
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        # Populate with new data
        for idx, row in self.df.iterrows():
            values = ['☐'] + list(row)
            self.tree.insert('', 'end', values=values, tags=(idx,))

    def setup_ui(self):
        self.root.title("DataFrame Viewer")
        self.root.geometry("1200x600")
        
        frame = ttk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True)
        
        self.tree = ttk.Treeview(
            frame, columns=('Select',) + tuple(self.column_structure), show='headings'
        )
        
        # Configure headings
        self.tree.heading('Select', text='Select')
        for col in self.column_structure:
            self.tree.heading(col, text=col)
            
        # Configure columns
        self.tree.column('Select', width=50, anchor='center')
        for col in self.column_structure:
            self.tree.column(col, width=100, anchor='center')
            
        # Scrollbars
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        vsb.pack(side='right', fill='y')
        hsb.pack(side='bottom', fill='x')
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind('<Button-1>', self.on_click)
        self.refresh_table()

    def setup_menu(self):
        menubar = tk.Menu(self.root)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Gamefile", command=self.load_gamefile)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Delete Selected Rows", command=self.delete_selected_rows)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        
        self.root.config(menu=menubar)  # Critical line to show the menu

    def on_tree_select(self, event):
        selected_item = self.tree.selection()
        if selected_item:
            self.plot_row(selected_item[0])

    def handle_keypress(self, event):
        if event.keysym in ('Up', 'Down'):
            self.root.after(10, self.update_plot_after_move)

    def update_plot_after_move(self):
        selected_item = self.tree.selection()
        if selected_item:
            self.plot_row(selected_item[0])
            
    def on_click(self, event):
        region = self.tree.identify("region", event.x, event.y)
        if region == "heading":
            return  # Ignore clicks on headers

        column = self.tree.identify_column(event.x)
        item = self.tree.identify_row(event.y)

        if item and column == '#1':  # Checkbox column
            self.toggle_checkbox(item)
        elif item:  # Other columns
            self.plot_row(item)

    def toggle_checkbox(self, item):
        current_values = list(self.tree.item(item, 'values'))
        current_values[0] = '☑' if current_values[0] == '☐' else '☐'
        self.tree.item(item, values=current_values)

    def plot_row(self, item):
        tags = self.tree.item(item, 'tags')
        idx = int(tags[0])
        row_data = self.df.loc[idx]

        # Get and plot current shot data
        current_shot = self.SA["Shot"][idx]["Route"]
        self.ps.plot(current_shot)  # Update existing plot
        self.ps.update()  # Trigger canvas update


    def delete_selected_rows(self):
        selected_items = []
        for item in self.tree.get_children():
            if self.tree.item(item, 'values')[0] == '☑':
                selected_items.append(item)

        # Remove from Treeview and DataFrame
        deleted_indices = []
        for item in selected_items:
            tags = self.tree.item(item, 'tags')
            deleted_indices.append(int(tags[0]))
            self.tree.delete(item)

        # Update both DataFrame and SA structure
        self.df = self.df.drop(index=deleted_indices)
        self.SA['Table'] = self.df  # Update the table in SA
        print(f"Deleted {len(selected_items)} rows. Remaining: {len(self.df)}")



if __name__ == "__main__":
    root = tk.Tk()
    app = DataFrameViewer(root)  # Start with empty dataframe
    root.mainloop()