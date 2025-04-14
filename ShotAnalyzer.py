import numpy as np
from tkinter import filedialog, messagebox
import subprocess
import sys
import tkinter as tk
from tkinter import ttk
import pandas as pd
import pickle
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from read_gamefile import read_gamefile  
from plot_shot import plot_shot
from extract_dataquality_start import extract_dataquality_start
from extract_b1b2b3_start import extract_b1b2b3_start
from extract_events_start import extract_events_start
from extract_shotdata_start import extract_shotdata_start

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

        self.SA = {}

        # create empty dataframe in SA["Table"]
        self.SA["Table"] = pd.DataFrame(columns=[])

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
        file_menu.add_command(label="Load JSON gamefile", command=self.load_jsonfile)
        file_menu.add_command(label="Load Gamefile", command=self.load_gamefile)
        file_menu.add_command(label="Save Gamefile", command=self.save_gamefile)
        file_menu.add_command(label="Export to CSV", command=self.export_csv)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Existing Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        edit_menu.add_command(label="Delete Selected Rows", command=self.delete_selected_rows)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        
        # Existing Analyse menu
        analyze_menu = tk.Menu(menubar, tearoff=0)
        analyze_menu.add_command(label="Quality Check", command=lambda: extract_dataquality_start(self))
        analyze_menu.add_command(label="Identify B1-B2-B3", command=lambda: extract_b1b2b3_start(self))
        analyze_menu.add_command(label="Extract Events", command=lambda: extract_events_start(self))
        analyze_menu.add_command(label="Run All", command=lambda: extract_shotdata_start(self))
        
        menubar.add_cascade(label="Analyze", menu=analyze_menu)

        self.root.config(menu=menubar)

    def load_jsonfile(self):
        filepath = filedialog.askopenfilename(
            title="Select Gamefile",
            filetypes=[("Game files", "*.txt *.csv"), ("All files", "*.*")]
        )
        
        if not filepath:
            return

        try:
            # Call the function directly
            self.SA = read_gamefile(filepath)
            
            # Update and refresh
            self.refresh_table()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load gamefile:\n{str(e)}")

    def load_gamefile(self):
        print("Menu click Load Gamefile")
        # Load SA using pickle from disk using filepicker
        filepath = filedialog.askopenfilename(defaultextension=".sapy", 
                                                filetypes=[("Shot Analyzer files", "*.sapy")])
        if not filepath:
            return
        try:
            # Load the SA structure from a file using pickle
            with open(filepath, 'rb') as f:
                self.SA = pickle.load(f)
            
            # Update and refresh table
            self.refresh_table()
            messagebox.showinfo("Success", "Gamefile loaded successfully.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load gamefile:\n{str(e)}")
        

    def save_gamefile(self):

        #save SA using pickle to disk using filepicker
        filepath = filedialog.asksaveasfilename(defaultextension=".sapy", 
                                                filetypes=[("Shot Analyzer files", "*.sapy")])
        if not filepath:
            return
        try:
            # Save the SA structure to a file using pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self.SA, f)
            messagebox.showinfo("Success", "Gamefile exported successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export gamefile:\n{str(e)}") 

    def export_csv(self):
        print("Menu click Export CSV")
        pass

    def refresh_table(self):
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Get current columns from the DataFrame
        current_columns = self.SA["Table"].columns.tolist()
        
        # Reconfigure the Treeview columns
        self.tree["columns"] = ["Select"] + current_columns
        
        # Clear existing headings and columns
        for col in self.tree["columns"]:
            self.tree.heading(col, text="")
            self.tree.column(col, width=0)
        
        # Set new headings and column configurations
        self.tree.heading("Select", text="Select")
        self.tree.column("Select", width=50, anchor="center")
        
        for col in current_columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor="center")
        
        # Populate with new data
        for idx, row in self.SA["Table"].iterrows():
            values = ['☐'] + list(row)
            self.tree.insert('', 'end', values=values, tags=(idx,))

    def setup_ui(self):
        self.root.title("DataFrame Viewer")
        self.root.geometry("1200x600")
        
        frame = ttk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True)
        
        # Initialize the Treeview with columns based on the initial empty DataFrame
        column_structure = self.SA["Table"].columns.tolist()
        self.tree = ttk.Treeview(
            frame, columns=('Select',) + tuple(column_structure), show='headings'
        )
        
        # Scrollbars
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        vsb.pack(side='right', fill='y')
        hsb.pack(side='bottom', fill='x')
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind('<Button-1>', self.on_click)
        
        # Initial call to refresh_table to set up headings and columns
        self.refresh_table()


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
        #row_data = self.SA["Table"].loc[idx]

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