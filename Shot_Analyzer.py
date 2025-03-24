import os
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QApplication, QTableWidget, 
                            QTableWidgetItem, QMenuBar, QToolBar, QAction)
from PyQt5.QtGui import QIcon, QPixmap, QColor
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.image as mpimg

class ShotAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Initialize global variables
        self.param = {}
        self.player = {}
        self.SA = {}
        
        self.setup_parameters()
        self.setup_player_settings()
        self.load_icons()
        self.setup_ui()
        
    def setup_parameters(self):
        """Initialize all the analysis parameters"""
        self.param['ver'] = 'Shot Analyzer v0.43i'
        self.param['size'] = np.array([1420, 2840])  # Table size in mm
        self.param['ballR'] = 61.5 / 2  # Ball radius
        
        # Create ball circumference points
        angles = np.linspace(0, 2*np.pi, 36)
        self.param['ballcirc'] = np.array([
            np.sin(angles) * self.param['ballR'],
            np.cos(angles) * self.param['ballR']
        ])
        
        # Other table parameters
        self.param['rdiam'] = 7
        self.param['cushionwidth'] = 50
        self.param['diamdist'] = 97
        self.param['framewidth'] = 147
        self.param['colors'] = 'wyr'
        
        # Detection parameters
        self.param['BallOutOfTableDetectionLimit'] = 30
        self.param['BallCushionHitDetectionRange'] = 50
        self.param['BallProjecttoCushionLimit'] = 10
        self.param['NoDataDistanceDetectionLimit'] = 600
        self.param['MaxTravelDistancePerDt'] = 0.1
        self.param['MaxVelocity'] = 12000
        self.param['timax_appr'] = 5
        self.param['with_title'] = 0  # Controls title plotting
        
    def setup_player_settings(self):
        """Initialize player settings"""
        self.player['framerate'] = 30
        self.player['dt'] = 1 / self.player['framerate']
        self.player['uptodate'] = 0
        
        # Player display settings
        self.player['setting'] = {
            'plot_selected': False,
            'plot_timediagram': False,
            'plot_check_extendline': False,
            'plot_only_blue_table': 'off',
            'lw': np.ones(3) * 7,
            'ball': [
                {
                    'ball': True, 'line': True, 'initialpos': True,
                    'marker': False, 'ghostball': True, 'diamondpos': False
                },
                {
                    'ball': True, 'line': True, 'initialpos': True,
                    'marker': False, 'ghostball': False, 'diamondpos': False
                },
                {
                    'ball': True, 'line': True, 'initialpos': True,
                    'marker': False, 'ghostball': False, 'diamondpos': False
                }
            ]
        }
        
    def load_icons(self):
        """Load all the icon images"""
        icon_path = 'icons'
        if not os.path.isdir(icon_path):
            print("Message from Ersin. Cemal, please copy the ICONS folder "
                  "in the location where the *.exe is executed. Kolay gelsin")
            return
            
        # Load icon images
        self.player['icon'] = {}
        try:
            self.player['icon']['first'] = QIcon(os.path.join(icon_path, 'first_16.gif'))
            self.player['icon']['fastbackward'] = QIcon(os.path.join(icon_path, 'fast_backward_16.gif'))
            self.player['icon']['onebackward'] = QIcon(os.path.join(icon_path, 'back_pause_16.gif'))
            self.player['icon']['play'] = QIcon(os.path.join(icon_path, 'play_16.gif'))
            self.player['icon']['pause'] = QIcon(os.path.join(icon_path, 'pause_24.gif'))
            self.player['icon']['oneforward'] = QIcon(os.path.join(icon_path, 'forward_pause_16.gif'))
            self.player['icon']['fastforward'] = QIcon(os.path.join(icon_path, 'fast_forward_16.gif'))
            self.player['icon']['last'] = QIcon(os.path.join(icon_path, 'last_16.gif'))
            self.player['icon']['record'] = QIcon(os.path.join(icon_path, 'record_16.gif'))
            self.player['icon']['previous'] = QIcon(os.path.join(icon_path, 'previous_16.gif'))
            self.player['icon']['next'] = QIcon(os.path.join(icon_path, 'next_16.gif'))
            
            # Create colored ball icons
            self.player['icon']['WhiteBall'] = self.create_color_icon(QColor(255, 255, 255))
            self.player['icon']['YellowBall'] = self.create_color_icon(QColor(255, 255, 0))
            self.player['icon']['RedBall'] = self.create_color_icon(QColor(255, 0, 0))
            
            # Line drawing icons
            self.player['icon']['drawwhitelines'] = QIcon(os.path.join(icon_path, 'draw_white_lines_16.gif'))
            self.player['icon']['drawyellowlines'] = QIcon(os.path.join(icon_path, 'draw_yellow_lines_16.gif'))
            self.player['icon']['drawredlines'] = QIcon(os.path.join(icon_path, 'draw_red_lines_16.gif'))
            
        except Exception as e:
            print(f"Error loading icons: {e}")
            
    def create_color_icon(self, color):
        """Create a colored ball icon"""
        pixmap = QPixmap(16, 16)
        pixmap.fill(color)
        return QIcon(pixmap)
        
    def setup_ui(self):
        """Set up the user interface"""
        self.setWindowTitle(self.param['ver'])
        self.setWindowIcon(self.player['icon']['play'])
        
        # Set window size and position
        screen = QApplication.primaryScreen()
        screen_size = screen.size()
        width = int(screen_size.width() * 0.38)
        height = int(width / 2 + 177)
        posx = 10
        posy = screen_size.height() - height - 10
        self.setGeometry(posx, posy, width, height)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create table widget
        self.table = QTableWidget()
        self.table.setColumnCount(0)  # Will be set when data is loaded
        self.table.setRowCount(0)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.cellClicked.connect(self.cell_selected)
        self.table.cellChanged.connect(self.cell_edited)
        self.setCentralWidget(self.table)
        
        # Create toolbar with player controls
        toolbar = QToolBar("Player Controls")
        self.addToolBar(toolbar)
        
        # Add player control buttons
        actions = [
            ('first', 'First', 'first'),
            ('fastbackward', 'Fast Backward', 'fastbackward'),
            ('onebackward', 'One Backward', 'onebackward'),
            ('play', 'Play', 'play'),
            ('pause', 'Pause', 'pause'),
            ('oneforward', 'One Forward', 'oneforward'),
            ('fastforward', 'Fast Forward', 'fastforward'),
            ('last', 'Last', 'last'),
            ('record', 'Record', 'record'),
            ('previous', 'Previous', 'previous'),
            ('next', 'Next', 'next')
        ]
        
        for name, text, icon in actions:
            action = QAction(self.player['icon'][icon], text, self)
            action.triggered.connect(lambda _, n=name: self.player_action(n))
            toolbar.addAction(action)
        
        # Initialize empty data structure
        self.SA['fullfilename'] = os.path.join('..', 'Gamedata')
        self.SA['Table'] = {
            'ShotID': [],
            'Mirrored': [],
            'Interpreted': [],
            'B1B2B3': [],
            'ErrorID': [],
            'ErrorText': [],
            'Selected': []
        }
        
    def create_menu_bar(self):
        """Create the menu bar with all menu items"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        file_menu.addAction('Open Game Data', self.read_all_game_data)
        file_menu.addAction('Exit', self.close)
        
        # View menu
        view_menu = menubar.addMenu('View')
        view_menu.addAction('Show Selected', lambda: self.toggle_setting('plot_selected'))
        view_menu.addAction('Show Time Diagram', lambda: self.toggle_setting('plot_timediagram'))
        view_menu.addAction('Show Extend Lines', lambda: self.toggle_setting('plot_check_extendline'))
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        tools_menu.addAction('Analyze Shots', self.analyze_shots)
        
    def toggle_setting(self, setting_name):
        """Toggle a boolean setting"""
        current = self.player['setting'][setting_name]
        self.player['setting'][setting_name] = not current
        self.update_display()
        
    def read_all_game_data(self, *args):
        """Load game data from files"""
        # TODO: Implement actual file loading
        print("Loading game data...")
        
        # For now, create some dummy data
        self.SA['Table']['ShotID'] = [1, 2, 3]
        self.SA['Table']['Mirrored'] = [0, 0, 0]
        self.SA['Table']['Interpreted'] = [0, 0, 0]
        self.SA['Table']['B1B2B3'] = ['WYR', 'WYR', 'WYR']
        self.SA['Table']['ErrorID'] = [None, None, None]
        self.SA['Table']['ErrorText'] = ['', '', '']
        self.SA['Table']['Selected'] = [False, False, False]
        
        self.update_table()
        
    def update_table(self):
        """Update the table widget with current data"""
        self.table.clear()
        
        # Set column headers
        headers = ['ShotID', 'Mirrored', 'Interpreted', 'B1B2B3', 'ErrorID', 'ErrorText', 'Selected']
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        
        # Add data rows
        row_count = len(self.SA['Table']['ShotID'])
        self.table.setRowCount(row_count)
        
        for row in range(row_count):
            for col, header in enumerate(headers):
                value = self.SA['Table'][header][row]
                item = QTableWidgetItem(str(value))
                
                # Make some columns editable
                if header in ['Interpreted', 'B1B2B3', 'Selected']:
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
                else:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    
                self.table.setItem(row, col, item)
                
    def cell_selected(self, row, col):
        """Handle cell selection"""
        # Get the ShotID and Mirrored status from the selected row
        shot_id = int(self.table.item(row, 0).text())
        mirrored = int(self.table.item(row, 1).text())
        
        # Find the corresponding shot in the database
        for i, (db_id, db_mirrored) in enumerate(zip(self.SA['Table']['ShotID'], 
                                                   self.SA['Table']['Mirrored'])):
            if db_id == shot_id and db_mirrored == mirrored:
                self.SA['Current_ti'] = row
                self.SA['Current_ShotID'] = shot_id
                self.SA['Current_si'] = i
                break
                
        self.player['uptodate'] = 0
        self.player_action('plotcurrent')
        
    def cell_edited(self, row, col):
        """Handle cell edits"""
        header = self.table.horizontalHeaderItem(col).text()
        new_value = self.table.item(row, col).text()
        
        # Convert value to appropriate type
        if header in ['ShotID', 'Mirrored', 'Interpreted', 'ErrorID']:
            try:
                new_value = int(new_value)
            except ValueError:
                new_value = 0
        elif header == 'Selected':
            new_value = new_value.lower() in ['true', '1', 'yes']
        
        # Update the database
        si = self.SA['Current_si']
        self.SA['Table'][header][si] = new_value
        
        self.player['uptodate'] = 0
        self.player_action('plotcurrent')
        
    def player_action(self, action):
        """Handle player control actions"""
        if action == 'plotcurrent':
            self.update_display()
        else:
            print(f"Player action: {action}")
            # TODO: Implement other player actions
            
    def update_display(self):
        """Update the visualization of the current shot"""
        # TODO: Implement the actual visualization
        print("Updating display for current shot...")
        
    def closeEvent(self, event):
        """Handle window close event"""
        # TODO: Add any cleanup needed
        event.accept()
        
    def analyze_shots(self):
        """Analyze all shots in the database"""
        print("Analyzing shots...")
        # TODO: Implement shot analysis

if __name__ == '__main__':
    app = QApplication([])
    window = ShotAnalyzer()
    window.show()
    app.exec_()