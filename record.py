#!/usr/bin/env python

"""Record training samples: screenshots of the emulator paired with
your controller input.

Starts mupen64plus for you in a normal window (default SDL input
plugin, so your gamepad works).
"""

import numpy as np
import os
import shutil
import mss
import matplotlib
matplotlib.use('TkAgg')
from datetime import datetime
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as FigCanvas

from PIL import ImageTk, Image

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as tkMessageBox

from emulator import launch_emulator
from utils import Screenshot, XboxController, find_emulator_window

ROM_PATH = 'MarioKart64.z64'

IMAGE_SIZE = (320, 240)
IDLE_SAMPLE_RATE = 1500
SAMPLE_RATE = 200
IMAGE_TYPE = ".png"

class MainWindow():
    """ Main frame of the application
    """

    def __init__(self):
        self.root = tk.Tk()
        self.sct = mss.MSS()

        self.root.title('Data Acquisition')
        self.root.geometry("660x325")
        self.root.resizable(False, False)

        # Init controller
        self.controller = XboxController()

        # Figure out where to capture from
        self.offset_x = None
        self.offset_y = None
        self.update_capture_offset()

         # Create GUI
        self.create_main_panel()

        # Timer
        self.rate = IDLE_SAMPLE_RATE
        self.sample_rate = SAMPLE_RATE
        self.idle_rate = IDLE_SAMPLE_RATE
        self.recording = False
        self.t = 0
        self.pause_timer = False
        self.on_timer()

        self.root.mainloop()

    def create_main_panel(self):
        # Panels
        top_half = tk.Frame(self.root)
        top_half.pack(side=tk.TOP, expand=True, padx=5, pady=5)
        message = tk.Label(self.root, text="(Note: UI updates are disabled while recording)")
        message.pack(side=tk.TOP, padx=5)
        bottom_half = tk.Frame(self.root)
        bottom_half.pack(side=tk.LEFT, padx=5, pady=10)

        # Images
        self.img_panel = tk.Label(top_half, image=ImageTk.PhotoImage("RGB", size=IMAGE_SIZE)) # Placeholder
        self.img_panel.pack(side = tk.LEFT, expand=False, padx=5)

        # Joystick
        self.init_plot()
        self.PlotCanvas = FigCanvas(figure=self.fig, master=top_half)
        self.PlotCanvas.get_tk_widget().pack(side=tk.RIGHT, expand=False, padx=5)

        # Recording
        textframe = tk.Frame(bottom_half, width=332, height=15, padx=5)
        textframe.pack(side=tk.LEFT)
        textframe.pack_propagate(0)
        self.outputDirStrVar = tk.StringVar()
        self.txt_outputDir = tk.Entry(textframe, textvariable=self.outputDirStrVar, width=100)
        self.txt_outputDir.pack(side=tk.LEFT)
        self.outputDirStrVar.set("samples/" + datetime.now().strftime('%Y-%m-%d_%H:%M:%S'))

        self.record_button = ttk.Button(bottom_half, text="Record", command=self.on_btn_record)
        self.record_button.pack(side = tk.LEFT, padx=5)


    def init_plot(self):
        self.plotMem = 50 # how much data to keep on the plot
        self.plotData = [[0] * (5)] * self.plotMem # mem storage for plot

        self.fig = Figure(figsize=(4,3), dpi=80) # 320,240
        self.axes = self.fig.add_subplot(111)
        self.axes.set_ylim(-1.1, 1.1)
        # keep the line objects and just update their data (much faster than re-plotting)
        self.lines = [self.axes.plot([0] * self.plotMem, color)[0]
                      for color in ('r', 'b', 'g', 'k', 'y')]


    def on_timer(self):
        self.poll()

        # stop drawing (and window hunting) if recording to avoid slow downs
        if self.recording == False:
            self.update_capture_offset()
            self.draw()

        if not self.pause_timer:
            self.root.after(self.rate, self.on_timer)


    def poll(self):
        self.img = self.take_screenshot()
        self.controller_data = self.controller.read()
        self.update_plot()

        if self.recording == True:
            self.save_data()
            self.t += 1


    def update_capture_offset(self):
        # Locate the emulator window if possible, otherwise fall back
        # to the static offsets (top left corner)
        window_pos = find_emulator_window()
        if window_pos is None:
            window_pos = (Screenshot.OFFSET_X, Screenshot.OFFSET_Y)

        if window_pos != (self.offset_x, self.offset_y):
            self.offset_x, self.offset_y = window_pos
            print(f'Capturing from {self.offset_x},{self.offset_y}')

    def take_screenshot(self):
        # Get raw pixels from the screen
        sct_img = self.sct.grab({  "top": self.offset_y,
                                  "left": self.offset_x,
                                 "width": Screenshot.SRC_W,
                                "height": Screenshot.SRC_H})

        # Create the Image
        return Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')


    def update_plot(self):
        self.plotData.append(self.controller_data) # adds to the end of the list
        self.plotData.pop(0) # remove the first item in the list, ie the oldest


    def save_data(self):
        image_file = self.outputDir+'/'+'img_'+str(self.t)+IMAGE_TYPE
        self.img.save(image_file)

        # write csv line
        self.outfile.write( image_file + ',' + ','.join(map(str, self.controller_data)) + '\n' )


    def draw(self):
        # Image
        self.img.thumbnail(IMAGE_SIZE, Image.Resampling.LANCZOS) # Resize
        self.img_panel.img = ImageTk.PhotoImage(self.img)
        self.img_panel['image'] = self.img_panel.img

        # Joystick
        x = np.asarray(self.plotData)
        for line, channel in zip(self.lines, x.T):
            line.set_ydata(channel)
        self.PlotCanvas.draw()


    def on_btn_record(self):
        # pause timer
        self.pause_timer = True

        if self.recording:
            self.recording = False
        else:
            self.start_recording()

        if self.recording:
            self.t = 0 # Reset our counter for the new recording
            self.record_button["text"] = "Stop"
            self.rate = self.sample_rate
            # make / open outfile
            self.outfile = open(self.outputDir+'/'+'data.csv', 'a')
        else:
            self.record_button["text"] = "Record"
            self.rate = self.idle_rate
            self.outfile.close()

        # un pause timer
        self.pause_timer = False
        self.on_timer()


    def start_recording(self):
        self.update_capture_offset()
        should_record = True

        # check that a dir has been specified
        if not self.outputDirStrVar.get():
            tkMessageBox.showerror(title='Error', message='Specify the Output Directory', parent=self.root)
            should_record = False

        else: # a directory was specified
            self.outputDir = self.outputDirStrVar.get()

            # check if path exists - i.e. may be saving over data
            if os.path.exists(self.outputDir):

                # overwrite the data, yes/no?
                if tkMessageBox.askyesno(title='Warning!', message='Output Directory Exists - Overwrite Data?', parent=self.root):
                    # delete & re-make the dir:
                    shutil.rmtree(self.outputDir)
                    os.mkdir(self.outputDir)

                # answer was 'no', so do not overwrite the data
                else:
                    should_record = False
                    self.txt_outputDir.focus_set()

            # directory doesn't exist, so make one
            else:
                os.mkdir(self.outputDir)

        self.recording = should_record


if __name__ == '__main__':
    # default input plugin (SDL) so the real controller drives the game
    launch_emulator(ROM_PATH)

    app = MainWindow()
