#!/usr/bin/env python

#import cv2
import numpy as np
import pygame
import os
import shutil
import wx
import matplotlib
matplotlib.use('WXAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigCanvas

class MainWindow(wx.Frame):
    """ Main frame of the application
    """
    title = 'Data Acquisition'           
         
            
    def __init__(self):
        wx.Frame.__init__(self, None, title=self.title, size=(660,330))
        
        # Init controller
        try:
            pygame.init()
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
        except:
            print 'unable to connect to Xbox Controller'
        
        # Init Camera
        #try:
        #    self.cam = cv2.VideoCapture(0)
        #except:
        #    print 'unable to connect to camera'   
        
     	# Create GUI
        self.create_main_panel()
        
        self.recording = False

        # Timer
        self.timer = wx.Timer(self)     
        self.Bind(wx.EVT_TIMER, self.on_timer, self.timer)        
        self.rate = 500
        self.timer.Start(self.rate)           
    
    
    def create_main_panel(self):
        
        # Panels
        self.img_panel = wx.Panel(self)
        self.joy_panel = wx.Panel(self)
        self.record_panel = wx.Panel(self)
        
        # Images
        img = wx.EmptyImage(320,240)
        self.image_widget = wx.StaticBitmap(self.img_panel, wx.ID_ANY, wx.BitmapFromImage(img))
    
        # Joystick
        self.init_plot()
        self.PlotCanvas = FigCanvas(self.joy_panel, wx.ID_ANY, self.fig)
    
        # Recording
        self.btn_record = wx.Button(self.record_panel, wx.ID_ANY, label="Record", pos=(500,20), size=(100,30))
        self.Bind(wx.EVT_BUTTON, self.on_btn_record, self.btn_record)
        self.Bind(wx.EVT_UPDATE_UI, self.on_update_btn_record, self.btn_record)

        self.txt_outputDir = wx.TextCtrl(self.record_panel, wx.ID_ANY, pos=(0,20), size=(440,30))

        # sizers
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.img_panel, 0, wx.ALL, 5)
        sizer.Add(self.joy_panel, 0, wx.ALL, 5)
       
        mainSizer_v = wx.BoxSizer(wx.VERTICAL)
        mainSizer_v.Add(sizer, 0 , wx.ALL, 5)
        mainSizer_v.Add(self.record_panel, 0 , wx.ALL, 5)

        # finalize layout
        self.SetAutoLayout(True)
        self.SetSizer(mainSizer_v)
        self.Layout()
    
    def init_plot(self):
        self.plotMem = 50 # how much data to keep on the plot    
        self.plotData = [[0] * (5)] * self.plotMem # mem storage for plot
        
        self.fig = Figure((4,3))
        self.axes = self.fig.add_subplot(111)
          
    def poll(self):
        
        # Image
        #flag, self.image = self.cam.read()
        
        # test
        #self.image = np.zeros((480,640,3),np.uint8)
    	
    	screen = wx.ScreenDC()
        size = screen.GetSize()
        self.bmp = wx.EmptyBitmap(size[0], size[1])
        mem = wx.MemoryDC(self.bmp)
        mem.Blit(0, 0, size[0], size[1], screen, 0, 0)
        self.bmp = self.bmp.GetSubBitmap(wx.RectPS([0,0],[800,600]))
        
    	# Joystick
        pygame.event.pump()

        self.x = self.joystick.get_axis(0)
    	self.y = self.joystick.get_axis(1)
    	self.a = self.joystick.get_button(0)
    	self.b = self.joystick.get_button(2) # b=1, x=2
        self.rb = self.joystick.get_button(5)    	

        # test
    	#self.x = float(np.random.rand(1))
    	#self.y = float(np.random.rand(1))
    	#self.a = float(np.random.randint(0,2,1))
    	#self.b = float(np.random.randint(0,2,1))
    	
    	self.plotData.append([self.x,self.y,self.a,self.b,self.rb]) #adds to the end of the list
    	self.plotData.pop(0) #remove the first item in the list, ie the oldest
    	
        if self.recording == True:
            self.bmp.SaveFile(self.outputDir+'/'+'img_'+str(self.t)+'.png', wx.BITMAP_TYPE_PNG)
            
            # make / open outfile
            outfile = open(self.outputDir+'/'+'joystick.csv', 'a')
            outfile.write( str(self.t)+','+str(self.x)+','+str(self.y)+','+str(self.a)+','+str(self.b)+','+str(self.rb)+'\n' )
            outfile.close()
            
            self.t += 1
      
    def draw(self):
        
        # test
        #image_r = cv2.resize( self.image, (320,240) )
        #img = wx.EmptyImage(320,240)
        #img.SetData( image_r.tostring() )
        #self.image_widget.SetBitmap( wx.BitmapFromImage(img) )
        
        # stop drawing if recording to avoid slow downs
        if self.recording == False:
        
            img = wx.ImageFromBitmap(self.bmp)
            img = img.Rescale(320,240)
            self.image_widget.SetBitmap( img.ConvertToBitmap() )
     
            # Joystick
            x = np.asarray(self.plotData)
            self.axes.plot(range(0,self.plotMem), x[:,0], 'r')
            self.axes.hold(True)
            self.axes.plot(range(0,self.plotMem), x[:,1], 'b')
            self.axes.plot(range(0,self.plotMem), x[:,2], 'g')
            self.axes.plot(range(0,self.plotMem), x[:,3], 'k')
            self.axes.plot(range(0,self.plotMem), x[:,4], 'y')
            self.axes.hold(False)
            self.PlotCanvas.draw()
 
    def on_timer(self, event):
        self.poll()
        self.draw()
    
    def on_update_btn_record(self, event):
        label = "Stop" if self.recording else "Record"
        self.btn_record.SetLabel(label)

    def on_btn_record(self, event):
        
        # pause timer
        self.timer.Stop()
        
        # switch state
        self.recording = not self.recording
    
        # if recording
        if self.recording:
      
            # check that a dir has been specified
            if self.txt_outputDir.IsEmpty():
            
                msg = wx.MessageDialog(self, 'Specify the Output Directory', 'Error', wx.OK | wx.ICON_ERROR)
                msg.ShowModal() == wx.ID_YES
                msg.Destroy()
                
                self.recording = False
            
            else:
                self.outputDir = self.txt_outputDir.GetString(0,-1)
                self.t = 0

                # check if path exists - ie may be saving over data
                if os.path.exists(self.outputDir):
                   
                    msg = wx.MessageDialog(self, 'Output Directory Exists - Overwrite Data?', 'Yes or No', wx.YES_NO | wx.ICON_QUESTION)
                    result = msg.ShowModal() == wx.ID_YES
                    msg.Destroy()
                   
                    # overwrite the data
                    if result == True:
                        
                        # delete the dir
                        shutil.rmtree(self.outputDir)
                        
                        # re-make dir
                        os.mkdir(self.outputDir)
                       
                    # do not overwrite the data
                    else: # result == False  
                        self.recording = False
                        self.txt_outputDir.SetFocus()       
                
                # no dir so make one
                else:
                    os.mkdir(self.outputDir)
        
        else: # not recording
            if self.t > 0: # just finished recording
                 info = open(self.outputDir+'/'+'info.txt', 'w')   
                 info.write('info.txt\n')
                 info.write(str(0)+'\n')
                 info.write(str(self.t)+'\n')
                 info.close()
        
        # un pause timer
        self.timer.Start(self.rate)
        
        return

    def on_exit(self, event):
        self.cam.release()
        pygame.exit()
        self.Destroy()       
            
if __name__ == '__main__':
    app = wx.App()
    app.frame = MainWindow()
    app.frame.Show()
    app.MainLoop()            
            
            
