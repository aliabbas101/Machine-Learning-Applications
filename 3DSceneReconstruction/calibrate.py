import cv2
import numpy as np
import wx

from gui import BaseLayout

def main():
    capture=cv2.VideoCapture(0)
    if not (capture.isOpened()):
        capture.open()
    
    capture.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH,640)
    capture.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT,480)
    
    app=wx.App()
    layout=CameraCalibration(None, -1, 'Camera Calibration',capture)
    
    




class CameraCalibration(BaseLayout):
    """Class for camera calibration with a proper GUI"""
    def _create_custom_layout(self):
        """ Creates a horizontal layout with a single button"""
        pnl=wx.Panel(self, -1)
        self.button_calibrate= wx.Button(pnl, label='Calibrate Camera')
        self.Bind(wx.EVT_BUTTON, self._on_button_calibrate)
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        hbox.Add(self.button_calibrate)
        pnl.SetSizer()
        
        
        