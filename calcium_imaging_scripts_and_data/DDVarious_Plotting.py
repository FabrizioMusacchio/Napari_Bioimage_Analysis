#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scripts for Plotting
"""

''' Importing scripts '''

import matplotlib.pyplot as plt
import matplotlib as mpl
#from scipy.optimize import curve_fit
#import scipy.signal as spsignal
#import numpy as np
import math
import os

class Folder:   
    def pltC(filesdir):
        ''' Imports Current Clamp Traces from Igor Binary Files and ... '''
        ''' ... Plots all in One Window '''
        import DDImport as FI
        
        Waves,TimeVec,SampFreq,RecTimes = FI.Folder.ImpC(filesdir)
        NumPlots = len(Waves)

        # Plotting:
        NumSubplt = math.ceil(math.sqrt(NumPlots))
        fig, ax = plt.subplots(nrows=NumSubplt,ncols=NumSubplt)
        
        count=0
        for W in Waves:
            plt.subplot(NumSubplt,NumSubplt,count+1)
            plt.plot(TimeVec[count],Waves[count])
            count +=1
    
        return plt.show(fig)

class ExtractPlotting:
    def __init__(self,Analysis):
        self.All = Analysis
        
        if hasattr(self.All, 'Figure'):
            self.Figure = self.All.Figure
        if hasattr(self.All, 'ax'):
            self.Ax = self.All.ax
        if hasattr(self.All, 'PlotWave'):
            self.Waves = self.All.PlotWave
        if hasattr(self.All, 'Annot'):
            self.Annot = self.All.Annot
        if hasattr(self.All, 'AnnotWave'):
            self.Annot = self.All.AnnotWave
        if hasattr(self.All, 'legend'):
            self.Legend = self.All.legend
        if hasattr(self.All, 'PlotAP'):
            self.WavesAP = self.All.PlotAP
        if hasattr(self.All, 'AnnotAP'):
            self.AnnotAP = self.All.AnnotAP
        if hasattr(self.All, 'PlotAHP'):
            self.WavesAHP = self.All.PlotAHP
        if hasattr(self.All, 'AnnotAHP'):
            self.AnnotAHP = self.All.AnnotAHP
        if hasattr(self.All, 'PlotAmp'):
            self.WavesAmp = self.All.PlotAmp 
        if hasattr(self.All, 'AnnotAmp'):
            self.AnnotAmp = self.All.AnnotAmp            
        if hasattr(self.All, 'PlotFreq'):
            self.WavesFreq = self.All.PlotFreq
        if hasattr(self.All, 'AnnotFreq'):
            self.AnnotFreq = self.All.AnnotFreq
            

def save(path, ext='png', close=True, verbose=True):
    """Save a figure from pyplot.
    Parameters
    ----------
    path : string
        The path (and filename, without the extension) to save the
        figure to.
    ext : string (default='png')
        The file extension. This must be supported by the active
        matplotlib backend (see matplotlib.backends module).  Most
        backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
    close : boolean (default=True)
        Whether to close the figure after saving.  If you want to save
        the figure multiple times (e.g., to multiple formats), you
        should NOT close it in between saves or you will have to
        re-plot it.
    verbose : boolean (default=True)
        Whether to print information about when and where the image
        has been saved.
    """
    mpl.rcParams['agg.path.chunksize'] = 10000
    # Extract the directory and filename from the given path
    directory = os.path.split(path)[0]
    filename = "%s.%s" % (os.path.split(path)[1], ext)
    if directory == '':
        directory = '.'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # The final path to save to
    savepath = os.path.join(directory, filename)

    if verbose:
        print("Saving figure to '%s'..." % savepath),

    # Actually save the figure
    plt.savefig(savepath,dpi=600)
    
    # Close it
    if close:
        plt.close()

    if verbose:
        print("Done")
        
        
def myHeatMapAppearance():
    from matplotlib.colors import LinearSegmentedColormap
    cdict = {'red': ((0., 1, 1),
             (0.05, 1, 1),
             (0.11, 0, 0),
             (0.66, 1, 1),
             (0.89, 1, 1),
             (1, 0.5, 0.5)),
     'green': ((0., 1, 1),
               (0.05, 1, 1),
               (0.11, 0, 0),
               (0.375, 1, 1),
               (0.64, 1, 1),
               (0.91, 0, 0),
               (1, 0, 0)),
     'blue': ((0., 1, 1),
              (0.05, 1, 1),
              (0.11, 1, 1),
              (0.34, 1, 1),
              (0.65, 0, 0),
              (1, 0, 0))}
    my_cmap = LinearSegmentedColormap('my_colormap',cdict,256)
    return my_cmap
    

####Example for Rotating 3D Plot:
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#import numpy as np
#import matplotlib.animation as animation
#raw = np.random.rand(100,3)
#fig = plt.figure()
#
#ax = fig.add_subplot(111, projection='3d')
#x = raw[:, 0]
#y = raw[:, 1]
#z = raw[:, 2]
#
#ax.scatter(x, y, -z, zdir='z', c='black', depthshade=False, s=2, marker=',')
#
#def rotate(angle):
#    ax.view_init(azim=angle)
#
#print("Making animation")
#rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 2), interval=100)
#rot_animation.save('rotation.gif', dpi=80, writer='imagemagick')
        