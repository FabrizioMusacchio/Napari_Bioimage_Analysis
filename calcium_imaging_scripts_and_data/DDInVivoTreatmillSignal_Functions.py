#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 14:21:46 2019

@author: DennisDa
"""

''' Import Modules: '''
import pandas as pd
import numpy as np
import os.path
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
import DDVarious_Plotting
from tqdm import tqdm

from nptdms import TdmsFile
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import hilbert, butter, filtfilt, find_peaks
from Kalman_Filt_v import Kalman_Filt_v



''' Import Function: '''

def Alt_ImportTreatmileFiles(FileNameCounter,FileNameAnalog):
    treadmill_raw = pd.DataFrame()
    #a) Import PositionObject:
    Position = TdmsFile(FileNameCounter)
    treadmill_raw['time'] = Position.object('counter_log_task', 'Dev4/ctr1').time_track()
    treadmill_raw['position'] = Position.object('counter_log_task', 'Dev4/ctr1').data
    
    #b) AnalogFile:
    AnalogFile = TdmsFile(FileNameAnalog)
    
    # Cut PositionRaw by length of Analog:
    #if len(treadmill_raw.index) > len(AnalogFile.object('analog_log_task', 'Dev1/ai0').data):
        #RowsToMuch = len(treadmill_raw.index)-len(AnalogFile.object('analog_log_task', 'Dev1/ai0').data)
        #treadmill_raw = treadmill_raw.iloc[:-RowsToMuch]
    
    treadmill_raw['licking'] = AnalogFile.object('analog_log_task', 'Dev4/ai0').data
    treadmill_raw['resotrigger'] = AnalogFile.object('analog_log_task', 'Dev4/ai1').data
    treadmill_raw['camtrigger'] = AnalogFile.object('analog_log_task', 'Dev4/ai2').data
    treadmill_raw['optotrigger'] = AnalogFile.object('analog_log_task', 'Dev4/ai3').data
    treadmill_raw['laptrigger'] = AnalogFile.object('analog_log_task', 'Dev4/ai4').data
    treadmill_raw['pump'] = AnalogFile.object('analog_log_task', 'Dev4/ai5').data
    treadmill_raw['brakepuff'] = AnalogFile.object('analog_log_task', 'Dev4/ai6').data
    
    # Cut treatmillraw to index position
    #IndexPosi = 7000000
    #if len(treadmill_raw.index)>IndexPosi:
        #treadmill_raw = treadmill_raw.iloc[:IndexPosi]
    
    return treadmill_raw

''' Whole Process of Belt Information with Import and Export:'''
def WholePrimaryBeltAnalysis(ImportPath,ExportPath,Conditions):
    
    ListofStrings = ImportPath.split("/")
    CellName = ListofStrings[-1]
    print(CellName)
    
    # Import:
    os.chdir(ImportPath)
    PositionFile = [f for f in os.listdir(os.getcwd()) if f.endswith('counter_log.tdms')]
    AnalogFile = [f for f in os.listdir(os.getcwd()) if f.endswith('analog_log.tdms')]
    TreatmillRaw = Alt_ImportTreatmileFiles(PositionFile[0],AnalogFile[0])  
    # ExportPath:
    ExportListOfStrings = ExportPath.split("/")
    ExportResultFolder = ExportListOfStrings[-2]
    ExportPreFolderList = ExportListOfStrings[0:-2]
    ExportPreFolder = '/'.join(ExportPreFolderList)
    os.chdir(ExportPreFolder)
    if not os.path.isdir(ExportResultFolder):
        os.makedirs(ExportResultFolder)        
    os.chdir(ExportResultFolder)
    if not os.path.isdir(CellName):
        os.makedirs(CellName)        
    os.chdir(CellName)
    # Calculations:
    AllBeldInfos = AllBeltInformation(TreatmillRaw,Conditions,CellName,1)
    Downsampling = DownsamplingToColum(AllBeldInfos,'ResoTrigger',DeleteZeroTrigger=1,AdaptedBinary=1,SavingName=CellName)
    print('Done: ', CellName)
    return Downsampling 


''' Get All BeltInformation: '''
def AllBeltInformation(TreatmillRaw,ConditionDic,SavingName=None,Plotting=None):
    ## ConditionDic:
    NumFrames = ConditionDic['NumFrames']
    BeltMin = ConditionDic['Beltmin']
    BeltMax = ConditionDic['BeltMax']
    
    # Calculations:
    ## SampFrequency:
    SampFreq= 1/TreatmillRaw.time[1]-TreatmillRaw.time[0]
    ## Correct Position for NI Jumps:
    TreatmillRaw.position = NIposition_processing(TreatmillRaw.position)
    ## Correction for Position Jumps:
    TreatmillManuallyCorrected = CorrectForBigPositionJumps(TreatmillRaw)
    ## Position and Lap Calculations:
    PositionRaw, Laps = FindLaps(TreatmillManuallyCorrected.position.copy(),TreatmillManuallyCorrected.laptrigger)#,'print')
    ## BeltLength Correction
    FullCorrectedPosi = ScalingToBelt(PositionRaw,BeltMin,BeltMax)
    
    
    # Velocity Calculations: 
    # Whole Distance Run:
    NumLaps = list(np.unique(Laps.Lap))
    DistancePerlap = [None]*len(NumLaps)
    i = 0
    while i < len(NumLaps):
        Interim = FullCorrectedPosi.loc[Laps.Lap==NumLaps[i]]
        DistancePerlap[i] = Interim.Position.max()-Interim.Position.min()
        i += 1
    WholeDistance = sum(DistancePerlap)
    ScalerVelo = MinMaxScaler(feature_range=(0, WholeDistance))
    ScaledPositionVelo = ScalerVelo.fit_transform(TreatmillRaw.position.to_frame()).flatten()  
#    Velocity = TreatmillRaw[['position']].copy() ## for Testing
#    Velocity.rename(columns={'position':'Velocity'}, inplace=True) ## for Testing
    _, Velocity = VelocityCalculationKalMan(ScaledPositionVelo,SampFreq)


    ## Other Signals:
    ResoTrigger = TriggerFinderResonant(TreatmillRaw.resotrigger,NumFrames)
    VideoTrigger = TriggerFinderVideos(TreatmillRaw.camtrigger)
    Licking = LickingAnalysis(TreatmillRaw.licking,SampFreq)
    Licking_MM = TreatmillRaw['pump'].apply(lambda x:1 if x <= -0.2 else 0)
    Pump = PumpAnalysis(TreatmillRaw.pump)
    
    # OneResult-DataFrame
    TreatmillInformation = pd.concat([TreatmillRaw.time,FullCorrectedPosi,Laps,Velocity,ResoTrigger,VideoTrigger,Pump,Licking],axis=1)
    TreatmillInfomration['Licking_MM'] = TreatmillRaw['pump'].apply(lambda x: 1 if x <= -0.02 else 0)
    TreatmillInformation.rename(columns={'time':'Time'}, inplace=True)
    # Saving:
    if SavingName != None:
        SavingName = SavingName+ '_processed.h5'
        TreatmillInformation.to_hdf(SavingName,key='TreatmillInformation',mode='w')
    
    # Plotting:
    if Plotting != None:
        if Plotting <= 2:
            plt.ioff()
        else:
            plt.ion()
        if Plotting >=1:
            Figure = plt.figure()
            Figure.set_size_inches(11.69, 8.27, forward=True)
        if Plotting == 1 or Plotting == 2:
            Figure.set_dpi(300)   
            
        # Title:
        GridTitle = gridspec.GridSpec(1,1)
        GridTitle.update(left=0.075, bottom= 0.91, top = 0.99, right=0.98 , hspace=0.1)
        MainTitle = plt.subplot(GridTitle[0,0])
        MainTitle.axis('off')
        if SavingName == None:
            SavingName = 'TBA'
        MainTitleText = 'Primary Treadmill Analysis for ' + SavingName
        MainTitle.text(0, 0.9 , MainTitleText, transform=MainTitle.transAxes, fontsize=14, verticalalignment='top',fontweight='bold')
            
        # Plotting:
        Grid = gridspec.GridSpec(5,1)
        Grid.update(left=0.075, bottom= 0.075, top = 0.9, right=0.98 , hspace=0.1)
        
        # XLabelCalcs:
        XLabels = np.arange(TreatmillInformation.Time.min(), TreatmillInformation.Time.max(), 40)
        XLabelsBlank = ['']*len(XLabels)
        Labelfontsize = 8
        
        # 1 Position and Lap:
        print('StartPlotting')
        PositionPlot = plt.subplot(Grid[0,0])
        grouped = TreatmillInformation.groupby('Lap')
        for key, group in grouped:
            group.plot(ax=PositionPlot, kind='line', x='Time', y='Position')
            PositionPlot.annotate(int(group.Lap[group.index[0]]),xy=(group.Time.mean(),TreatmillInformation.Position.max()),xytext=(0,-10),xycoords='data',textcoords='offset points', fontsize = 10)
        PositionPlot.get_legend().remove()
        PositionPlot.spines['top'].set_visible(False)
        PositionPlot.spines['right'].set_visible(False) 
        PositionPlot.set_xticks(XLabels)
        PositionPlot.set_xticklabels(XLabelsBlank)
        PositionPlot.set_yticks(np.arange(TreatmillInformation.Position.min(), TreatmillInformation.Position.max()+1, 40))
        PositionPlot.tick_params(axis='both', which='major', labelsize=Labelfontsize)
        PositionPlot.tick_params(axis='both', which='minor', labelsize=Labelfontsize)
        PositionPlot.set_ylabel('BeltPosition [cm]',fontsize=Labelfontsize)
        
        # 2 Velocity
        VeloPlot = plt.subplot(Grid[1,0])
        VeloPlot.plot(TreatmillInformation.Time,TreatmillInformation.Velocity,'k')
        VeloPlot.set_xlim([TreatmillInformation.Time.min(),TreatmillInformation.Time.max()])
        VeloPlot.spines['top'].set_visible(False)
        VeloPlot.spines['right'].set_visible(False) 
        VeloPlot.set_xticks(XLabels)
        VeloPlot.set_xticklabels(XLabelsBlank)
        VeloPlot.tick_params(axis='both', which='major', labelsize=Labelfontsize)
        VeloPlot.tick_params(axis='both', which='minor', labelsize=Labelfontsize)
        VeloPlot.set_ylabel('Velocity [cm/sec]',fontsize=Labelfontsize)
        
        # 3 Pump and licks:
        LickPlot = plt.subplot(Grid[2,0])
        LickPlot.plot(TreatmillRaw.time,TreatmillRaw.pump,color=[0.8,0.8,0.8],linewidth=0.5)
        # Plot results scaled up:
        Pump_Scaled = OwnScaler(TreatmillInformation.Pump,0,TreatmillRaw.pump.max())
        LickPlot.plot(TreatmillInformation.Time,Pump_Scaled,color=[0.964, 0.494, 0.054],linewidth=1)
        # Licking:
        LickPlot.plot(TreatmillRaw.time,OwnScaler(TreatmillRaw.licking,0,TreatmillRaw.pump.max()),color=[0.5,0.5,0.5],linewidth=0.5)
        LickPlot.plot(TreatmillInformation.Time[TreatmillInformation.Licks>0.5],TreatmillInformation.Licks[TreatmillInformation.Licks>0.5]*Pump_Scaled.max(),'o',color=[0.231, 0.486, 0.094],markersize=3)
        # Apperance: 
        LickPlot.set_xlim([TreatmillInformation.Time.min(),TreatmillInformation.Time.max()])
        LickPlot.spines['top'].set_visible(False)
        LickPlot.spines['right'].set_visible(False)         
        LickPlot.set_xticks(XLabels)
        LickPlot.set_xticklabels(XLabelsBlank)
        LickPlot.axes.get_yaxis().set_ticks([])
        LickPlot.set_ylabel('Licking & Pump',fontsize=Labelfontsize)
       
        # ResoTrigger:
        Resoplot = plt.subplot(Grid[3,0])
        Resoplot.plot(TreatmillRaw.time,TreatmillRaw.resotrigger,color=[0.8,0.8,0.8],linewidth=0.25)
        Resoplot.plot(TreatmillInformation.Time,OwnScaler(TreatmillInformation.ResoTrigger,TreatmillRaw.resotrigger.min(),TreatmillRaw.resotrigger.max()),color=[0.964, 0.494, 0.054],linewidth=1)
        # Annotate: Number of Frames:
        InfoText = ('%.0f Frames' % int(TreatmillInformation.ResoTrigger.max())) 
        Resoplot.annotate(InfoText,xy=(TreatmillInformation.Time.max()/2,TreatmillRaw.resotrigger.max()/2),xytext=(0,0),xycoords='data',textcoords='offset points', fontsize = 10)
        # Appreance:
        Resoplot.set_xlim([TreatmillInformation.Time.min(),TreatmillInformation.Time.max()])
        Resoplot.spines['top'].set_visible(False)
        Resoplot.spines['right'].set_visible(False) 
        Resoplot.set_xticks(XLabels)
        Resoplot.set_xticklabels(XLabelsBlank)
        Resoplot.axes.get_yaxis().set_ticks([])
        Resoplot.set_ylabel('Frame Trigger \n Resonant',fontsize=Labelfontsize)
        
        # CamTrigger:
        Camplot = plt.subplot(Grid[4,0])
        Camplot.plot(TreatmillRaw.time,TreatmillRaw.camtrigger,color=[0.8,0.8,0.8],linewidth=0.25)
        Camplot.plot(TreatmillInformation.Time,OwnScaler(TreatmillInformation.VideoTrigger,TreatmillRaw.camtrigger.min(),TreatmillRaw.camtrigger.max()),color=[0.964, 0.494, 0.054],linewidth=1)
        # Annotate: Number of Frames:
        InfoText = ('%.0f Frames' % int(TreatmillInformation.VideoTrigger.max())) 
        Camplot.annotate(InfoText,xy=(TreatmillInformation.Time.max()/2,TreatmillRaw.camtrigger.max()/2),xytext=(0,0),xycoords='data',textcoords='offset points', fontsize = 10)
        # Appreance:
        Camplot.set_xlim([TreatmillInformation.Time.min(),TreatmillInformation.Time.max()])
        Camplot.spines['top'].set_visible(False)
        Camplot.spines['right'].set_visible(False) 
        Camplot.set_xticks(XLabels)
        Camplot.tick_params(axis='both', which='major', labelsize=Labelfontsize)
        Camplot.tick_params(axis='both', which='minor', labelsize=Labelfontsize)
        Camplot.set_xlabel('Time [sec]',fontsize=Labelfontsize)
        Camplot.axes.get_yaxis().set_ticks([])
        Camplot.set_ylabel('Frame Trigger \n Cams',fontsize=Labelfontsize)
        
        # Saving Figure:                
        if Plotting == 1:
            SavingName = SavingName+'_PrimaryTreadmillAnalysis'
            DDVarious_Plotting.save(SavingName, ext="png", close=False, verbose=True)
            plt.close('All')
            plt.ion()
        if Plotting == 2: 
            SavingName = SavingName+'_PrimaryTreadmillAnalysis'
            DDVarious_Plotting.save(SavingName, ext="svg", close=True, verbose=True)
            plt.close('All')
            plt.ion()
        
    return TreatmillInformation
    
    
    

''' Initial Processing Of BeltInformation: '''
def NIposition_processing(Position_Raw):
    # NI-Board with increasing numbers until points until value of 32767. Then starting from -32768.0
    # Get Continous Position Signal:  
    PositionMax = np.max(Position_Raw)
    PositionMin = np.min(Position_Raw)
    PositionDiff = PositionMax-PositionMin
    PositionJumps = Position_Raw.index[Position_Raw.diff()<-20000].tolist()
    for PositionIndex in PositionJumps:
        Position_Raw[PositionIndex:] = Position_Raw[PositionIndex:] + PositionDiff    
    return Position_Raw

def VelocityCalculationKalMan(Position,SampFreq,Plotting=None):
    # Using Kalman-Filter to extraploate Velocity
    # Constant = 1/SamplingRate NI board
    TimeWindow = 1/SampFreq
#    TimeWindow = 0.00001
    posF, velocity = Kalman_Filt_v(Position, TimeWindow) 
    # Ass DataFrame:
    velocity = pd.DataFrame(columns = ['Velocity'], data=velocity)
    # Plotting:
    if Plotting != None:
        Scaler = MinMaxScaler(feature_range=(np.min(Position),np.max(Position)))
        Velocity_scaled = Scaler.fit_transform(velocity['Velocity'].to_frame()).flatten()
        plt.figure()
        plt.plot(Velocity_scaled)
        plt.plot(Position)
    return posF, velocity
    
def FindLaps(Position,LapTrigger,Plotting=None):
    LapJumps =[]
    LapJumps = Position.index[LapTrigger.diff()>0.5].tolist()
    # Clean LapJumps: Stange 
    LapJumpsDiff =[]
    LapJumpsDiff = np.where(np.diff(LapJumps)<50000) 
    if len(LapJumpsDiff) > 0:
        LapJumps = np.delete(LapJumps, LapJumpsDiff)
    
    Lap = np.ones(len(Position.index))
    # Zero Down Position:
    if Position[0]>0:
        Position = Position - Position.min()
        
    for idx, posindex in enumerate(LapJumps):
        Position[posindex:] = Position[posindex:] - Position[posindex]
        Lap[posindex:] = Lap[posindex:] +1
    
    # Adjust first Jump:
    if len(LapJumps) > 0: # if at least one lap signal has been triggered
        Position[:LapJumps[0]] = Position[:LapJumps[0]] + (max(Position) - max(Position[0:LapJumps[0]]))
    
    # # Ass DataFrame:
    Position = pd.DataFrame(columns = ['Position'], data=Position.values) 
    Lap = pd.DataFrame(columns = ['Lap'], data=Lap)
    # Plotting:
    if Plotting != None:
        plt.ion()
        Scaler = MinMaxScaler(feature_range=(np.min(Lap.Lap),np.max(Lap.Lap)))
        Position_scaled = Scaler.fit_transform(Position['Position'].to_frame()).flatten()
        plt.figure()
        # plt.plot(Position)
        plt.plot(Position_scaled)
        plt.plot(LapTrigger)
        plt.plot(Lap)
        
    return Position, Lap  

def CorrectForBigPositionJumps(PositionTable):
    DiffPosi = PositionTable.index[PositionTable.position.diff() > 10.000].tolist()
    if DiffPosi:
        PositionTableCorrected = PositionTable
        i = 0
        while i < len(DiffPosi):
            PositionTableCorrected.position[DiffPosi[i]:] = PositionTableCorrected.position[DiffPosi[i]:]-(PositionTableCorrected.position[DiffPosi[i]]-PositionTableCorrected.position[DiffPosi[i]-1])    
            i += 1
    else:      
        PositionTableCorrected = PositionTable
#    PositionTableCorrected = PositionTable
    return PositionTableCorrected

def ScalingToBelt(Position,MinBelt,MaxBelt):  
    Scaler = MinMaxScaler(feature_range=(MinBelt, MaxBelt))
    NewPositions = Scaler.fit_transform(Position['Position'].to_frame()).flatten()  
    # Ass DataFrame:
    NewPositions = pd.DataFrame(columns = ['Position'], data=NewPositions) 
    return NewPositions  


''' Analysis of Triggers: '''   
def TTLTriggerFinder(TriggerSignal,ThresholdAmp=None,ThresholdNumFrames=None):
    # TriggerSignal from Amplitude:
    if ThresholdAmp == None:
        SignalsSTD = np.std(TriggerSignal)
        TriggerPointsPre = TriggerSignal.index[TriggerSignal.diff()>2*SignalsSTD].values
    else:
        TriggerPointsPre = TriggerSignal.index[TriggerSignal.diff()>ThresholdAmp].values    
    # TriggerSignal cleaned with NumFrames in between:
    if ThresholdNumFrames == None:
        TrueTrigger = np.where(np.diff(TriggerPointsPre)>2)
    else:
        TrueTrigger = np.where(np.diff(TriggerPointsPre)>ThresholdNumFrames)
    TriggerPoints = TriggerPointsPre[TrueTrigger]  
    
    return TriggerPoints   

def TriggerFinderResonant(TriggerSignal,NumFrames,Plotting=None):
    #Plotting = 1
    # Digitalize TriggerSignal Signal:
    DigiTriggerSignal = TriggerSignal.copy()
    DigiTriggerThreshold = (DigiTriggerSignal.max()/2)
    DigiTriggerSignal.loc[DigiTriggerSignal >= DigiTriggerThreshold] = DigiTriggerSignal.max()
    DigiTriggerSignal.loc[DigiTriggerSignal <= DigiTriggerThreshold] = 0
    # Differentiation Analysis
    # TriggerIndiciesP = TTLTriggerFinder(TriggerSignal,2,20)
    TriggerIndiciesP = TTLTriggerFinder(DigiTriggerSignal,2,20)
#    TriggerIndicies = TriggerIndiciesP[0:] # First Trigger is Good!
    TriggerIndicies = TriggerIndiciesP[1:] # First Trigger is Bullshit!
    if int(len(TriggerIndicies)) != int(NumFrames):
        print('8ung: Number of Frames does not fit!')
        print('Num Triggers: ', len(TriggerIndicies), 'NumFrames: ',NumFrames)
 
    # Create Column for DataFrame:
    ResoTrigDuration = int(np.mean(np.diff(TriggerIndicies)))
    ResonantTriggerColumn = np.zeros(len(TriggerSignal))
    i = 1
    while i < len(TriggerIndicies):
        ResonantTriggerColumn[TriggerIndicies[i-1]:TriggerIndicies[i]] = i
        i += 1
    # Last Trigger:
    ResonantTriggerColumn[TriggerIndicies[-1]:TriggerIndicies[-1]+ResoTrigDuration]=i    
    # Ass DataFrame:
    ResonantTriggerColumn = pd.DataFrame(columns = ['ResoTrigger'], data=ResonantTriggerColumn) 
    # Plotting:
    if Plotting != None:
        Scaler = MinMaxScaler(feature_range=(np.min(TriggerSignal),np.max(TriggerSignal)))
        ResonantTrigger_scaled = Scaler.fit_transform(ResonantTriggerColumn)
        plt.figure()
        plt.plot(TriggerSignal)
        plt.plot(DigiTriggerSignal,'r')
        plt.plot(TriggerIndicies,np.ones(len(TriggerIndicies)),'o')
        plt.plot(ResonantTrigger_scaled)
#        end
    return ResonantTriggerColumn        

def TriggerFinderVideos(TriggerSignal,Plotting=None):
    # Digitalize TriggerSignal Signal:
    DigiTriggerSignal = TriggerSignal.copy()
    DigiTriggerThreshold = (DigiTriggerSignal.max()/2)
    DigiTriggerSignal.loc[DigiTriggerSignal >= DigiTriggerThreshold] = DigiTriggerSignal.max()
    DigiTriggerSignal.loc[DigiTriggerSignal <= DigiTriggerThreshold] = 0
    #TriggerIndicies = TTLTriggerFinder(TriggerSignal,1,10)
    TriggerIndicies = TTLTriggerFinder(DigiTriggerSignal,1,10)
    # Create Column for DataFrame:
    TrigDuration = int(np.mean(np.diff(TriggerIndicies)))
    VideoTriggerColumn = np.zeros(len(TriggerSignal))
    i = 1
    while i < len(TriggerIndicies):
        VideoTriggerColumn[TriggerIndicies[i-1]:TriggerIndicies[i]] = i
        i += 1
    # Last Trigger:
    VideoTriggerColumn[TriggerIndicies[-1]:TriggerIndicies[-1]+TrigDuration]=i
    # Ass DataFrame:
    VideoTriggerColumn = pd.DataFrame(columns = ['VideoTrigger'], data=VideoTriggerColumn) 
    # Plotting:
    if Plotting != None:
        Scaler = MinMaxScaler(feature_range=(np.min(TriggerSignal),np.max(TriggerSignal)))
        VideoTrigger_scaled = Scaler.fit_transform(VideoTriggerColumn)
        plt.figure()
        plt.plot(TriggerSignal)
        plt.plot(TriggerIndicies,np.ones(len(TriggerIndicies)),'o')
        plt.plot(VideoTrigger_scaled)
        end
    return VideoTriggerColumn


''' Analysis of Further Information: ''' 
def OwnScaler(Series,Min,Max):  
    Scaler = MinMaxScaler(feature_range=(Min, Max))
    Scaled = Scaler.fit_transform(Series.to_frame()).flatten()  

    return Scaled  

def LickingAnalysis(LickingSignal,SampFreq,Plotting=None):
    # Window Threshold: 1 sec
    # ZeroBaseline:
    LickingZeroed = LickingSignal - np.mean(LickingSignal)
    # Hilbert-Transform for Envelope:
    Analytic_signal = hilbert(LickingZeroed)
    Licking_Envelope = np.abs(Analytic_signal)
    # Low-Pass Butter filter:
    fc = 2  # Cut-off frequency of the filter
    w = fc / (SampFreq / 2) # Normalize the frequency to SamplingFrequency
    b, a = butter(5, w, 'low')
    LickingEnvFilt = filtfilt(b, a, Licking_Envelope)
    LickingEnvFilt[0:int(1.5*SampFreq)] = 0
    
    # Find Lick-Events:
    LickAmpThreshold = np.std(LickingEnvFilt)*2
    Lickevents,LickAmpsDic = find_peaks(LickingEnvFilt,height=LickAmpThreshold)#,ThresholdAmp=LickAmpThreshold,ThresholdNumFrames=10)
    LickAmps = LickAmpsDic['peak_heights']
    # Control in Time Domain (Window in which two licks can occure):
    WindowSize = 1*SampFreq
    i = 0
    TrueLickEvents = []
    while i < len(Lickevents):
#        print(i)
        DistanceForSingle = np.absolute(Lickevents-Lickevents[i])
#        print(DistanceForSingle)
        Possible = np.where(DistanceForSingle<WindowSize)
#        print(Possible)
        if len(Possible[0]) > 1:
            # print('First')
            MaxInt = np.argmax(LickAmps[Possible[0]])   
            MaxIdx = Possible[0][MaxInt]
            TrueLickEvents.append(Lickevents[MaxIdx])
            i = i+len(Possible[0])-1
        else:
            # print('Second')
            TrueLickEvents.append(Lickevents[i]) 
            i += 1
    
    # Get Array for LickEvents
    LickingColumn = np.zeros(shape=(len(LickingSignal),1))
    i = 0
    while i < len(TrueLickEvents):
        LickingColumn[TrueLickEvents[i]]=1
        i+=1
    # Ass DataFrame:
    LickingColumn = pd.DataFrame(columns = ['Licks'], data=LickingColumn)
    
    # Plotting:
    if Plotting != None:
        plt.figure()
        plt.plot(LickingZeroed)
        plt.plot(LickingEnvFilt)
        plt.plot(TrueLickEvents,LickingEnvFilt[TrueLickEvents],'o')
        end
    return LickingColumn
    
def PumpAnalysis(PumpSignal,Plotting=None):
    # Digitalize Pump Signal:
    DigiPumpSignal = PumpSignal.copy()
    DigiThreshold = (DigiPumpSignal.max()/2)
    DigiPumpSignal.loc[DigiPumpSignal >= DigiThreshold] = DigiPumpSignal.max()
    DigiPumpSignal.loc[DigiPumpSignal <= DigiThreshold] = 0
    # Differentiation Analysis
    DiffPumpSignal = np.diff(DigiPumpSignal) #np.diff(PumpSignal)
    PumpThreshold = np.std(DiffPumpSignal)*50 #75
    PumpStarts,Amp = find_peaks(DiffPumpSignal,height=PumpThreshold)
    PumpEnds,Amp = find_peaks(DiffPumpSignal*-1,height=PumpThreshold)
    PumpTrigger = np.zeros((len(PumpSignal)))
    if len(PumpStarts) > len(PumpEnds):
        a = np.array([len(DiffPumpSignal)])
        PumpEnds = np.append(PumpEnds,a)
    #print('StartLen: ',len(PumpStarts))
    #print('EndLen: ',len(PumpEnds))

    i = 0
    while i < len(PumpStarts):
        PumpTrigger[PumpStarts[i]:PumpEnds[i]] = 1
        #print('From: ',PumpStarts[i], 'To: ',PumpEnds[i])
        i += 1
    # Ass DataFrame:
    PumpColumn = pd.DataFrame(columns = ['Pump'], data=PumpTrigger) 
    # Plotting:
    if Plotting != None:
        Scaler = MinMaxScaler(feature_range=(np.min(PumpSignal),np.max(PumpSignal)))
        PumpColumn_scaled = Scaler.fit_transform(PumpColumn)
        plt.figure()
        plt.plot(PumpSignal)
        plt.plot(PumpColumn_scaled)
        plt.plot(PumpStarts,PumpColumn_scaled[PumpStarts],'o')
        plt.plot(PumpEnds,PumpColumn_scaled[PumpEnds],'o')
        
    return PumpColumn
    
   
    

''' Downsampling: '''
def DownsamplingToColum(TreadmillData,ColumnName,DeleteZeroTrigger=None,AdaptedBinary=None,SavingName=None):
    ColumCategories = list(np.unique(TreadmillData[ColumnName].values))
    # Delete Zero Trigger:
    if DeleteZeroTrigger == 1:
        ColumCategories = [i for i in ColumCategories if i != 0] # Delete Zero ResoTrigger
    DownSampledTableList = []
    with tqdm(total=len(ColumCategories)) as pbar:
        i = 0
        while i < len(ColumCategories):
            InterimTable = TreadmillData.loc[TreadmillData[ColumnName]==ColumCategories[i]]
            DownSampledTableList.append(InterimTable.mean(axis=0))
            pbar.update(1)
            i += 1
    
    # Combine and transpose    
    DownSampledTableInterim = pd.concat(DownSampledTableList, axis=1)
    DownSampledTable = DownSampledTableInterim.transpose()
    
    # Adapt Lap and Position + Licks and Pump binarised:
    if AdaptedBinary != None:
        DownSampledTable.Pump[DownSampledTable.Pump > 0] = 1
        DownSampledTable.Licks[DownSampledTable.Licks > 0] = 1
        i = 0
        while i < len(DownSampledTable.index):
            if DownSampledTable.loc[i,'Lap']%1!=0:
                DownSampledTable.loc[i,'Lap'] = np.ceil(DownSampledTable.loc[i,'Lap'])  
                DownSampledTable.loc[i,'Position'] = 0
            i += 1
    
    if SavingName != None:
        SavingName = SavingName+ '_DownsampledTo' + ColumnName + '.h5'
        DownSampledTable.to_hdf(SavingName,key='TreatmillInformation',mode='w')    
            
        
    return DownSampledTable

def DownsamplingByWindow(TreadmillData,Windowsize,AdaptedBinary=None):
    CheckWindowSize = len(TreadmillData.index)/Windowsize
    
    if CheckWindowSize %1!=0:
        print('WindowSize does not fit with number of Datapoints!')
        return TreadmillData
    else:
        CheckWindowSize = int(CheckWindowSize)
        WindowLimits= list(range(0,CheckWindowSize,1))
        DownSampledTableList = []
        with tqdm(total=len(WindowLimits)) as pbar:
            i = 0
            while i < len(WindowLimits):
                Start = WindowLimits[i]*Windowsize
                if i < len(WindowLimits)-1:
                    Stop = WindowLimits[i+1]*Windowsize
                else:
                    Stop = len(TreadmillData.index)
                InterimTable = TreadmillData.iloc[Start:Stop]
                InterimTable.ResoTrigger[:] = Stop
                DownSampledTableList.append(InterimTable.mean(axis=0))
                pbar.update(1)
                i += 1         
        # Combine and transpose    
        DownSampledTableInterim = pd.concat(DownSampledTableList, axis=1)
        DownSampledTable = DownSampledTableInterim.transpose()
        
        # Adapt Lap and Position + Licks and Pump binarised:
        if AdaptedBinary != None:
            DownSampledTable.Pump[DownSampledTable.Pump > 0] = 1
            DownSampledTable.Licks[DownSampledTable.Licks > 0] = 1
            i = 0
            while i < len(DownSampledTable.index):
                if DownSampledTable.loc[i,'Lap']%1!=0:
                    DownSampledTable.loc[i,'Lap'] = np.ceil(DownSampledTable.loc[i,'Lap'])  
                    DownSampledTable.loc[i,'Position'] = 0
                i += 1
        
    
    return DownSampledTable



          











