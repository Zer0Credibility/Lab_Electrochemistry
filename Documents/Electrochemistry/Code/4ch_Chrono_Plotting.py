# encoding: utf-8
from plotnine import *
from plotnine.themes.themeable import axis_ticks_minor, axis_ticks_major
import pandas as pd
import numpy as np
import math
from time import sleep

#data = '../Datasets/All_Data/2017-06-27_Threaded_Rhodo_PC_NC-C_NC-NC_Glycerol_50umol/Rhodo_Chrono_NoGlycerol.xlsx'
#data = '../Datasets/All_Data/2017-05-12_No_Cells/2017-05-31_NC_NS_NC/NC_NC_NS.xlsx'
#data = '../Datasets/All_Data/2017-05-09_Rhodo_Glucose_V2/chronoamperometry_try_2_(day_2).xlsx'
#data = '../Datasets/All_Data/2017-05-09_Rhodo_Glucose_V2/chronoamperometry_try_2_day_3.xlsx'
#data = '../Datasets/All_Data/2017-05-09_Rhodo_Glucose_V2/Copy of chronoamperometry try 2 (day 2).xlsx'
#data = '../Datasets/All_Data/2017-05-12_No_Cells/2017-05-31_NC_NS_NC/NC_NC_NS.xlsx'
data = '../Datasets/All_Data/2017-10-09/Chronoamperometry_2017-10-09.xlsx'



df = pd.read_excel(data, encoding='utf8', skiprows=range(1, 2), dtype={ 'CH1': np.int32, 'Unnamed: 1': np.float64,
                                                                        'CH5': np.int32, 'Unnamed: 3': np.float64,
                                                                        'CH6': np.int32, 'Unnamed: 5': np.float64,
                                                                        'CH10': np.int32, 'Unnamed: 7': np.float64})

print (df)

sleep(.2)

#exit()
df.columns = ['Time', 'CH1', 'CH2_Time', 'CH2', 'CH3_Time', 'CH3', 'CH4_Time', 'CH4']
df = df.drop(['CH2_Time', 'CH3_Time', 'CH4_Time'], axis=1)
df = df.drop([20])
#df = df[:-5000]
df = df.round(decimals=4)
df['Anson'] = (1/np.sqrt(df['Time']))
df = df.drop([0])

print (df)


plot = (ggplot(df)
     + scale_x_log10()
     + scale_y_log10()
    # + axis_ticks_minor()
    # + axis_ticks_major()
    + ylab(u'Current (μAmperes)')
    + xlab('Time (Seconds)')
    # + xlab('1/sqrt(Time [sec])')
    # + theme_bw()
    + geom_line(aes(x='Time', y='CH1', group=1))
    + geom_line(aes(x='Time', y='CH2', group=1))
    + geom_line(aes(x='Time', y='CH3', group=1))
    + geom_line(aes(x='Time', y='CH4', group=1)))
    # + geom_line(aes(x='Anson', y='CH1', group=1))
    # + geom_line(aes(x='Anson', y='CH2', group=1))
    # + geom_line(aes(x='Anson', y='CH3', group=1)))



print (plot)

exit()
# Figure out x axis scale

pd.read_excel('file_name.xlsx', dtype={'a': np.float64, 'b': np.int32})

xmin = min(df['Time'])
xmax = max(df['Time'])
xscale = ((xmax-xmin)/10)
xscale = math.ceil(xscale / 500.0) * 500.0
xbreaks = np.arange(xmin, xmax, xscale)

# Figure out y axis scale

ymin = min(min(df['CH1']), min(df['CH2']), min(df['CH3']))
ymax = max(max(df['CH1']), max(df['CH2']), max(df['CH3']))
yscale = ((ymax-ymin)/10)
ybreaks = np.arange(ymin, ymax, yscale)


plot = (ggplot(df)
    #+ scale_x_discrete(breaks=xbreaks)
    #+ scale_y_discrete(breaks=ybreaks)
    + geom_line(aes(x='Time', y='CH1', group=1))
    + geom_line(aes(x='Time', y='CH2', group=1))
    + geom_line(aes(x='Time', y='CH3', group=1)))

#plot = (ggplot(df)
#    + geom_line(data=df, mapping=(aes(x='Time', y='CH1', group=1)))
#    + geom_line(aes(x='Time', y='CH2', group=1))
#    + geom_line(aes(x='Time', y='CH3', group=1)))

#print (df['Time'])
print (plot)

