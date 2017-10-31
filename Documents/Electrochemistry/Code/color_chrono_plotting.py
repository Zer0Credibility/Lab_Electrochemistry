# encoding: utf-8
from plotnine import *
from plotnine.themes.themeable import axis_ticks_minor, axis_ticks_major
import pandas as pd
import numpy as np
import math

#data = '../Datasets/All_Data/2017-06-27_Threaded_Rhodo_PC_NC-C_NC-NC_Glycerol_50umol/Rhodo_Chrono_NoGlycerol.xlsx'
#data = '../Datasets/All_Data/2017-05-12_No_Cells/2017-05-31_NC_NS_NC/NC_NC_NS.xlsx'
#data = '../Datasets/All_Data/2017-05-09_Rhodo_Glucose_V2/chronoamperometry_try_2_(day_2).xlsx'
#data = '../Datasets/All_Data/2017-05-09_Rhodo_Glucose_V2/chronoamperometry_try_2_day_3.xlsx'
#data = '../Datasets/All_Data/2017-05-09_Rhodo_Glucose_V2/Copy of chronoamperometry try 2 (day 2).xlsx'
data = '../Datasets/All_Data/2017-05-12_No_Cells/2017-05-31_NC_NS_NC/NC_NC_NS.xlsx'


def construct_dataframe(raw_data):

    df = pd.read_excel(raw_data, encoding='utf8', skiprows=range(1, 2))

    headers = list(df.columns.values)

    times = headers[0::2]
    bad_times = headers[2::2]
    data = headers[1::2]

    ch_names = [s.encode('ascii') for s in times]
    ch_names = [x.strip(' ') for x in ch_names]

    # print(ch_names)

    df = df.drop(bad_times, axis=1)

    df.columns = ['Time'] + ch_names

    #print(df)
    #exit()

    df = df.astype(np.float64)
    df = df.round(decimals=4)

    df = pd.melt(df, id_vars=['Time'], value_vars=ch_names, var_name='Channel', value_name='Current')

    return df


def plot_dataframe(data_frame):

    plot = (ggplot(data_frame)
            + ylab(u'Current (μA)')
            + xlab('Time (seconds)')
            + geom_line(aes(x='Time', y='Current', group=1, color='factor(Channel)')))
    return(plot)


df = construct_dataframe(data)

plot = plot_dataframe(df)

print(plot)




exit()

#channels = [x + ': np.int32' for x in channels]
#times = [x + ': np.float64' for x in times]

exit()

print (channels)
print(times)

df = pd.melt(df)


print(df.shape)

# print(df.iloc[::])
# print(df[:0])

#datatype =

exit()

df = pd.read_excel(data, encoding='utf8', skiprows=range(1, 2), dtype={ 'CH1': np.int32, 'Unnamed: 1': np.float64,
                                                                        'CH2': np.int32, 'Unnamed: 3': np.float64,
                                                                        'CH3': np.int32, 'Unnamed: 5': np.float64})

#print (df)
#exit()
df.columns = ['Time', 'CH1', 'CH2_Time', 'CH2', 'CH3_Time', 'CH3']
df = df.drop(['CH2_Time', 'CH3_Time'], axis=1)
df = df.drop([20])
# df = df[:-5000]
df = df.round(decimals=4)
#df['Anson'] = (1/np.sqrt(df['Time']))
df = df.drop([0])

print (df)

df = pd.melt(df, id_vars=['Time'], value_vars=['CH1', 'CH2', 'CH3'], var_name='Channel', value_name='Current')

print (df)

plot = (ggplot(df)
    + ylab(u'Current (μA)')
    + xlab('Time (seconds)')
    + geom_line(aes(x='Time', y='Current', group=1, color='factor(Channel)')))


print (plot)
exit()