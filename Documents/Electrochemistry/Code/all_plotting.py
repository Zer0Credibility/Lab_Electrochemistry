# encoding: utf-8

#data = '../Datasets/All_Data/2017-06-27_Threaded_Rhodo_PC_NC-C_NC-NC_Glycerol_50umol/Rhodo_Chrono_NoGlycerol.xlsx'
#data = '../Datasets/All_Data/2017-05-12_No_Cells/2017-05-31_NC_NS_NC/NC_NC_NS.xlsx'
#data = '../Datasets/All_Data/2017-05-09_Rhodo_Glucose_V2/chronoamperometry_try_2_(day_2).xlsx'
#data = '../Datasets/All_Data/2017-05-09_Rhodo_Glucose_V2/chronoamperometry_try_2_day_3.xlsx'
#data = '../Datasets/All_Data/2017-05-09_Rhodo_Glucose_V2/Copy of chronoamperometry try 2 (day 2).xlsx'
#data = '../Datasets/All_Data/2017-05-12_No_Cells/2017-05-31_NC_NS_NC/NC_NC_NS.xlsx'
data = '../Datasets/All_Data/2017-10-31_Synechocystis_Stabilization_12Plate_Light_Partial/Chronoamperometry_2017_10_31_Synechocystis_light_700ul_Stabilization_Partial.xlsx'


def construct_dataframe(raw_data):
    import pandas as pd
    import numpy as np

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
    from plotnine import ggplot, ylab, xlab, geom_line, aes, stat_smooth, geom_smooth

    plot = (

        (ggplot(data_frame, aes('Time', 'Current', color='Channel'))
        + ylab(u'Current (μA)')
        + xlab('Time (seconds)')
        + geom_line())
        + geom_smooth(method='loes')

    )

    return plot


df = construct_dataframe(data)

plot = plot_dataframe(df)

print(plot)