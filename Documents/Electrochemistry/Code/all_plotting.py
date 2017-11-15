# encoding: utf-8
from tqdm import *
from plotnine import *
from time import sleep
import pandas as pd
from scipy import stats

#data = '../Datasets/All_Data/2017-06-27_Threaded_Rhodo_PC_NC-C_NC-NC_Glycerol_50umol/Rhodo_Chrono_NoGlycerol.xlsx'
#data = '../Datasets/All_Data/2017-05-12_No_Cells/2017-05-31_NC_NS_NC/NC_NC_NS.xlsx'
#data = '../Datasets/All_Data/2017-05-09_Rhodo_Glucose_V2/chronoamperometry_try_2_(day_2).xlsx'
#data = '../Datasets/All_Data/2017-05-09_Rhodo_Glucose_V2/chronoamperometry_try_2_day_3.xlsx'
#data = '../Datasets/All_Data/2017-05-09_Rhodo_Glucose_V2/Copy of chronoamperometry try 2 (day 2).xlsx'
#data = '../Datasets/All_Data/2017-05-12_No_Cells/2017-05-31_NC_NS_NC/NC_NC_NS.xlsx'
#data = '../Datasets/All_Data/2017-10-31_Synechocystis_Stabilization_12Plate_Light_Partial/Chronoamperometry_2017_10_31_Synechocystis_light_700ul_Stabilization_Partial.xlsx'

#data = '../Datasets/All_Data/2017-11-03_LS_vs_NoLS/Chronoamperometry_5ml_Light_synechocystis_full_no_ls_pretreatment.xlsx'
#data = '../Datasets/All_Data/2017-11-03_LS_vs_NoLS/Chronoamperometry_5ml_Light_synechocystis_full_with_ls_pretreatment.xlsx'

# data2 = '../Datasets/All_Data/2017-11-08/Rhodo_5ml_lowlight_stabilization.xlsx'
# data2 = '../Datasets/All_Data/2017-11-09/Rhodo_Glycerol_lowlight_25mM_stabilized_v2.xlsx'
# data2 = '../Datasets/All_Data/2017-11-09/Rhodo_Glycerol_lowlight_25mM.xlsx'
# data2 = '../Datasets/All_Data/2017-11-13/Chronoamperometry_0.1mM_Fericyanide_5000s_900mV_Bias_Potential.xlsx'
# data2 = '../Datasets/All_Data/2017-11-13/Chronoamperometry_0.1mM_Fericyanide_5000s.xlsx'
#data2 = '../Datasets/All_Data/2017-11-11/Rhodo_Glucose_lowlight_25mM_stabilized.xlsx'

data1 = '../Datasets/All_Data/2017-11-14/Chronoamperometry_Combined_Cathode_Negative_Control_(maybesomeferricyanyde).xlsx'
data2 = '../Datasets/All_Data/2017-11-14/Chronoamperometry_Separate_Cathode_Negative_Control_(maybesomeferricyanyde).xlsx'
data3 = '../Datasets/All_Data/2017-11-14/Chronoamperometry_Separate+Combined_Cathode_Negative_Control_(maybesomeferricyanyde)_Full_Run.xlsx'

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

    return df, ch_names


def plot_dataframe(data_frame, span):
    from plotnine import ggplot, ylab, xlab, geom_line, aes, stat_smooth, geom_smooth

    plot = (

        (ggplot(data_frame, aes('Time', 'Current', color='Channel'))
            + ylab(u'Current (μA)')
            + xlab('Time (seconds)')
            + geom_line())
            # + scale_y_log10()
            # + scale_x_log10())
            # + theme_bw()
            # + scale_color_grey()

            # + geom_smooth(span=span, method='lowess'))

            # + stat_smooth(span=span, method='loess')

    )

    return plot


def loess_fit(x, y, span):
    from skmisc.loess import loess
    import numpy as np
    """
    loess fit and confidence intervals
    """
    # setup
    lo = loess(x, y, span=span)
    # fit
    lo.fit()
    # Predict
    prediction = lo.predict(x, stderror=True)
    # Compute confidence intervals
    ci = prediction.confidence(0.05)
    # Since we are wrapping the functionality in a function,
    # we need to make new arrays that are not tied to the
    # loess objects
    yfit = np.array(prediction.values)
    ymin = np.array(ci.lower)
    ymax = np.array(ci.upper)
    return yfit, ymin, ymax


def construct_loess_regression(raw_data):
    import pandas as pd
    import numpy as np

    df = construct_dataframe(data)[0]
    channels = construct_dataframe(data)[1]

    dfs = dict(list(df.groupby("Channel")))

    print('Constructing regression smoothing data series...')

    for i in range(0, len(channels)):
        print(dfs[channels[i]])

        y = pd.DataFrame.as_matrix(df, columns=(['Current']))
        x = pd.DataFrame.as_matrix(df, columns=['Time'])
        #x = np.ndarray.flatten(x)
        y = np.ndarray.flatten(y)

        print(x.shape)
        print(y.shape)

        span = 0.75

        yfit, ymin, ymax = loess_fit(x, y, span)

        print yfit

        # print(prediction.values)

        exit()

    # print (dfs)
    exit()

    return(df)


def lowess(x, y, f=2. / 3., iter=1):
    from math import ceil
    import numpy as np
    from scipy import linalg


    """lowess(x, y, f=2./3., iter=3) -> yest
    Lowess smoother: Robust locally weighted regression.
    The lowess function fits a nonparametric regression curve to a scatterplot.
    The arrays x and y contain an equal number of elements; each pair
    (x[i], y[i]) defines a data point in the scatterplot. The function returns
    the estimated (smooth) values of y.
    The smoothing span is given by f. A larger value for f will result in a
    smoother curve. The number of robustifying iterations is given by iter. The
    function will run faster with a smaller number of iterations.
    """
    n = len(x)
    r = int(ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in tqdm(range(n))]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in tqdm(range(n)):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest


def construct_lowess_regression(raw_data, span):
    '''
    Creates a smoothed regression based on the Lowess algorithm.
    '''

    import numpy as np
    from statsmodels.nonparametric.smoothers_lowess import lowess

    df = construct_dataframe(raw_data)[0]

    channels = construct_dataframe(raw_data)[1]

    dfs = dict(list(df.groupby("Channel")))

    dfs_lo = []

    for i in tqdm(range(0, len(channels))):

        dfs_i = dfs[channels[i]]

        x = dfs_i['Time']
        x = np.asarray(x)

        y = dfs_i['Current']
        y = np.asarray(y)

        lo = lowess(y, x, frac=span, it=1, delta=0.0, is_sorted=True, return_sorted=False)

        dfs_lo.append(lo)

    dfs_lo = np.concatenate(dfs_lo).ravel()  # .tolist()

    # print (dfs_lo)

    #exit()

    df['Regression'] = dfs_lo

    return df


def calculate_median_absolute_deviation_from_signal(data_frame, raw_data, df_name):
    import numpy as np
    from time import sleep
    import pandas as pd

    df = data_frame

    channels = construct_dataframe(raw_data)[1]

    dfs = dict(list(df.groupby("Channel")))

    ads_list = []
    ch_list = []

    print ('calculating median absolute deviation of measured current from regression signal...')

    for i in tqdm(range(0, len(channels))):

        dfs_i = dfs[channels[i]]

        n = dfs_i['Current']
        n = np.asarray(n)

        s = dfs_i['Regression']
        s = np.asarray(s)

        d = np.median(abs(np.subtract(n, s)))

        ads_list.append(d)

        ch_list.append('CH' + str(i+1))

    ads_df = pd.DataFrame(ads_list, columns=['Deviation'])

    ads_df['Experiment'] = df_name
    ads_df['Channel'] = ch_list

    #ads_df['Channel']

    print (ads_df)

    return ads_df


def calculate_absolute_deviation_from_signal_per_channel(data_frame, raw_data):
    import numpy as np
    from time import sleep
    import pandas as pd

    df = data_frame

    channels = construct_dataframe(raw_data)[1]

    dfs = dict(list(df.groupby("Channel")))

    ads_list = []

    print ('calculating median absolute deviation of measured current from regression signal...')

    for i in tqdm(range(0, len(channels))):

        dfs_i = dfs[channels[i]]

        n = dfs_i['Current']
        n = np.asarray(n)

        s = dfs_i['Regression']
        s = np.asarray(s)

        d = abs(np.subtract(n, s))

        ads_list.append(d)

    dfs_ads = np.concatenate(ads_list).ravel()  # .tolist()

    df['Deviation'] = dfs_ads

    return df


def plot_median_absolute_deviation_from_signal(ads):

    plot = (

        (ggplot(ads, aes('Experiment', 'Deviation'))
         + stat_boxplot())

    )

    # print (plot)

    return plot


def plot_absolute_deviation_from_signal_per_channel(dev_df):

    plot = (

        (ggplot(dev_df, aes('Channel', 'Deviation'))
         + stat_boxplot())

    )

    return plot


def compare_absolute_deviation_between_runs(data1, data2):

    span = 0.2
    df1_name = 'no_ls'
    df2_name = 'ls'

    df1 = construct_lowess_regression(data1, span)
    df2 = construct_lowess_regression(data2, span)

    ads_df1 = calculate_median_absolute_deviation_from_signal(df1, data1, df1_name)
    ads_df2 = calculate_median_absolute_deviation_from_signal(df2, data2, df2_name)

    all_df = ads_df1.append(ads_df2)

    plot = plot_median_absolute_deviation_from_signal(all_df)
    print(plot)

    return all_df


def anova_test(data_frame, raw_data):
    import numpy as np
    from time import sleep
    import pandas as pd

    df = data_frame

    channels = construct_dataframe(raw_data)[1]

    ch_list = []
    og_ch_list = []

    print ('Running Analysis of Variance...')

    for i in (range(0, len(channels))):

        ch_list.append('CH' + str(i + 1))
        og_ch_list.append('CH' + str(i + 1))

    for i in range(0, len(ch_list)):

        ch_list[i] = df['Channel'] == ch_list[i]  # Construct the true/false dataframe
        ch_list[i] = df[ch_list[i]]               # Each Dataframe has only one channel
        ch_list[i] = ch_list[i]['Current']        # Pull only the current out of the dataframe

    f_val, p_val = stats.f_oneway(ch_list)

    print ("One-way ANOVA P =", p_val)
    print ('One-Way ANOVA F =', f_val)


    print (ch_list)
    exit()

    return anova



#all_df = compare_absolute_deviation_between_runs(data1, data2)

#exit()

span = 0.2
df_name = 'no_ls'

df = construct_dataframe(data3)
df = df[0]

anova_test(df, data3)

exit()

CH1 = df['Channel'] == 'CH1'
CH1 = df[CH1]

variance = CH1['Current'].var()
sem = stats.sem(CH1['Current'])  # standard error of the mean

#df = construct_lowess_regression(data2, span)

print (variance)
print (sem)
exit()

sleep(.1)

plot = plot_dataframe(df, span)

print(plot)
exit()

# dev_df = calculate_absolute_deviation_from_signal_per_channel(df, data)

ads = calculate_median_absolute_deviation_from_signal(df, data, df_name)

plot = plot_median_absolute_deviation_from_signal(ads)

# plot = plot_absolute_deviation_from_signal_per_channel(dev_df)

print (plot)

# print (ads)

exit()

print (df)

# df = construct_dataframe(data)[0]

plot = plot_dataframe(df, span)

print(plot)