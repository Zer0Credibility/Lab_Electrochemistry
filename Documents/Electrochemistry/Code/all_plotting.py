# encoding: utf-8
from tqdm import *

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

    return df, ch_names


def plot_dataframe(data_frame, span):
    from plotnine import ggplot, ylab, xlab, geom_line, aes, stat_smooth, geom_smooth

    plot = (

        (ggplot(data_frame, aes('Time', 'Current', color='Channel'))
        + ylab(u'Current (μA)')
        + xlab('Time (seconds)')
        + geom_line())
        + stat_smooth(span=span, method='lowess')

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


    import pandas as pd
    import numpy as np
    from statsmodels.nonparametric.smoothers_lowess import lowess

    df = construct_dataframe(raw_data)[0]

    #print (df.shape)
    #exit()

    channels = construct_dataframe(raw_data)[1]

    dfs = dict(list(df.groupby("Channel")))

    # dfs_lo = np.empty(shape=(73596, 2), dtype=np.float64)

    dfs_lo = []

    for i in tqdm(range(0, len(channels))):
        # print(dfs[channels[i]])

        dfs_i = dfs[channels[i]]

        x = dfs_i['Time']
        x = np.asarray(x)

        y = dfs_i['Current']
        y = np.asarray(y)


        #print(x.shape)
        #print(y.shape)

        lo = lowess(y, x, frac=span, it=1, delta=0.0, is_sorted=True, return_sorted=False)

        #lo.tolist()

        #print (lo)
        #print(lo.shape)
        #exit()

        #np.append(dfs_lo, lo)

        dfs_lo.append(lo)

    #print (dfs_lo.shape)

    dfs_lo = np.concatenate(dfs_lo).ravel()  # .tolist()

    print (dfs_lo)

    #exit()

    df['Regression'] = dfs_lo

    return df


span = 0.2

df = construct_lowess_regression(data, span)

print (df)

# df = construct_dataframe(data)[0]

plot = plot_dataframe(df, span)

print(plot)