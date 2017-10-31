# encoding: utf-8
from plotnine import *
from plotnine.themes.themeable import axis_ticks_minor, axis_ticks_major
import pandas as pd
import numpy as np
import math

data = '../Datasets/All_Data/2016-11-07_Rhodo/2016-11-07-Chronoamperometry_GOOD.txt'

df = pd.read_csv(data, sep='\t')
df.columns = ['Time', 'CH1']

print (df)

plot = (ggplot(df)
    #+ scale_x_log10()
    #+ scale_y_log10()
    + ylab(u'Current (μAmperes)')
    + xlab('Time (Seconds)')
    + geom_line(aes(x='Time', y='CH1', group=1)))


area = np.trapz(df['CH1'])

print (area)
print (plot)
