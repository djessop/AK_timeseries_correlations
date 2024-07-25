#!/usr/bin/env python
# coding: utf-8

# In[54]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# timeseries_correlations.ipynb
# Created by D. E. Jessop, 2024-05-01 (ish)

import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from matplotlib.dates import DateFormatter, MonthLocator, DayLocator
from matplotlib.ticker import MultipleLocator
from datetime import datetime as DT, timedelta as TD
from timeSeriesAnalysis.timeSeriesAnalysis import butter_lowpass_filtfilt, secs_in_day

# from IPython.display import set_matplotlib_formats

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('pdf')

plt.rcParams['figure.dpi']  = 150
plt.rcParams['savefig.dpi'] = 150

# Definition of functions for testing
def sinusoid(x, offset, amplitude, frequency, phase):
    return offset + amplitude * np.sin(2 * np.pi * frequency * x - phase)

def gaussian(x, loc, scale):
    return np.exp( -((x - loc) / scale)**2 )

def square(x, start, stop):
    return np.where(np.logical_and(x > start, x <= stop), 1, 0)

# Simplifies processing of natural data
def detrend_normalise_data(data):
    d = scipy.signal.detrend(data)
    return d / np.linalg.norm(d)

def last_day_previous_month(ip_dt):
    """Returns the last day of the previous calendar month."""
    dts = DT.fromisoformat(ip_dt).replace(day=1) - TD(1) 
    return(dts.strftime("%F"))


# # Tests of understandability
# 
# Here, we run some simple tests to help understand the correlation functions and lags

# In[2]:


# Parameters for the sinusoids: offset, amplitude, "understandable" frequency and phase
p0 = [0, 1, 1, 0]
p1 = [0, 1, 1, np.pi / 4]  # Play around with the phase.  

n_samples = (501, 5001, 50_001)  # I get the same result regardless of number of samples

fig, axes = plt.subplots(nrows=2)

for n_samples in n_samples:
    x  = np.linspace(0, 4, n_samples)
    dx = x[1] - x[0]       # sampling interval
    fs = 1 / dx            # sampling frequcency
    y0 = sinusoid(x, *p0)
    y1 = sinusoid(x, *p1)

    a  = y0 / np.linalg.norm(y0)
    b  = y1 / np.linalg.norm(y1)

    corr = scipy.signal.correlate(a, b, 'same')
    lags = scipy.signal.correlation_lags(len(a), len(b), 'same') * dx  # lags multiplied by sampling interval

    axes[1].plot(lags, corr, label=f'{n_samples} samples')
    axes[1].set_ylim([-1.1, 1.1])

axes[0].plot(x, y0, '-C0', x, y1, '-C1')
axes[0].set_xlabel('Duration')
axes[0].set_ylabel('Signal')
axes[1].set_ylabel('Correlation')
axes[1].set_xlabel('Lag')
    
print(f"Signal y1 lags signal y0 by {-lags[np.argmax(corr)]}")
axes[1].legend(loc=1)
fig.tight_layout()
plt.show()


# The lag with a phase of $\pi/4$ (i.e. $\frac{\pi / 4}{2\pi} = 1/8$ cycles) is $\approx 0.125$.

# In[3]:


t = np.linspace(0, 10, 101)
dt = t[1] - t[0]
a, b = gaussian(t, 1, 1), gaussian(t, 4, 1)


fig, axes = plt.subplots(nrows=2)
axes[0].plot(t, a, '-', label='$a$')
axes[0].plot(t, b, '-', label='$b$')
axes[0].legend()
axes[0].set_title(f'Simple gaussians with offset of 3 time units')
axes[0].set_xlabel("``time''")
axes[0].set_ylabel('Signal')

a /= np.linalg.norm(a)
b /= np.linalg.norm(b)
corr = scipy.signal.correlate(a, b, 'same')
lags = scipy.signal.correlation_lags(len(a), len(b), 'same') * dt
axes[1].plot(lags, corr)
axes[1].annotate(r'$\int_{-\infty}^{\infty} a(t)b(t+\tau) dt$', 
                 xy=(-1.5, corr[lags==-1.5]), xytext=(-1.5, .8),
                 arrowprops=dict(arrowstyle='-', color='grey', connectionstyle='arc3,rad=-.1'))
axes[1].set_xlabel(r'Lags, $\tau$')
axes[1].set_ylabel('Correlation fn.')

for ax in axes: 
    ax.xaxis.set_minor_locator(MultipleLocator(.5))
    ax.yaxis.set_minor_locator(MultipleLocator(.25))

fig.tight_layout()

print(f'Signal $a$ comes after signal $b$ by {lags[np.argmax(corr)]}')


# Signal $a$ comes after signal $b$ by -3.0, i.e. $b$ lags $a$ by 3.

# Correlation tests taken from https://stackoverflow.com/questions/49372282/find-the-best-lag-from-the-numpy-correlate-output

# In[4]:


add_noise = True

x_1 = np.linspace(0, 10, 101)
x_2 = np.linspace(0, 7, 71)
data_1 = np.sin(x_1)
data_2 = np.cos(x_2)
dx  = x_1[1] - x_1[0]
if add_noise:
    data_1 += np.random.uniform(size=data_1.shape)
    data_2 += np.random.uniform(size=data_2.shape)

fig0, ax0 = plt.subplots()
ax0.plot(x_1, data_1, '.', label=r'$x(t)$')
ax0.plot(x_2, data_2, '.', label=r'$y(t)$')
ax0.legend()
ax0.set_xlabel('Time, $t$/[au]');
ax0.set_ylabel('Signal/[au]');
ax0.set_title('Time series of data');
# ax0.xaxis.set_major_locator(MultipleLocator(np.pi))
xlims = ax0.get_xlim()
ax0.set_xticks(np.arange(-np.pi, 5*np.pi, np.pi))
ax0.xaxis.set_minor_locator(MultipleLocator(np.pi/4))
labels = [r'$-\pi$', '$0$', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$']
ax0.set_xticklabels(labels)
ax0.set_xlim(xlims)


a, b = data_1.copy(), data_2.copy()
a /= np.linalg.norm(a)
b /= np.linalg.norm(b)
corr = scipy.signal.correlate(a, b, 'full')
lags = scipy.signal.correlation_lags(len(a), len(b), 'full') * dx
lag  = (corr.argmax() - (len(data_1) - 1)) * dx

fig, ax = plt.subplots()
plt.plot(lags, corr, '-', zorder=5)
ax.xaxis.set_major_locator(MultipleLocator(np.pi))
ax.xaxis.set_minor_locator(MultipleLocator(np.pi/4))
ax.axvline(2*np.pi, c='r', alpha=.3, zorder=3)
ax.axvline(-2*np.pi, c='r', alpha=.3, zorder=3)
ax.plot(lags[np.argmax(corr)], corr[np.argmax(corr)], 'ro')
ax.grid()
xlims = ax.get_xlim()
ax.set_xticks(np.arange(-3*np.pi, 5*np.pi, np.pi))
ax.xaxis.set_minor_locator(MultipleLocator(np.pi/4))
labels = [r'$-3\pi$', r'$-2\pi$', r'$-\pi$', '$0$', r'$\pi$', r'$2\pi$', r'$3\pi$', r'$4\pi$']
ax.set_xticklabels(labels);
ax.set_xlim(xlims);
ax.set_xlabel('Lags/[au]')
ax.set_ylabel('Correlation function/[-]')

# print(lag / np.pi, lags[np.argmax(corr)] / np.pi)
print(f"Signal $y$ is advanced by 1/4 period ($\pi/2$).  Max lag found at {(lags[np.argmax(corr)] / np.pi):.3f}$\pi$.")


# # Correlation of real data
# 
# Here, we're going to use the ground water level (GWL) data, self polarisation (SP) and sismic (count from WO-MC3).  As the GWL data is produced once per day, the two other data sets have been subsampled to match this sampling frequency: for SP, the resampling is the mean of the daily value; for the sismic data, this is the sum of the daily data.

# In[5]:


GWL_data = pd.read_csv('../data/GWL_modelling/GWL_TAS_model_inversion.csv', 
                       sep=',', parse_dates=['# Date (yyyy-mm-dd)'])
GWL_data = GWL_data.rename(mapper={'# Date (yyyy-mm-dd)': 'Date', 
                                   ' GWL (m) ': 'GWL', 
                                   ' error (m)': 'error'}, axis=1)
# print(GWL_data.head())

# Depth of Tarissan lake - for comparison with GWL
TAR_data = pd.read_csv('../data/OVSG_EAUX_2024-06-06.csv', sep=';', parse_dates=['Date']).dropna(subset=['Niveau (m)'])
TAR_data =  TAR_data[['Date', 'Niveau (m)']]
# print(TAR_data)

rain_data = pd.read_csv('../data/GWL_modelling/Rainfall_summit_avg.csv', 
                        sep=',', parse_dates=['# Date (yyyy-mm-dd)'])
rain_data = rain_data.rename(mapper={'# Date (yyyy-mm-dd)': 'Date', 
                                     ' Rainfall (mm)': 'rain'}, axis=1)
rain_data['rain'] = rain_data['rain'].astype(float)
rain_data['cumul'] = rain_data['rain'].cumsum()


# In[6]:


## Plot for SM
d = DT(2017, 1, 1)
data = GWL_data.query('Date > @d')
plt.plot(data['Date'], -data['GWL'], '-', label='GWL');

y_   = detrend_normalise_data(data['GWL'])
corr = scipy.signal.correlate(y_, y_, 'same')
lags = scipy.signal.correlation_lags(len(y_), len(y_), 'same')

data = TAR_data.query('Date > @d')
plt.plot(data['Date'], data['Niveau (m)'], '.', label='TAR');
ax = plt.gca()
ax2  = ax.twinx()

data = rain_data.query('Date > @d')
ax2.plot(data['Date'], data['rain'], '-C2', label='rain');

# ax = plt.gca()
# ask matplotlib for the plotted objects and their labels
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=2)
ax.xaxis.set_minor_locator(MonthLocator(7))
ax.set_xlabel('Date');
ax.set_ylabel('Profondeur du lac en dessous cratère/[m]');
ax2.set_ylabel('Cumulative rainfall/[m]')

fig = plt.gcf()
fig.savefig('../plots/GWL_timeseries_2017-present.pdf', dpi=150)


# In[7]:


SP_data  = pd.read_csv("../data/SP_all_converted.csv",
    #"../data/SP_05May2022-20April2023.csv",
    #'../data/PSGUADA_NOEUD_263_PS-data-as-joinbyfield-2024-05-06-09-51-08.csv', 
    #'../data/PSGUADA_NOEUD_263_PS-data-as-joinbyfield-2024-06-06-15-03-20.csv',
    sep=',', parse_dates=['Time'], dayfirst=True)

# print(SP_data.dtypes)
ps_label = 'PS' #'{ps_label}'

# mapper = {f'SP{ind}': f'{ps_label}{ind}' for ind in range(1, 9)}
# SP_data = SP_data.rename(mapper=mapper, axis=1)
# dtypes = {f'{ps_label}{ind}': float for ind in range(1, 9)}

# SP_data = SP_data.astype(dtypes)
   
# print(SP_data.head())

SP_data_daily = SP_data.copy()
SP_data_daily.index = SP_data['Time']
# SP_data_daily = SP_data_daily.loc[:, SP_data_daily.columns.str.contains('Average')]
SP_data_daily = SP_data_daily.loc[:, SP_data_daily.columns.str.contains('PS')]
SP_data_daily = SP_data_daily.resample('1D').mean()
SP_data_daily = SP_data_daily.reset_index().interpolate()

SP_data_daily


# In[8]:


fig, axes = plt.subplots(ncols=2, figsize=(9.2, 6.4))

data = rain_data['rain'].interpolate()
fs   = 2 / secs_in_day 

f, psd = scipy.signal.periodogram(data, fs, window='tukey')

ax = axes[0]

ax.loglog(f, psd / psd[1], '-', label='raw')

cutoff = fs / 32 # Filter out higher frequencies 
order  = 4
print(fs, cutoff)
rain_ff   = butter_lowpass_filtfilt(data, cutoff, fs, order=order)
ff, psd_f = scipy.signal.periodogram(rain_ff, fs, window='tukey')

ax.loglog(ff, psd_f/psd_f[1], '-', label='filtered')

dayInSec   = 1 / secs_in_day
weekInSec  = dayInSec  / 7
monthInSec = weekInSec  / 4     # Lunar month
intervals  = (dayInSec, weekInSec, monthInSec)
labels = ('1 day', '1 week', '1 month')
offset = 1. #np.sqrt(2)

# Indicate interesting intervals on PSD
for interval, label in zip(intervals, labels):
    hl = ax.axvline(interval,  c='r', lw=2, zorder=0)
    ht = ax.annotate(label, xy=(interval * offset, 1e-3), xycoords='data', 
                      color='r', rotation=-90, va='top', zorder=3)

ax.set_ylim((4.3524539244017724e-05, 6.3118732430955395))
ax.set_xlabel(r'Frequency/[Hz]')
ax.set_ylabel(r'Normalised PSD/[-]')

ax = axes[1]
ax.plot_date(rain_data['Date'], detrend_normalise_data(np.cumsum(data)), '-', label='raw')
ax.plot_date(rain_data['Date'], detrend_normalise_data(np.cumsum(rain_ff)), '-', label='filtered')

ax.set_ylabel('detrended and normalised cumulative rainfall/[-]');
ax.legend()
ax.set_title('Effect of filtering on signal');


# In[9]:


sismo = pd.read_csv('../data/WO_OVSG_MC3_OVSG_dump_bulletin.csv', 
                    sep=';', skiprows=2, 
                    names=['date', 'count', 'duration', 'amplitude', 
                           'magnitude', 'e(j)', 'longitude', 'latitude', 
                           'depth', 'type', 'file', 'locmode', 'loctype', 
                           'projection', 'operator', 'timestamp', 'id'], 
                    parse_dates=['date'], date_format='%Y%m%d %H%M%S.%f')
sismo = sismo[['date', 'count', 'duration', 'amplitude', 
               'longitude', 'latitude', 'depth']]
sismo['cumul'] = np.cumsum(sismo['count'])

sismo_daily = sismo.copy()
sismo_daily = sismo_daily.set_index('date')
sismo_daily = sismo_daily.drop(['amplitude', 'duration', 
                                'longitude', 'latitude', 'depth'], axis=1)

sismo_daily = sismo_daily.loc[:, ~sismo_daily.columns.str.contains('date')]
sismo_daily = sismo_daily.resample('1D').sum().reset_index()
sismo_daily['cumul'] = np.cumsum(sismo_daily['count'])


# In[10]:


## Load NAPN data, though currently there is no overlap with SP data
names = ['year', 'month', 'day', 'hours', 'minutes', 'seconds', 'temperature']
napn = pd.read_csv('../data/GGWNAPN0_all.txt', sep=' ', comment='#', names=names)
napn.index = pd.to_datetime(napn[['year', 'month', 'day', 'hours', 'minutes', 'seconds']], utc=True)
napn = napn['temperature']

napn_daily = napn.resample('1D').mean().reset_index()
napn = napn.reset_index()

# print(napn.head())
# print(napn_daily.head())


# ## Plots of raw data.

# In[11]:


fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(5, 8))
axes[0].plot_date(GWL_data['Date'], GWL_data['GWL'], '.', 
                  zorder=3, label='GWL')
axes[0].errorbar(GWL_data['Date'], GWL_data['GWL'], yerr=GWL_data['error'], 
                 marker='.', c='k', lw=.8, zorder=1)
axes[0].set_title('GWL data, relative values')
axes[0].set_ylabel('GWL or Tarrisan depth/[m]')
# axes[0].set_ylim([56, 94])
axes[0].plot_date(TAR_data['Date'], -TAR_data['Niveau (m)'], 
                  'o', label='TAR')

axes[1].plot_date(SP_data['Time'], 
                  # SP_data['{ps_label}1'], 
                  SP_data['PS1'], 
                  '.', zorder=3, label='raw data')
axes[1].plot_date(SP_data_daily['Time'],
                  SP_data_daily['PS1'],
                  # SP_data_daily['{ps_label}1'],
                  '-.', lw=4, zorder=3, label='daily average')
axes[1].set_xlim([SP_data['Time'].min(), SP_data['Time'].max()])
axes[1].set_ylabel('SP node 1/[mV]')
axes[1].set_title('SP data')

axes[2].plot_date(sismo['date'], sismo['cumul'], '.', 
                  label='raw, cumulative data')
axes[2].plot_date(sismo_daily['date'], sismo_daily['cumul'], 
                  '.', lw=2, label='daily cumulative data')
# axes[2].set_ylim([2400, 7800])
axes[2].set_title('Sismic data')
axes[2].set_ylabel('Sismic count')
for ax in axes: ax.legend()

fig.autofmt_xdate()
fig.tight_layout()


# The depth of the Tarissan lake is equivalent to the level below a reference point.  As the lowest depths (i.e. largest magnitude of the -vely signed data) correspond to the highest values of the GWL data.  Hence we take the -ve of the GWL signals in the following analyses

# In[37]:


## simplified variables for ease of manipulation
x0 = GWL_data['Date']
y0 = GWL_data['GWL']
x0 = x0[np.logical_and(x0 >= SP_data['Time'].min(), 
                       x0 <= SP_data['Time'].max())]
y0 = y0[x0.index]

x1 = SP_data_daily['Time']
y1 = SP_data_daily[f'{ps_label}1'].interpolate()
x1 = x1[np.logical_and(x1 >= SP_data['Time'].min(), 
                       x1 <= SP_data['Time'].max())]
y1 = y1[x1.index]

xs = sismo_daily['date']
ys = np.cumsum(sismo_daily['count'])
# ys = ys[np.logical_and(xs >= SP_data['Time'].min(), 
#                        xs <= SP_data['Time'].max())]
xs = xs[np.logical_and(xs >= SP_data['Time'].min(), 
                       xs <= SP_data['Time'].max())]
ys = ys[xs.index]

xr = rain_data['Date']
yr = rain_data['cumul'].interpolate()
# yr = yr[np.logical_and(xr >= SP_data['Time'].min(), 
#                        xr <= SP_data['Time'].max())]
xr = xr[np.logical_and(xr >= SP_data['Time'].min(), 
                       xr <= SP_data['Time'].max())]
yr = yr[xr.index]
# ys = detrend_normalise_data(ys)


# ## Test of correlation of the raw data

# In[13]:


# corr = scipy.signal.correlate(-y0, ys, 'same')
# lags = scipy.signal.correlation_lags(len(y0), len(ys), 'same')
# plt.plot(lags, corr, '-', label='cross-correlation GWL-sismo')

# auto_corr = scipy.signal.correlate(-y0, -y0, 'same')
# auto_lags = scipy.signal.correlation_lags(len(y0), len(y0), 'same')
# plt.plot(auto_lags, auto_corr, '-', label='auto-correlation GWL')

# auto_corr = scipy.signal.correlate(ys, ys, 'same')
# auto_lags = scipy.signal.correlation_lags(len(ys), len(ys), 'same')
# plt.plot(auto_lags, auto_corr, '-', label='auto-correlation sismo')

# plt.legend()
# plt.title('Correlation of raw data');


# The correlation functions are dominated by the long-period signals and, in particular, by the sismic data which is large in magnitude.  We will detrend the data to remove the long-period information.  We will also normalise the data (by its vector norm) to scale the correlation function to the interval [-1, 1].

# In[14]:


fig = plt.figure(figsize=(10, 6))

ax0 = fig.add_axes([.05, .75, .9, .3])
ax1 = fig.add_axes([.05, .40, .9, .3])
ax2 = fig.add_axes([.05, .08, .9, .3], sharex=ax1)


for observable, indep_var, linesty, label, ax in zip((yr, ys), (xr, xs), 
                                                     ('-', '--'), ('rain', 'sismo'), (ax1, ax2)):
    a  = detrend_normalise_data(observable)
    ax0.plot_date(indep_var, a, f'{linesty}k', lw=2, label=label)
    # ax0.plot_date(xs, # detrend_normalise_data(np.cumsum(sismo_daily['count'][xs.index])), 
                  # '--k', lw=3, label='cumul VT')
    for ind in range(1, 9):
        # y1 = SP_data_daily[f'{ps_label}{ind}']
        y1 = SP_data_daily[f'{ps_label}{ind}']
        y1 = y1[x1.index].interpolate('linear')
        
    
        b  = detrend_normalise_data(y1)

        if label == 'sismo':
            ax0.plot_date(x1, b, f'-C{ind-1}', label=f'SP{ind}')
    
        corr = scipy.signal.correlate(a, b, 'same')
        lags = scipy.signal.correlation_lags(len(a), len(b), 'same')
    
        ax.plot(lags, corr, f'-C{ind-1}', label=f'SP{ind}')
        
        # print(f"SP{ind} signal lags {label} signal by {lags[np.argmax(corr)]}")
        # if ind == 8:
        #     print()

ax0.legend(ncols=5)
ax1.legend(ncols=2, loc=2)

ax0.xaxis.set_major_locator(MonthLocator(interval=3))
ax0.xaxis.set_minor_locator(MonthLocator()) #DayLocator(15))
ax0.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
ax0.set_title('Detrended and normalised data')

ax1.set_ylabel(r'Correlation fn, rain -- SP')
ax1.xaxis.set_tick_params(labelbottom=False)
ax1.xaxis.set_minor_locator(MultipleLocator(20))
ax1.yaxis.set_major_locator(MultipleLocator(.5))
ax1.grid()

ax2.set_ylabel(r'Correlation fn, sismo -- SP')
ax2.set_xlabel('Lag/[days]')
ax2.yaxis.set_major_locator(MultipleLocator(.5))
ax2.grid()
# ax0.plot_date(x0, ((y0 - y0.mean()) / (y0.max() - y0.min())), '-.')

plt.show()
fig.savefig('../plots/timeseries_correlations.pdf', dpi=300)


# In[15]:


y0   = detrend_normalise_data(y0)
yr   = detrend_normalise_data(yr)
corr = scipy.signal.correlate(y0, yr, 'full')
lags = scipy.signal.correlation_lags(len(y0), len(yr), 'full')

plt.plot(lags, corr, '-')
plt.title('GWL-Rain correlation function');
plt.xlabel('Lags/[days]');
plt.ylabel('Correlation coefficient/[-]');
plt.grid();
print(lags[np.argmax(corr)])


# ## What are the lags for the all peaks and troughs of the GWL-SP correlations?

# In[16]:


a = detrend_normalise_data(-y0)
for ind in range(1, 9):
    y1 = SP_data_daily[f'{ps_label}{ind}']
    y1 = y1[x1.index].interpolate('linear')

    b  = detrend_normalise_data(y1)

    corr = scipy.signal.correlate(a, b, 'same')
    lags = scipy.signal.correlation_lags(len(a), len(b), 'same')

    pos_lags = lags[lags > 0]
    pos_corr = corr[lags > 0]

    peaks_troughs = scipy.signal.find_peaks(abs(corr), prominence=.04)[0]

    # print(f'SP{ind} : {pos_lags[np.argmin(pos_corr)]}')
    print(f'SP{ind} : {lags[peaks_troughs]}')

# sorted(np.append(lags[peaks_troughs], lags[scipy.signal.find_peaks(-corr, prominence=.05)[0]]))


# ### Table: lag of peaks/troughs seen in -GWL--SP correlation functions
# 
# Peaks and troughs found using the scipy.signal.find_peaks function, requiring a prominence of .04
# 
# |     |   1 |   2 |  3 |   4 |   5 |
# | ----|-----|-----|----|-----|-----|
# | SP1 |     | -18 |  4 |  32 |     |
# | SP2 | -41 | -20 |  4 |  27 |     |
# | SP3 |     | -23 |  3 |  32 |     |
# | SP4 |     | -29 |  3 |  32 |  45 |
# | SP5 |     | -23 |  3 |  33 |     |
# | SP6 |     | -17 |  5 |  35 |     |
# | SP7 | -43 |     |  2 |  32 |     |
# | SP8 |     | -27 |  3 |  32 |     |

# ## Correlation of GWL and sismic data

# In[17]:


a = detrend_normalise_data(y0)  # GWL
# a /= np.linalg.norm(a)
b = detrend_normalise_data(ys)  # sismo
# b /= np.linalg.norm(b)

corr = scipy.signal.correlate(a, b, 'same')
lags = scipy.signal.correlation_lags(len(a), len(b), 'same')

plt.plot(lags, corr, '-')
plt.xlabel('lag/[days]')
plt.ylabel('Correlation fn.')
plt.title('Detrended and normalised GWL - sismo correlation fn');
# print(f'{lags[np.argmax(corr)]}')

peaks_troughs = scipy.signal.find_peaks(abs(corr), prominence=.1)[0]
plt.plot(lags[peaks_troughs], corr[peaks_troughs], 'ro')
print(f"Peaks and troughs at: {lags[peaks_troughs]} days, with max at {lags[np.argmax(corr)]} days")


# In[66]:


from astropy.timeseries import LombScargle
from scipy.fft import fft, ifft, fftshift, ifftshift, fftfreq

y_ = detrend_normalise_data(y0)
x_ = (x0 - x0.iloc[0]).dt.total_seconds().values
dx = x_[1] - x_[0]
fs = 1 / dx
period = x_[-1] - x_[0]
fp = 1 / period
ls = LombScargle(x_, y_)
freq, power = ls.autopower(minimum_frequency=fp, 
                           maximum_frequency=fs/2)

f_p, p_p = scipy.signal.periodogram(y_, fs, window='tukey')
f_w, p_w = scipy.signal.welch(y_, fs, window='tukey')

days = [2, 4, 8, 16, 32]

for day in days:
    f_day = 1 / (day * 24 * 3600)
    plt.axvline(f_day, linestyle='-', c='k', linewidth=.8, zorder=1);
    plt.annotate(f'{day:2d} days', xy=(.8*f_day, 1e-5), xycoords='data', 
                 rotation=90, zorder=9, color='k', fontsize='small',
                 path_effects=[pe.withStroke(linewidth=1.5, foreground='w')])
                 # bbox=dict(fc='w', ec='none', pad=.01, 
                 #                                  alpha=.7, boxstyle='round,pad=.2'))

plt.loglog(freq, power / power[0], '-', zorder=3, label='Lomb-Scargle')
plt.loglog(f_p, p_p / p_p[1], '-', zorder=3, label="Periodogram")
plt.loglog(f_w, p_w / p_w[1], '-', zorder=3, label="Welch")

# Do it the hard way - by FFT!
y_fft = fftshift(fft(y_))
freqs = fftshift(fftfreq(y0.size, d=dx))
# print(freqs, y_fft * np.conj(y_fft))

y_fft2 = y_fft[freqs >= 0]
freqs2 = freqs[freqs >= 0]
psd    = y_fft * np.conj(y_fft)
psd2   = y_fft2 * np.conj(y_fft2)
plt.loglog(freqs2, psd2 / psd2[1], # (psd2 - psd2.min()) / (psd2.max() - psd2.min())
           '-', label='FFT')

plt.legend()
plt.xlabel('Frequency/[Hz]');
plt.ylabel('PSD/[au]');
plt.title('PSD of GWL data');
plt.ylim((11e-7, 11));


# In[34]:


1 / (365.25 / 12 * secs_in_day)


# In[19]:


((x0 - x0.iloc[0]).dt.total_seconds().values // 86400).astype(int)


# Regardless of the method used to calculate them, the PSDs are essentially identical up to a multiplicative factor

# ## Calculate the correlation fn from the PSDs -- to be completed

# In[20]:


y_corr = ifftshift(ifft(ifftshift(psd)))
plt.plot(lags, y_corr, label='Via FFT')
a_corr = scipy.signal.correlate(y_, y_, 'same')
plt.plot(lags, a_corr, label='Direct')
plt.legend()
plt.xlabel('Lags/[days]');
plt.ylabel('Correlation coefficient');
plt.title('Autocorrelation of GWL')
plt.grid()


# In[21]:


def correlation_matrix_plot(df, column_names):
    '''Plot the (cross- and auto-) correlations for n datasets in the form of a n-by-n matrix of plots'''

    ndata = len(column_names)
    fig, ax = plt.subplots(nrows=ndata, ncols=ndata, sharex=True, sharey=True, figsize=(10, 10))
    
    for indy in range(ndata):
        series_a = detrend_normalise_data(df[column_names[indy]].interpolate())
        for indx in range(indy, ndata):
            series_b = detrend_normalise_data(df[column_names[indx]].interpolate())
            corr = scipy.signal.correlate(series_a, series_b, 'same')
            lags = scipy.signal.correlation_lags(len(series_a), len(series_b), 'same')
            ax[indy, indx].plot(lags, corr, '-k')
            ax[indy, indx].grid()

            ax[indy, indx].plot(lags[np.argmax(abs(corr))], corr[np.argmax(abs(corr))], '.r')
    return fig, ax


column_names = SP_data_daily.columns[SP_data_daily.columns.str.contains('((p|P)(s|S))', regex=True)]
fig, ax = correlation_matrix_plot(SP_data_daily, column_names)

for ind in range(1, len(column_names)+1):
    ax[0, ind-1].set_title(f'SP{ind}')
    ax[ind-1, 0].set_ylabel(f'SP{ind}')


# ## Moving window correlations
# 
# Use 3 month (?) window of data for the following periods:
# - May -- July 2022 (2022-05-01 -- 2022-07-31)
# - Aug -- Oct 2022 (2022-08-01 -- 2022-10-31)
# - Nov 2022 -- Jan 2023 (2022-11-01 -- 2023-01-31)
# - Feb -- Apr 2023 (2023-02-01 -- 2023-04-30)

# In[22]:


# Correlate GWL with one SP signal per period
fig, axes = plt.subplots(ncols=4, figsize=(8, 5), nrows=2, sharex=True, sharey=True)
for ind, ax in enumerate(axes.ravel()):
    year  = 2022
    month = 5
    d_end = DT(2022, 1, 1)
    while d_end < DT(2024, 2, 1):
        d_start = DT(year, month, 1)
        month += 3
        if month > 12: 
            month = month % 12
            year += 1
        d_end   = DT.strptime(last_day_previous_month(DT(year, month, 1).strftime('%F')), 
                              '%Y-%m-%d')
    
        x0, y0 = GWL_data.query('Date >= @d_start and Date <= @d_end')[['Date', 'GWL']].values.T
        xs, ys = sismo_daily.query('date >= @d_start and date <= @d_end')[['date', 'cumul']].values.T
        datasp = SP_data_daily.interpolate('linear').query('Time >= @d_start and Time <= @d_end')
    
        y1 = datasp[f'{ps_label}{ind+1}'].interpolate().values
        x1 = datasp['Time'].values
    
        y0 = detrend_normalise_data(y0)
        ys = detrend_normalise_data(ys)
        y1 = detrend_normalise_data(y1)
        corr = scipy.signal.correlate(-y0, y1, 'same')
        lags = scipy.signal.correlation_lags(len(y0), len(y1), 'same')
    
        ax.plot(lags, corr, '-', label=d_start.strftime('%F') + '--' + d_end.strftime('%F'))

    if ind == 0: fig.legend(ncols=4, fontsize=8, loc='outside upper center')
    ax.grid()
    ax.set_ylim([-1.1, 1.1]);
    ax.annotate(f'GWL--SP{ind+1}', xy=(10, -.9), 
                fontsize=8, bbox=dict(fc="white", ec="none", alpha=.8));

fig.supxlabel('Lags/[days]');
fig.tight_layout()
fig.savefig('../plots/xcorr_GWL-oneSP_per_period.pdf', dpi=300)


# In[23]:


# Correlate GWL with each SP signal per period
fig, axes = plt.subplots(ncols=4, figsize=(8, 5), nrows=2, sharex=True, sharey=True)
year  = 2022
month = 5
for ind, ax in enumerate(axes.ravel()):
    d_start = DT(year, month, 1)
    month += 3
    if month > 12: 
        month = month % 12
        year += 1
    d_end   = DT.strptime(last_day_previous_month(DT(year, month, 1).strftime('%F')), 
                          '%Y-%m-%d')

    x0, y0 = GWL_data.query('Date >= @d_start and Date <= @d_end')[['Date', 'GWL']].values.T
    a = detrend_normalise_data(y0) #GWL data
    for spind in range(1, 9):
        datasp = SP_data_daily.query('Time >= @d_start and Time <= @d_end')[['Time', f'{ps_label}{spind}']]
        y1 = datasp[f'{ps_label}{spind}'].interpolate().values
        x1 = datasp['Time'].values    
        b  = detrend_normalise_data(y1) # SP data
        
        corr = scipy.signal.correlate(-a, b, 'same')
        lags = scipy.signal.correlation_lags(len(a), len(b), 'same')
    
        ax.plot(lags, corr, f'-C{spind-1}', label=f'SP{spind}')

    if ind == 0: 
        fig.legend(ncols=8, fontsize=8, loc='upper center', framealpha=.9)
    ax.grid()
    ax.annotate(d_start.strftime('%F') + '--' + d_end.strftime('%F'), xy=(-32, -.8), 
                fontsize=8, bbox=dict(fc="white", ec="none", alpha=.8));
    
# ax.set_ylim([-.8, .8]);
ax.yaxis.set_major_locator(MultipleLocator(.5))
ax.xaxis.set_major_locator(MultipleLocator(25))

fig.supxlabel('Lags/[days]');
fig.supylabel('Correlation coefficient, GWL-SP/[-]');
fig.tight_layout()
fig.savefig('../plots/xcorr_GWL-eachSP_per_period.pdf', dpi=300)


# In[24]:


# Correlate sismo with each SP signal per period
fig, axes = plt.subplots(ncols=4, figsize=(8, 5), nrows=2, sharex=True, sharey=True)
year  = 2022
month = 5
for ind, ax in enumerate(axes.ravel()):
    d_start = DT(year, month, 1)
    month += 3
    if month > 12: 
        month = month % 12
        year += 1
    d_end   = DT.strptime(last_day_previous_month(DT(year, month, 1).strftime('%F')), 
                          '%Y-%m-%d')

    xs, ys = sismo_daily.query('date >= @d_start and date <= @d_end')[['date', 'cumul']].values.T
    a = detrend_normalise_data(ys) #GWL data
    for spind in range(1, 9):
        datasp = SP_data_daily.query('Time >= @d_start and Time <= @d_end')[['Time', f'{ps_label}{spind}']]
        y1 = datasp[f'{ps_label}{spind}'].interpolate().values
        x1 = datasp['Time'].values    
        b  = detrend_normalise_data(y1) # SP data
        
        corr = scipy.signal.correlate(a, b, 'same')
        lags = scipy.signal.correlation_lags(len(a), len(b), 'same')
    
        ax.plot(lags, corr, f'-C{spind-1}', label=f'SP{spind}')

    if ind == 0: fig.legend(ncols=8, fontsize=8, loc='upper center', framealpha=.9)
    ax.grid()
    # ax.set_ylim([-1.1, 1.1]);
    ax.annotate(d_start.strftime('%F') + '--' + d_end.strftime('%F'), xy=(-32, -.65), 
                fontsize=8, bbox=dict(fc="white", ec="none", alpha=.8));
ax.yaxis.set_major_locator(MultipleLocator(.5))
ax.xaxis.set_major_locator(MultipleLocator(25))

fig.supxlabel('Lags/[days]');
fig.supylabel('Correlation coefficient, sismo-SP/[-]');
fig.tight_layout()
fig.savefig('../plots/xcorr_GWL-eachSP_per_period.pdf', dpi=300)


# In[25]:


# Correlate rain station data with each SP signal per period
fig, axes = plt.subplots(ncols=4, figsize=(8, 5), nrows=2, sharex=True, sharey=True)
year  = 2022
month = 5

start_dates, end_dates, SPs, max_cs, lag_max_cs = [], [], [], [], []
header = f'{"Period":>22}  SP   max_c  lag'
print(header, '-'*len(header), sep='\n')
for ind, ax in enumerate(axes.ravel()):
    d_start = DT(year, month, 1)
    month += 3
    if month > 12: 
        month = month % 12
        year += 1
    d_end = DT.strptime(last_day_previous_month(DT(year, month, 1).strftime('%F')), 
                        '%Y-%m-%d')

    if d_end < DT(2024, 5, 1):
        xr, yr = rain_data.query('Date >= @d_start and Date <= @d_end')[['Date', 'rain']].values.T
        a = detrend_normalise_data(yr)
        for spind in range(1, 9):
            datasp = SP_data_daily.query('Time >= @d_start and Time <= @d_end')[['Time', f'{ps_label}{spind}']]
            y1 = datasp[f'{ps_label}{spind}'].interpolate().values
            x1 = datasp['Time'].values    
            b  = detrend_normalise_data(y1) # SP data
            
            corr = scipy.signal.correlate(a, b, 'same')
            lags = scipy.signal.correlation_lags(len(a), len(b), 'same')

            corr_max_ind = np.argmax(abs(corr))
            fstring = f' {spind}  {corr[corr_max_ind]:>6.3f}  {lags[corr_max_ind]:>3d}'
            if spind == 1:
                print(d_start.strftime('%F') + '--' + d_end.strftime('%F') + '  ' + fstring)
            else:
                print(' '*24 + fstring)
        
            ax.plot(lags, corr, f'-C{spind-1}', label=f'SP{spind}')

            start_dates.append(d_start)
            end_dates.append(d_end)
            SPs.append(spind)
            max_cs.append(corr[corr_max_ind])
            lag_max_cs.append(lags[corr_max_ind])
    
        if ind == 0: fig.legend(ncols=8, fontsize=8, loc='upper center', framealpha=.9)
        ax.grid()
        # ax.set_ylim([-1.1, 1.1]);
        ax.annotate(d_start.strftime('%F') + '--' + d_end.strftime('%F'), xy=(-32, -.7), 
                    fontsize="x-small", bbox=dict(fc="white", ec="none", alpha=.8));
        

ax.xaxis.set_major_locator(MultipleLocator(25))
ax.yaxis.set_major_locator(MultipleLocator(.5))
ax.yaxis.set_minor_locator(MultipleLocator(.25))
fig.supxlabel('Lags/[days]');
fig.supylabel('Correlation coefficient, rain-SP/[-]');
fig.tight_layout()
fig.savefig('../plots/xcorr_rain-eachSP_per_period.pdf', dpi=300)

rain_ps_correlations = pd.DataFrame.from_records(
    np.array([start_dates, end_dates, SPs, max_cs, lag_max_cs]).T,
    columns=('start', 'end', 'SP', 'max_correlation', 'lag_max_correlation'))


# In[26]:


rain_ps_correlations.to_csv('rain-SP-correlations-by-period.csv', index=False)
print(rain_ps_correlations.to_markdown(index=False))


# In[ ]:




