#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

from matplotlib.dates import DateFormatter, MonthLocator, DayLocator
from matplotlib.ticker import MultipleLocator

def sinusoid(x, offset, amplitude, frequency, phase):
    return offset + amplitude * np.sin(2 * np.pi * frequency * x - phase)
    


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
axes[1].set_xlabel('Lag')
axes[1].set_ylabel('Correlation')
    
print(f"Signal y1 lags signal y0 by {-lags[np.argmax(corr)]}")
axes[1].legend(loc=1)
fig.tight_layout()
plt.show()


# ## Real data

# In[3]:


GWL_data = pd.read_csv('../data/GWL_modelling/GWL_TAS_model_inversion.csv', 
                       sep=',', parse_dates=['# Date (yyyy-mm-dd)'])
GWL_data = GWL_data.rename(mapper={'# Date (yyyy-mm-dd)': 'Date', 
                                   ' GWL (m) ': 'GWL', 
                                   ' error (m)': 'error'}, axis=1)
print(GWL_data.head())


SP_data  = pd.read_csv('../data/PSGUADA_NOEUD_263_PS-data-as-joinbyfield-2024-05-06-09-51-08.csv', 
                       sep=',', parse_dates=['Time'])
print(SP_data.head())

SP_data_daily = SP_data.copy()
SP_data_daily.index = SP_data['Time']
SP_data_daily = SP_data_daily.loc[:, SP_data_daily.columns.str.contains('Average')]
SP_data_daily = SP_data_daily.resample('1D').mean()
SP_data_daily = SP_data_daily.reset_index()


# In[4]:


fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(6, 8))

axes[0].plot_date(GWL_data['Date'], GWL_data['GWL'], '.', zorder=3)
axes[0].errorbar(GWL_data['Date'], GWL_data['GWL'], yerr=GWL_data['error'], 
                 marker='.', c='k', lw=.8, zorder=1)

axes[1].plot_date(SP_data['Time'], 
                  SP_data['Average json.data-tension-ps1'], '.', zorder=3)
axes[1].plot_date(SP_data_daily['Time'],
                  SP_data_daily['Average json.data-tension-ps1'], '-.', lw=4, zorder=3)

axes[1].set_xlim([SP_data['Time'].min(), SP_data['Time'].max()])
axes[0].set_ylim([65, 85])

fig.autofmt_xdate()
fig.tight_layout()


# In[5]:


def detrend_normalise_data(data):
    d = scipy.signal.detrend(data)
    return d / np.linalg.norm(d)


# In[6]:


x0 = GWL_data['Date']
y0 = GWL_data['GWL']
x0 = x0[np.logical_and(x0 >= SP_data['Time'].min(), 
                       x0 <= SP_data['Time'].max())]
y0 = y0[x0.index]

x1 = SP_data_daily['Time']
x1 = x1[np.logical_and(x1 > SP_data['Time'].min(), 
                       x1 <= SP_data['Time'].max())]

sismo = pd.read_csv('/home/david/FieldWork/Soufriere-Guadeloupe/AKlein/data/WO_OVSG_MC3_OVSG_dump_bulletin.csv', 
                    sep=';', skiprows=2, names=['date', 'count', 'duration', 'amplitude', 'magnitude', 'e(j)', 'longitude', 'latitude', 'depth',
                                                'type', 'file', 'locmode', 'loctype', 'projection', 'operator', 'timestamp', 'id'], 
                    parse_dates=['date'], date_format='%Y%m%d %H%M%S.%f')
sismo = sismo[['date', 'count', 'duration', 'amplitude', 'longitude', 'latitude', 'depth']]

sismo_daily = sismo.copy()
sismo_daily.index = sismo_daily['date']

sismo_daily = sismo_daily.loc[:, ~sismo_daily.columns.str.contains('date')]
sismo_daily = sismo_daily.resample('1D').sum().reset_index()

xs = sismo_daily['date']
ys = np.cumsum(sismo_daily['count'])
ys = ys[np.logical_and(xs >= SP_data['Time'].min(), 
                       xs <= SP_data['Time'].max())]
xs = xs[np.logical_and(xs >= SP_data['Time'].min(), 
                       xs <= SP_data['Time'].max())]
# ys = detrend_normalise_data(ys)


# In[7]:


# [GWL_data['Date'] > SP_data['Time'].min() & GWL_data['Date'] <= SP_data['Time'].max()]
fig = plt.figure(figsize=(10, 9))
ax0 = fig.add_axes([.05, .75, .9, .3])
ax1 = fig.add_axes([.05, .41, .9, .3])
ax2 = fig.add_axes([.05, .10, .9, .3], sharex=ax1)


for observable, indep_var, linesty, label, ax in zip((y0, ys), (x0, xs), ('-', '--'), ('GWL', 'sismo'), (ax1, ax2)):
    a  = detrend_normalise_data(observable)
    ax0.plot_date(indep_var, a, f'{linesty}k', lw=3, label=label)
    # ax0.plot_date(xs, # detrend_normalise_data(np.cumsum(sismo_daily['count'][xs.index])), 
                  # '--k', lw=3, label='cumul VT')
    for ind in range(1, 9):
        y1 = SP_data_daily[f'Average json.data-tension-ps{ind}']
        y1 = y1[x1.index].interpolate('linear')
    
        b  = detrend_normalise_data(y1)

        if label == 'sismo':
            ax0.plot_date(x1, b, f'-C{ind-1}', label=f'SP{ind}')
    
        corr = scipy.signal.correlate(a, b, 'same')
        lags = scipy.signal.correlation_lags(len(a), len(b), 'same')
    
        ax.plot(lags, corr, f'-C{ind-1}', label=f'SP{ind}')
        
        print(f"SP{ind} signal lags {label} signal by {-lags[np.argmax(corr)]}")
        if ind == 8:
            print()

ax0.legend(ncols=5)
ax1.legend(ncols=4, loc=1)

ax0.xaxis.set_major_locator(MonthLocator())
ax0.xaxis.set_minor_locator(DayLocator(15))
ax0.xaxis.set_major_formatter(DateFormatter('%Y-%m'))
ax0.set_title('Detrended and normalised data')

ax1.set_ylabel('Correlation fn, GWL -- SP')
ax1.xaxis.set_tick_params(labelbottom=False)
ax1.xaxis.set_minor_locator(MultipleLocator(5))

ax2.set_ylabel('Correlation fn, sismo -- SP')
ax2.set_xlabel('Lag/[days]')

plt.show()


# In[ ]:




