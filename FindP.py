from astropy.stats import LombScargle
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import glob
import pandas as pd
import scipy.signal as sig
from astropy.convolution import convolve, Box1DKernel

dir = '/Volumes/Zoe Bell Backup/everest/c08/229200000/28610/'
file = 'hlsp_everest_k2_llc_229228610-c08_kepler_v2.0_lc.fits'
fileLis = glob.glob('/Volumes/Zoe Bell Backup/everest/c08/229200000/*/*.fits')
goodFile = '/Volumes/Zoe Bell Backup/everest/c08/229200000/28988/hlsp_everest_k2_llc_229228988-c08_kepler_v2.0_lc.fits'
badFile = '/Volumes/Zoe Bell Backup/everest/c08/229200000/28995/hlsp_everest_k2_llc_229228995-c08_kepler_v2.0_lc.fits'
#badFile = '/Volumes/Zoe Bell Backup/everest/c08/229200000/28967/hlsp_everest_k2_llc_229228967-c08_kepler_v2.0_lc.fits'
okayFile = '/Volumes/Zoe Bell Backup/everest/c08/229200000/28386/hlsp_everest_k2_llc_229228386-c08_kepler_v2.0_lc.fits'

def findMultipleP(files, name, gen_plots=False, gen_file_type='.png', gen_min_period = 0.1, gen_max_period = 30):
    '''
    Takes a list of file names for .fits files from the Everest K2 data, and optionally
    whether you want plots saved, the file type you want them saved as, and the minimum and maximum periods you want to look for.
    Returns the output of findP for each file in a list and saves them in a .csv file.
    '''
    lis = []
    for file_name in files:
        lis.append(findP(file_name, plots=gen_plots, file_type=gen_file_type, min_period=gen_min_period, max_period=gen_max_period))
    output = pd.DataFrame(data=lis, columns=['File Name', 'Best Period','Max Power','False Alarm Prob'])
    output.to_csv('/Volumes/Zoe Bell Backup/FindPOutput/' + name + '.csv')
    return output

def findP(file_name, plots=False, file_type='.png', min_period = 0.1, max_period = 30):
    '''
    Takes the file name of a .fits file from the Everest K2 data, 
    and optionally whether you want a plot saved, the file type you want it saved as, and the minimum and maximum periods you want to look for.
    Returns the file name, period that best fits the corrected flux data in that range, the power at that period, and the associated false alarm probability.
    '''
    start = file_name.rfind('/') + 1
    name = file_name[start:-5]

    data = Table.read(file_name, format='fits')
    ok = np.where((data['QUALITY']==0) & (np.isfinite(data['TIME'])) & (np.isfinite(data['FCOR']) & (np.isfinite(data['FRAW_ERR']))))

    t = np.array(data['TIME'][ok])
    fcor = np.array(data['FCOR'][ok])
    frawErr = np.array(data['FRAW_ERR'][ok])

    ls = LombScargle(t, fcor, frawErr)
    freq, power = ls.autopower(minimum_frequency=1/max_period, maximum_frequency=1/min_period)
    best_freq = freq[np.argmax(power)]
    max_power = np.max(power)
    
    if(plots):
        plt.figure(figsize=(10,7))

        plt.subplot(211)
        plt.plot(1/freq, power)
        plt.title('Periodogram')
        plt.xlabel('Period')
        plt.ylabel('Power')
        plt.annotate('best period', xy=(1/best_freq, max_power), xytext=(1/best_freq*0.5, max_power*0.9), 
                    arrowprops=dict(facecolor='black', width=1, headwidth=5))

        plt.subplot(212)
        t_fit = np.linspace(np.min(t),np.max(t)) # make just select first and last
        f_fit = LombScargle(t, fcor, frawErr).model(t_fit, best_freq)
        plt.plot(t, fcor)
        plt.plot(t_fit, f_fit)
        plt.title('Comparison of Data and Model')
        plt.xlabel('Time')
        plt.ylabel('Flux')

        plt.suptitle(name)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig('PlotOutputs/' + name + file_type, dpi=150)
        plt.close()

    return [name, 1/best_freq, max_power, ls.false_alarm_probability(max_power)]

def ACFfindMultipleP(files, name, gen_plots=False, gen_file_type='.png', gen_min_period = 0.1, gen_max_period = 30):
    '''Similar to findMultipleP but uses autocorrelation instead of Lomb-Scargle.''' # doesn't produce data file yet
    lis = []
    for file_name in files:
        lis.append(ACFfindP(file_name, plots=gen_plots, file_type=gen_file_type, min_period=gen_min_period, max_period=gen_max_period))
    output = pd.DataFrame(data=lis, columns=['File Name', 'Best Period','Max Autocorrelation'])
    #output.to_csv('/Volumes/Zoe Bell Backup/ACFfindPOutput/' + name + '.csv')
    return output

def ACFfindP(file_name, plots=False, file_type='.png', min_period = 0.1, max_period = 30):
    '''Similar to findP but uses autocorrelation instead of Lomb-Scargle.'''
    start = file_name.rfind('/') + 1
    name = file_name[start:-5]

    data = Table.read(file_name, format='fits')
    ok = np.where((data['QUALITY']==0) & (np.isfinite(data['TIME'])) & (np.isfinite(data['FCOR']) & (np.isfinite(data['FRAW_ERR']))))

    t = np.array(data['TIME'][ok])
    fcor = list(data['FCOR'][ok])
    fcor_median = fcor-np.median(fcor)

    N = len(fcor)
    t_step = np.nanmedian(t[1:]-t[0:-1])
    t_range = np.arange(N)*t_step

    arr = sig.correlate(fcor_median,fcor_median, mode='full')
    ACF = arr[N-1:]

    okP = np.where(((t_range>min_period) & (t_range<max_period)))
    t_search = t_range[okP]
    ACF_search = ACF[okP]

    #box_width = findWidthAtHalfMax(t_search, ACF_search)
    #print(box_width)

    #ACF_search = convolve(ACF_search, Box1DKernel(box_width*2))
    ACF_search = convolve(ACF_search, Box1DKernel(100)) # 200 good

    #peaks = sig.argrelmax(np.array([t_search, ACF_search]))
    #peaks = sig.find_peaks(ACF_search, threshold=np.max(ACF_search)/100)
    #peaks = sig.find_peaks(ACF_search, prominence=100000) #is this always going to be a good prominence threshold? (nope)
    #peaks = sig.find_peaks(ACF_search, prominence=np.max(ACF_search)/10)
    peaks = sig.find_peaks(ACF_search, prominence=np.max(ACF_search)/100)
    first_peak = peaks[0]
    best_period = t_search[first_peak]
    max_ACF = ACF[first_peak]

    if(plots):
        plt.figure(figsize=(10,7))

        plt.subplot(211)
        plt.title('Unsmoothed')
        plt.xlabel('Time Shift')
        plt.ylabel('Autocorrelation')
        plt.plot(t_range, ACF)

        plt.subplot(212)
        plt.title('Smoothed')
        plt.xlabel('Time Shift')
        plt.ylabel('Autocorrelation')
        plt.plot(t_search, ACF_search)

        formated_periods = []
        for n in best_period[:10]:
            formated_periods.append(format(n, '.2f'))

        plt.suptitle(formated_periods)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig('ACFPlotOutputs/' + name + file_type, dpi=150)
        plt.close()

    return [name, best_period, max_ACF]

def findWidthAtHalfMax(t_search, ACF_search):
    max = ACF_search[0]
    index = 0
    for i in range(len(ACF_search)):
        if ACF_search[i] < 0.5*max:
            index = i
            break
    return t_search[index]

# write func that calcs derivs and finds zeros and neg curvature
# boxcar smooth (running average) for gross ones
# use full width at half max for box size
# look at 2013 paper for how to use ACF (use same limits)
# save plot files

# N = npsize(fcor)
# ACF = arr[N:]
# np.nanmedian(t[1:]-t[0:-1])
# tan = np.arange(N)*av. time step

# put stuff to GitHub for prob problem
# autocorrelation func (ACF) in astropy? scipy.correlate2d or something (looking for first local max > 0.2 days),
# return period and height of ACF output

# pandas to output file (to_csv)
# output=pd.DataFrame(lis) # name cols
# output.to_csv()

# add label to periodogram at peak, name and lable /... .fitz/png, make two-panel figure (subplot)
# l = kdjf('/)
# name[l[-1]+1:-4]+'.png' (make ftype a variable)
# plt.figure(figsize(optional,takes x and y inches)) --stuff-- plt.savefig(name) plt.close()

# panda.rolling rolling median; import glob.glob as glob to read in multiple files
# create function(file name) return period and uncertainty/power, optional args plots=False, smooth window=10, period range
# push to GitHub
# plot np.arange(0.001, 0.1, 0.001)




'''
data = Table.read(dir + file, format='fits')
ok = np.where((data['QUALITY']==0) & (np.isfinite(data['TIME'])) & (np.isfinite(data['FLUX']) & (np.isfinite(data['FRAW_ERR']))))


plt.figure()
plt.plot(data['TIME'][ok], data['FLUX'][ok])
plt.show()


t = np.array(data['TIME'][ok])
flux = np.array(data['FLUX'][ok])
frawErr = np.array(data['FRAW_ERR'][ok])
freq, power = LombScargle(t, flux, frawErr).autopower(minimum_frequency=1/30, maximum_frequency=1/0.1)
plt.plot(1/freq, power)
plt.xscale('log')
plt.show()

bestFreq = freq[np.argmax(power)]
t_fit = np.linspace(2555,2640)
y_fit = LombScargle(t, flux, frawErr).model(t_fit, bestFreq)
plt.plot(t_fit, y_fit)
plt.plot(t, flux)
plt.show()

ls = LombScargle(t, flux, frawErr)
freq, power = ls.autopower(minimum_frequency=1/30, maximum_frequency=1/0.1)
ls.false_alarm_probability(np.max(power))



t = np.array(data['TIME'][ok])
fraw = np.array(data['FRAW'][ok])
frawErr = np.array(data['FRAW_ERR'][ok])
freq, power = LombScargle(t, fraw, frawErr).autopower()
plt.plot(freq, power)
plt.show()

bestFreq = freq[np.argmax(power)]
t_fit = np.linspace(2555,2640)
y_fit = LombScargle(t, fraw, frawErr).model(t_fit, bestFreq)
plt.plot(t_fit, y_fit)
plt.plot(t, fraw)
plt.show()

ls = LombScargle(t, fraw, frawErr)
freq, power = ls.autopower()
ls.false_alarm_probability(power.max())
'''