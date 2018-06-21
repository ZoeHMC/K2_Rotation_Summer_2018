from astropy.stats import LombScargle
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import glob

dir = '/Volumes/Zoe Bell Backup/everest/c08/229200000/28610/'
file = 'hlsp_everest_k2_llc_229228610-c08_kepler_v2.0_lc.fits'
fileLis = glob.glob('/Volumes/Zoe Bell Backup/everest/c08/229200000/*/*.fits')
goodFile = '/Volumes/Zoe Bell Backup/everest/c08/229200000/28988/hlsp_everest_k2_llc_229228988-c08_kepler_v2.0_lc.fits'
badFile = '/Volumes/Zoe Bell Backup/everest/c08/229200000/28967/hlsp_everest_k2_llc_229228967-c08_kepler_v2.0_lc.fits'

def findMultipleP(files, gen_min_period = 0.1, gen_max_period = 30):
    '''
    Takes a list of file names for .fits files from the Everest K2 data, 
    and optionally the minimum and maximum periods you want to look for.
    Returns the output of findP for each file in a list.
    '''
    lis = []
    for file_name in files:
        lis.append(findP(file_name, min_period=gen_min_period, max_period=gen_max_period))
    return lis

def findP(file_name, plots=False, min_period = 0.1, max_period = 30):
    '''
    Takes the file name of a .fits file from the Everest K2 data, 
    and optionally whether you want plots printed and the minimum and maximum periods you want to look for.
    Returns the period that best fits the corrected flux data in that range and the associated false alarm probability.
    '''
    data = Table.read(file_name, format='fits')
    ok = np.where((data['QUALITY']==0) & (np.isfinite(data['TIME'])) & (np.isfinite(data['FCOR']) & (np.isfinite(data['FRAW_ERR']))))

    t = np.array(data['TIME'][ok])
    fcor = np.array(data['FCOR'][ok])
    frawErr = np.array(data['FRAW_ERR'][ok])

    ls = LombScargle(t, fcor, frawErr)
    freq, power = ls.autopower(minimum_frequency=1/max_period, maximum_frequency=1/min_period)
    if(plots):
        plt.plot(1/freq, power)
        #plt.xscale('log')
        plt.title('Periodogram')
        plt.xlabel('Period')
        plt.ylabel('Power')
        plt.show()
    
    best_freq = freq[np.argmax(power)]
    if(plots):
        t_fit = np.linspace(np.min(t),np.max(t)) # make just select first and last
        f_fit = LombScargle(t, fcor, frawErr).model(t_fit, best_freq)
        plt.plot(t, fcor)
        plt.plot(t_fit, f_fit)
        plt.title('Comparison of Data and Model')
        plt.xlabel('Time')
        plt.ylabel('Flux')
        plt.show()

    return [1/best_freq, ls.false_alarm_probability(np.max(power))]




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