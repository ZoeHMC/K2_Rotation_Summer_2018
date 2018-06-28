from astropy.stats import LombScargle
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import glob
import pandas as pd

dir = '/Volumes/Zoe Bell Backup/everest/c08/229200000/28610/'
file = 'hlsp_everest_k2_llc_229228610-c08_kepler_v2.0_lc.fits'
fileLis = glob.glob('/Volumes/Zoe Bell Backup/everest/c08/229200000/*/*.fits')
goodFile = '/Volumes/Zoe Bell Backup/everest/c08/229200000/28988/hlsp_everest_k2_llc_229228988-c08_kepler_v2.0_lc.fits'
badFile = '/Volumes/Zoe Bell Backup/everest/c08/229200000/28967/hlsp_everest_k2_llc_229228967-c08_kepler_v2.0_lc.fits'

def findMultipleP(files, name, gen_Plots=False, gen_file_type='.png', gen_min_period = 0.1, gen_max_period = 30):
    '''
    Takes a list of file names for .fits files from the Everest K2 data, and optionally
    whether you want plots saved, the file type you want them saved as, and the minimum and maximum periods you want to look for.
    Returns the output of findP for each file in a list.
    '''
    lis = []
    for file_name in files:
        lis.append(findP(file_name, plots=gen_Plots, file_type=gen_file_type, min_period=gen_min_period, max_period=gen_max_period))
    output = pd.DataFrame(data=lis, columns=['File Name', 'Best Period','Max Power','False Alarm Prob'])
    output.to_csv('/Volumes/Zoe Bell Backup/FindPOutput/' + name + '.csv')
    #return output

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