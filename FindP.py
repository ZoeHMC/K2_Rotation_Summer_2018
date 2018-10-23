from astropy.stats import LombScargle
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
import glob
import pandas as pd
import scipy.signal as sig
import scipy.stats as stats
from astropy.convolution import convolve, Box1DKernel, Gaussian1DKernel
import time

dir = '/Volumes/Zoe Bell Backup/everest/c08/229200000/28610/'
file = 'hlsp_everest_k2_llc_229228610-c08_kepler_v2.0_lc.fits'
fileLis = glob.glob('/Volumes/Zoe Bell Backup/everest/c08/229200000/*/*.fits')
goodFile = '/Volumes/Zoe Bell Backup/everest/c08/229200000/28988/hlsp_everest_k2_llc_229228988-c08_kepler_v2.0_lc.fits'
badFile = '/Volumes/Zoe Bell Backup/everest/c08/229200000/28995/hlsp_everest_k2_llc_229228995-c08_kepler_v2.0_lc.fits'
#badFile = '/Volumes/Zoe Bell Backup/everest/c08/229200000/28967/hlsp_everest_k2_llc_229228967-c08_kepler_v2.0_lc.fits'
okayFile = '/Volumes/Zoe Bell Backup/everest/c08/229200000/28386/hlsp_everest_k2_llc_229228386-c08_kepler_v2.0_lc.fits'
testFile = '/Volumes/Zoe Bell Backup/everest/c08/229200000/28407/hlsp_everest_k2_llc_229228407-c08_kepler_v2.0_lc.fits'
weirdFile = '/Volumes/Zoe Bell Backup/everest/c08/229200000/28726/hlsp_everest_k2_llc_229228726-c08_kepler_v2.0_lc.fits' #maxes weren't being found by my func (now are)

campaign0files = glob.glob('/Volumes/Zoe Bell Backup/everest/c00/*/*/*.fits')
campaign5files = glob.glob('/Volumes/Zoe Bell Backup/everest/c05/*/*/*.fits')
campaign7files = glob.glob('/Volumes/Zoe Bell Backup/everest/c07/*/*/*.fits')
campaign8files = glob.glob('/Volumes/Zoe Bell Backup/everest/c08/*/*/*.fits')
campaign13files = glob.glob('/Volumes/Zoe Bell Backup/everest/c13/*/*/*.fits')
 
def mergeFiles(campaign_output_file_name, root_dir = '/Volumes/Zoe Bell Backup/'): # implicit assumption using nearest instance of non-unique row
    '''
    Takes the file name (including extension) of an output of bothFindP and optionally the root directory where this file and the everest files are saved.
    Merges the GAIA data for K2 with the output of bothFindP, saves this as a .csv file in the bothFindPOutput folder in the same root directory, 
    and returns it as a pandas dataframe. Before doing so, eliminates duplicate rows in the GAIA data, taking the one with the lowest angular distance. 
    Assumes the GAIA data is saved in the root directory as k2_dr2_1arcsec.fits and the bothFindP output is stored in the BothFindPOutput folder in the 
    root directory.
    '''
    k2_dr2 = Table.read(root_dir + 'k2_dr2_1arcsec.fits', format='fits')
    #k2_dr2 = Table.read(root_dir + 'everest/c08/k2_dr2_1arcsec.fits', format='fits')
    k2_dr2 = k2_dr2.to_pandas()
    ss = np.argsort(k2_dr2['k2_gaia_ang_dist'])
    uu = np.unique(k2_dr2['epic_number'].values[ss], return_index=True)[1]
    good_k2_dr2 = k2_dr2.iloc[ss[uu],:]
    campaign_output = pd.read_csv(root_dir + 'BothFindPOutput/' + campaign_output_file_name)
    name_column = campaign_output['File Name'].str.partition('llc_')[2].str.partition('-c')[0]
    name_column = pd.to_numeric(name_column)
    campaign_output = campaign_output.assign(epic_number=name_column)
    merged_output = campaign_output.merge(good_k2_dr2, left_on='epic_number', right_on='epic_number')
    merged_output.to_csv(root_dir + 'BothFindPOutput/Merged' + campaign_output_file_name)
    return merged_output

def concatenateFiles(merged_output_lis):
    '''
    Takes a list of outputs from mergeFiles and returns the concatentation of all of these, so the data from each can be analyzed together by makePlots.
    '''
    concatenated_file = merged_output_lis[0]
    if len(merged_output_lis)>1:
        for i in range(1,len(merged_output_lis)):
            concatenated_file = concatenated_file.append(merged_output_lis[i])
    return concatenated_file

def makePlots(merged_output, campaign, lower_threshold=0.1, upper_threshold=0.1, linear=False, close_only=False, LS=False, ACF=False, root_dir='/Volumes/Zoe Bell Backup/'):
    '''
    Takes the output of mergeFiles (or concatenateFiles) and a string name for the corresponding campaign(s) and prints and saves the following plots: 
    a comparison of LS Best Period, ACF First Period, and ACF Linear Period, Period v. BP-RP, a histogram of periods for BP-RP>=0.2, and M_G v. BP-RP colored by period.
    Several quality cuts are made to the data for the latter three plots. Optionally takes the lower and upper thresholds for selecting the data (if both equal 0.1, 
    then selects stars with LS and ACF periods within 10% of each other), whether to use the linear or first ACF period (with boolean linear), whether to only look at 
    stars within 300 parsecs (with boolean close_only), whether to just consider LS periods using the false alarm probability for selecting data (with boolean LS), whether 
    to just consider ACF periods using the ACF ratio for selection data (with boolean ACF), and the root directory in which to save the plots within the folder Plots!.
    '''
    plt.figure(figsize=(10,7))

    x = np.arange(0.0, 1.7, 0.1)

    log_LS_period = np.log10(merged_output['LS Best Period'])
    log_ACF_first_period = np.log10(merged_output['ACF First Period'])
    log_ACF_best_period = np.log10(merged_output['ACF Best Period'])

    low_prob = np.where(merged_output['False Alarm Prob']==0.0) #gets 4000 stars w/ campaign 8 with ==0.0     ##
    high_ACF = np.where(merged_output['Autocorrelation Ratio']>=0.26)  #0.26 gets 4000 stars w/ campaign 8

    close = np.where(merged_output['r_est']<=300)
    if close_only:
        log_LS_period = log_LS_period.values[close]
        log_ACF_first_period = log_ACF_first_period.values[close]
        log_ACF_best_period = log_ACF_best_period.values[close]

    plt.subplot(221)
    plt.plot(log_LS_period, log_ACF_first_period, 'bo', alpha=0.1)
    if LS:
        plt.plot(log_LS_period.values[low_prob], log_ACF_first_period.values[low_prob], 'ro', alpha=0.1)
    elif ACF:
        plt.plot(log_LS_period.values[high_ACF], log_ACF_first_period.values[high_ACF], 'yo', alpha=0.1)
    plt.plot(x, x,lw=3, color='g')
    plt.plot(x, x+np.log10(1-lower_threshold), 'g--')
    plt.plot(x, x+np.log10(1+upper_threshold), 'g--')
    plt.xlabel('log(LS Best Period)')
    plt.ylabel('log(ACF First Period)')
    
    plt.subplot(223)
    plt.plot(log_LS_period, log_ACF_best_period, 'bo', alpha=0.1)
    if LS:
        plt.plot(log_LS_period.values[low_prob], log_ACF_best_period.values[low_prob], 'ro', alpha=0.1)
    elif ACF:
        plt.plot(log_LS_period.values[high_ACF], log_ACF_best_period.values[high_ACF], 'yo', alpha=0.1)
    plt.plot(x, x,lw=3, color='g')
    plt.plot(x, x+np.log10(1-lower_threshold), 'g--')
    plt.plot(x, x+np.log10(1+upper_threshold), 'g--')
    plt.xlabel('log(LS Best Period)')
    plt.ylabel('log(ACF Linear Period)')

    plt.subplot(224)
    plt.plot(log_ACF_first_period, log_ACF_best_period, 'bo', alpha=0.1)
    if LS:
        plt.plot(log_ACF_first_period.values[low_prob], log_ACF_best_period.values[low_prob], 'ro', alpha=0.1)
    elif ACF:
        plt.plot(log_ACF_first_period.values[high_ACF], log_ACF_best_period.values[high_ACF], 'yo', alpha=0.1)
    plt.plot(x, x,lw=3, color='g')
    plt.plot(x, x+np.log10(1-lower_threshold), 'g--')
    plt.plot(x, x+np.log10(1+upper_threshold), 'g--')
    plt.xlabel('log(ACF First Period)')
    plt.ylabel('log(ACF Linear Period)')

    plt.suptitle('Period Comparison')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    if LS:
        plt.savefig(root_dir + 'Plots!/' + campaign + ' LS Method Comparison with Lower Threshold = ' + str(lower_threshold) + ' and Upper Threshold = ' + str(upper_threshold) + '.png', dpi=150)
    elif ACF:
        plt.savefig(root_dir + 'Plots!/' + campaign + ' ACF Method Comparison with Lower Threshold = ' + str(lower_threshold) + ' and Upper Threshold = ' + str(upper_threshold) + '.png', dpi=150)
    else:
        plt.savefig(root_dir + 'Plots!/' + campaign + ' Method Comparison with Lower Threshold = ' + str(lower_threshold) + ' and Upper Threshold = ' + str(upper_threshold) + '.png', dpi=150)
    plt.show()

    M_G = merged_output[u'phot_g_mean_mag'].values - 5. * np.log10(merged_output[u'r_est'].values)  + 5
    
    if LS:
        good = np.where((merged_output['False Alarm Prob']==0.0) & ###
                    (np.isfinite(merged_output[u'parallax'])) & 
                    (merged_output[u'parallax_error'] < 0.1) &
                    (merged_output[u'r_modality_flag'] == 1) & 
                    (merged_output[u'r_result_flag'] == 1) &
                    (np.isfinite(merged_output[u'bp_rp'])) & 
                    (merged_output[u'phot_bp_mean_flux_error']/merged_output[u'phot_bp_mean_flux'] < 0.01) & 
                    (merged_output[u'phot_rp_mean_flux_error']/merged_output[u'phot_rp_mean_flux'] < 0.01) & 
                    (merged_output[u'phot_g_mean_flux_error']/merged_output[u'phot_g_mean_flux'] < 0.01) &
                    (M_G>=4))[0]
    elif ACF:
        good = np.where((merged_output['Autocorrelation Ratio']>=0.26) & 
                    (np.isfinite(merged_output[u'parallax'])) & 
                    (merged_output[u'parallax_error'] < 0.1) &
                    (merged_output[u'r_modality_flag'] == 1) & 
                    (merged_output[u'r_result_flag'] == 1) &
                    (np.isfinite(merged_output[u'bp_rp'])) & 
                    (merged_output[u'phot_bp_mean_flux_error']/merged_output[u'phot_bp_mean_flux'] < 0.01) & 
                    (merged_output[u'phot_rp_mean_flux_error']/merged_output[u'phot_rp_mean_flux'] < 0.01) & 
                    (merged_output[u'phot_g_mean_flux_error']/merged_output[u'phot_g_mean_flux'] < 0.01) &
                    (M_G>=4))[0]
    else:
        if linear:
            good = np.where((merged_output['ACF Best Period']/merged_output['LS Best Period']>1-lower_threshold) & 
                        (merged_output['ACF Best Period']/merged_output['LS Best Period']<1+upper_threshold) &
                        (np.isfinite(merged_output[u'parallax'])) & 
                        (merged_output[u'parallax_error'] < 0.1) &
                        (merged_output[u'r_modality_flag'] == 1) & 
                        (merged_output[u'r_result_flag'] == 1) &
                        (np.isfinite(merged_output[u'bp_rp'])) & 
                        (merged_output[u'phot_bp_mean_flux_error']/merged_output[u'phot_bp_mean_flux'] < 0.01) & 
                        (merged_output[u'phot_rp_mean_flux_error']/merged_output[u'phot_rp_mean_flux'] < 0.01) & 
                        (merged_output[u'phot_g_mean_flux_error']/merged_output[u'phot_g_mean_flux'] < 0.01) &
                        (M_G>=4))[0]
        else:
            good = np.where((merged_output['ACF First Period']/merged_output['LS Best Period']>1-lower_threshold) & 
                        (merged_output['ACF First Period']/merged_output['LS Best Period']<1+upper_threshold) &
                        (np.isfinite(merged_output[u'parallax'])) & 
                        (merged_output[u'parallax_error'] < 0.1) &
                        (merged_output[u'r_modality_flag'] == 1) & 
                        (merged_output[u'r_result_flag'] == 1) &
                        (np.isfinite(merged_output[u'bp_rp'])) & 
                        (merged_output[u'phot_bp_mean_flux_error']/merged_output[u'phot_bp_mean_flux'] < 0.01) & 
                        (merged_output[u'phot_rp_mean_flux_error']/merged_output[u'phot_rp_mean_flux'] < 0.01) & 
                        (merged_output[u'phot_g_mean_flux_error']/merged_output[u'phot_g_mean_flux'] < 0.01) &
                        (M_G>=4))[0]
    
    if close_only:
        r_est = merged_output[u'r_est'].values[good]
        good = good[np.where((r_est <= 300.))[0]]

    selected_LS_periods = merged_output['LS Best Period'].values[good]
    if linear:
        selected_ACF_periods = merged_output['ACF Best Period'].values[good]
    else:
        selected_ACF_periods = merged_output['ACF First Period'].values[good]
    
    bp_rp = merged_output['bp_rp'].values[good]
    #plt.figure(figsize=(7,4))                                  ###
    if LS:
        plt.plot(bp_rp, selected_LS_periods, 'bo', alpha=0.05) # change to alpha=0.05 for combos
    else:
        plt.plot(bp_rp, selected_ACF_periods, 'bo', alpha=0.05) # change to alpha=0.05 for combos
    plt.yscale('log')
    #plt.xlim(0.5,2.7)                                          ###
    #plt.ylim(1,100)                                            ###
    plt.xlabel('bp_rp')
    if LS:
        plt.ylabel('LS Best Period')
        plt.savefig(root_dir + 'Plots!/' + campaign + ' LS Period v. BP-RP with Lower Threshold = ' + str(lower_threshold) + ' and Upper Threshold = ' + str(upper_threshold) + '.png', dpi=150)
    else:
        name = root_dir + 'Plots!/' + campaign
        if linear:
            plt.ylabel('ACF Linear Period')
            name = name + ' Linear'
        else:
            plt.ylabel('ACF First Period')
        if ACF:
            plt.savefig(name + ' ACF Period v. BP-RP with Lower Threshold = ' + str(lower_threshold) + ' and Upper Threshold = ' + str(upper_threshold) + '.png', dpi=150)
        else:
            plt.savefig(name + ' Period v. BP-RP with Lower Threshold = ' + str(lower_threshold) + ' and Upper Threshold = ' + str(upper_threshold) + '.png', dpi=150)
    plt.show()

    ok = np.where(merged_output['bp_rp'].values[good]>=2)
    if LS:
        plt.hist(selected_LS_periods[ok], bins=15)
        plt.xlabel('LS Best Period (where bp_rp>=2)')
        plt.savefig(root_dir + 'Plots!/' + campaign + ' LS Period Histogram with Lower Threshold = ' + str(lower_threshold) + ' and Upper Threshold = ' + str(upper_threshold) + '.png', dpi=150)
    else:
        name = root_dir + 'Plots!/' + campaign
        plt.hist(selected_ACF_periods[ok], bins=15)
        if linear:
            plt.xlabel('ACF Linear Period (where bp_rp>=2)')
            name = name + ' Linear'
        else:
            plt.xlabel('ACF First Period (where bp_rp>=2)')
        if ACF:
            plt.savefig(name + ' ACF Period Histogram with Lower Threshold = ' + str(lower_threshold) + ' and Upper Threshold = ' + str(upper_threshold) + '.png', dpi=150)
        else:
            plt.savefig(name + ' Period Histogram with Lower Threshold = ' + str(lower_threshold) + ' and Upper Threshold = ' + str(upper_threshold) + '.png', dpi=150)
    plt.show()

    if LS:
        plt.scatter(bp_rp, M_G[good], c=selected_LS_periods, alpha=0.7, s=2, cmap=plt.cm.get_cmap('Spectral_r'))
    else:
        plt.scatter(bp_rp, M_G[good], c=selected_ACF_periods, alpha=0.7, s=2, cmap=plt.cm.get_cmap('Spectral_r'))
    plt.ylim(12,-2)
    plt.xlabel('bp_rp')
    plt.ylabel('M_G')
    cb = plt.colorbar()
    if LS:
        cb.set_label('LS Best Period (days)')
        plt.savefig(root_dir + 'Plots!/' + campaign + ' LS M_G v. BP-RP with Lower Threshold = ' + str(lower_threshold) + ' and Upper Threshold = ' + str(upper_threshold) + '.png', dpi=150)
    else:
        name = root_dir + 'Plots!/' + campaign
        if linear:
            cb.set_label('ACF Linear Period (days)')
            name = name + ' Linear'
        else:
            cb.set_label('ACF First Period (days)')
        if ACF:
            plt.savefig(name + ' ACF M_G v. BP-RP with Lower Threshold = ' + str(lower_threshold) + ' and Upper Threshold = ' + str(upper_threshold) + '.png', dpi=150)
        else:
            plt.savefig(name + ' M_G v. BP-RP with Lower Threshold = ' + str(lower_threshold) + ' and Upper Threshold = ' + str(upper_threshold) + '.png', dpi=150)
    plt.show()

def bothFindP(files, output_name, gen_LSplots=False, gen_ACFplots=False, gen_file_type='.png', gen_min_period=0.1, gen_max_period=30, gen_min_max_distance=1, gen_medfilt_kernel_size=11, gen_fcor_box_kernel_size=11, gen_acf_box_kernal_size=100, gen_root_dir = '/Volumes/Zoe Bell Backup/'):
    '''
    Takes a list of file names for .fits files from the Everest K2 data and a name for the output file,
    and optionally whether you want LS or ACF plots saved, the file type you want it saved as, the minimum and maximum periods you want to look for, 
    the minimum distance between maxes in the ACF you want to consider, several kernal sizes of various filters and smoothing functions, and the 
    root directory in which to save the output .csv file within the folder BothFindPOutput.
    Returns the outputs of LSfindP and ACFfindP for each file in a list and saves them in a .csv file with the following headers: 
    'File Name', 'LS Best Period', 'Max Power', 'False Alarm Prob', 'ACF Best Period', 'ACF First Period', 'Autocorrelation Ratio', and 'Peaks List.'
    '''
    start_time = time.clock()
    lis = []
    for file_name in files:
        start = file_name.rfind('/') + 1
        name = file_name[start:-5]
        data = Table.read(file_name, format='fits')
        new_row = LSfindP(name, data, plots=gen_LSplots, file_type=gen_file_type, min_period=gen_min_period, max_period=gen_max_period, root_dir=gen_root_dir) + ACFfindP(name, data, plots=gen_ACFplots, file_type=gen_file_type, min_period=gen_min_period, max_period=gen_max_period, min_max_distance=gen_min_max_distance, fcor_box_kernel_size=gen_fcor_box_kernel_size, acf_box_kernal_size=gen_acf_box_kernal_size, root_dir=gen_root_dir)[1:]
        lis.append(new_row)
    output = pd.DataFrame(data=lis, columns=['File Name', 'LS Best Period', 'Max Power', 'False Alarm Prob', 'ACF Best Period', 'ACF First Period', 'Autocorrelation Ratio', 'Peaks List'])
    output.to_csv(gen_root_dir + 'BothFindPOutput/' + output_name + '.csv')
    end_time = time.clock()
    print(str(end_time-start_time) + ' seconds')
    return output

def LSfindMultipleP(files, output_name, gen_plots=False, gen_file_type='.png', gen_min_period = 0.1, gen_max_period = 30, gen_root_dir = '/Volumes/Zoe Bell Backup/'):
    '''
    Takes a list of file names for .fits files from the Everest K2 data and a name for the output file, and optionally
    whether you want plots saved, the file type you want them saved as, the minimum and maximum periods you want to look for, 
    and the root directory in which to save the output .csv within the folder FindPOutput.
    Returns the output of LSfindP for each file in a list and saves them in a .csv file.
    '''
    lis = []
    for file_name in files:
        start = file_name.rfind('/') + 1
        name = file_name[start:-5]
        data = Table.read(file_name, format='fits')
        lis.append(LSfindP(name, data, plots=gen_plots, file_type=gen_file_type, min_period=gen_min_period, max_period=gen_max_period, root_dir=gen_root_dir))
    output = pd.DataFrame(data=lis, columns=['File Name', 'Best Period','Max Power','False Alarm Prob'])
    output.to_csv(gen_root_dir + 'FindPOutput/' + output_name + '.csv')
    return output

def LSfindP(file_name, data, plots=False, file_type='.png', min_period = 0.1, max_period = 30, root_dir = '/Volumes/Zoe Bell Backup/'):
    '''
    Takes the name of a .fits file (without the extension) from the Everest K2 data and the data from that file read into a table,
    and optionally whether you want a plot saved, the file type you want it saved as, the minimum and maximum periods you want to look for, 
    and the root directory in which to save the plot within the folder LSPlotOutputs.
    Uses the Lomb-Scargle periodogram method to return the file name, the period that best fits the corrected flux data in that range, 
    the power at that period, and the associated false alarm probability.
    '''
    #start = file_name.rfind('/') + 1
    #name = file_name[start:-5]
    name = file_name

    #data = Table.read(file_name, format='fits')
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
        plt.savefig(root_dir + 'LSPlotOutputs/' + name + file_type, dpi=150)
        plt.close()

    return [name, 1/best_freq, max_power, ls.false_alarm_probability(max_power)]

def ACFfindMultipleP(files, output_name, gen_plots=False, gen_file_type='.png', gen_min_period=0.1, gen_max_period=30, gen_min_max_distance=0.2, gen_medfilt_kernel_size=11, gen_fcor_box_kernel_size=11, gen_acf_box_kernal_size=100, gen_root_dir = '/Volumes/Zoe Bell Backup/'):
    '''
    Takes a list of file names for .fits files from the Everest K2 data and a name for the output file, 
    and optionally whether you want plots saved, the file type you want it saved as, the minimum and maximum periods you want to look for, 
    the minimum distance between maxes in the ACF you want to consider, several kernal sizes of various filters and smoothing functions, 
    and the root directory in which to save the output .csv within the folder ACFfindPOutput.
    Returns the output of ACFfindP for each file in a list and saves them in a .csv file.
    '''
    start_time = time.clock()
    lis = []
    for file_name in files:
        start = file_name.rfind('/') + 1
        name = file_name[start:-5]
        data = Table.read(file_name, format='fits')
        lis.append(ACFfindP(name, data, plots=gen_plots, file_type=gen_file_type, min_period=gen_min_period, max_period=gen_max_period, min_max_distance=gen_min_max_distance, fcor_box_kernel_size=gen_fcor_box_kernel_size, acf_box_kernal_size=gen_acf_box_kernal_size, root_dir=gen_root_dir))
    output = pd.DataFrame(data=lis, columns=['File Name', 'Best Period', 'First Period', 'Autocorrelation Ratio', 'Period List'])
    output.to_csv(gen_root_dir + 'ACFfindPOutput/' + output_name + '.csv')
    end_time = time.clock()
    print(str(end_time-start_time) + ' seconds')
    return output

def ACFfindP(file_name, data, plots=False, file_type='.png', min_period=0.1, max_period=30, min_max_distance=0.22, medfilt_kernel_size=11, fcor_box_kernel_size=11, acf_box_kernal_size=100, root_dir = '/Volumes/Zoe Bell Backup/'): #maybe default should be 40?
    '''
    Takes the name of a .fits file (without the extension) from the Everest K2 data and the data from that file read into a table,
    and optionally whether you want a plot saved, the file type you want it saved as, the minimum and maximum periods you want to look for, 
    the minimum distance between maxes in the ACF you want to consider, several kernal sizes of various filters and smoothing functions, 
    and the root directory in which to save the plot within the folder ACFPlotOutputs.
    Uses the Auto-Correlation Function method to return the file name, the period that best fits the corrected flux data in that range (based on all maxes found), 
    the first peak found, the ratio of the autocorrelation magnitude of the highest peak to the magnitude when shifted by zero, and the first 10 peaks found.
    '''
    #start = file_name.rfind('/') + 1
    #name = file_name[start:-5]
    name = file_name

    #data = Table.read(file_name, format='fits')
    ok = np.where((data['QUALITY']==0) & (np.isfinite(data['TIME'])) & (np.isfinite(data['FCOR']) & (np.isfinite(data['FRAW_ERR']))))

    t = np.array(data['TIME'][ok])
    fcor = list(data['FCOR'][ok])
    fcor_median = fcor/np.median(fcor)-1
    fcor_median = sig.medfilt(fcor_median, kernel_size=medfilt_kernel_size)
    fcor_median = convolve(fcor_median, Box1DKernel(fcor_box_kernel_size, mode='center'))

    N = len(fcor)
    # t_step = np.nanmedian(t[1:]-t[0:-1]) # 29.4 min expected but this ended up chronically giving us too short overall time frames
    t_step = (np.nanmax(t) - np.nanmin(t)) / np.float(np.size(t))
    t_range = np.arange(N)*t_step

    print(t_step)
    plt.plot((t-t[0]) - t_range)
    plt.xlabel('Data Point')
    plt.ylabel('Actual Time Elapsed - Regularized Time Elapsed (Days)')
    plt.show()

    arr = sig.correlate(fcor_median,fcor_median, mode='full')
    ACF = arr[N-1:]
    baseline_ACF = ACF[0]

    okP = np.where(((t_range>min_period) & (t_range<max_period)))
    t_search = t_range[okP]
    ACF_search = ACF[okP]

    ACF_search = convolve(ACF_search, Box1DKernel(acf_box_kernal_size)) # 200 good; automatically does linear_interp, can change mode='center'
    
    maxes = findMaxes(t_search, ACF_search, min_max_distance)
    periods = maxes[0]
    ACFs = maxes[1]
    first_peak = -1
    if len(periods)>1:
        first_peak = periods[1]
    if len(periods)<2:
        if(plots):
            plt.figure(figsize=(10,7))

            plt.title('Unsmoothed v. Smoothed')
            plt.xlabel('Time Shift (days)')
            plt.ylabel('Autocorrelation')
            plt.plot(t_range, ACF)
            plt.plot(t_search, ACF_search)

            formated_periods = []
            for n in periods[:10]:
                plt.axvline(n)                                                      #####
                formated_periods.append(format(n, '.2f'))

            plt.suptitle(formated_periods)
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            plt.savefig(root_dir + 'ACFPlotOutputs/' + name + file_type, dpi=150)
            plt.close()
        return [name, -1, -1, -1, periods[:10]]
    max_ACF = np.max(ACFs[1:])
    
    linPossible = len(periods)>2
    if linPossible:
        linInfo = stats.linregress(range(len(periods)-1), periods[1:])
        linSlope = linInfo[0] # /actual best period
        linIntercept = linInfo[1]
    else:
        linSlope = periods[1]

    if(plots):
        plt.figure(figsize=(10,7))

        plt.subplot(211)
        plt.title('Unsmoothed v. Smoothed')
        plt.xlabel('Time Shift (days)')
        plt.ylabel('Autocorrelation')
        plt.plot(t_range, ACF)
        plt.plot(t_search, ACF_search)

        formated_periods = []
        for n in periods[:10]:
            plt.axvline(n)
            formated_periods.append(format(n, '.2f'))

        if linPossible:
            plt.subplot(212)
            plt.title('Maxes with the Best Period ' + format(linSlope, '.2f'))
            plt.xlabel('Index')
            plt.ylabel('Time Shift (days)')
            plt.plot(periods[1:], 'bo')
            x = range(len(periods)-1)
            y = []
            for x_val in x:
                y.append(linIntercept + linSlope*x_val)
            plt.plot(x, y)

        plt.suptitle(formated_periods)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.savefig(root_dir + 'ACFPlotOutputs/' + name + file_type, dpi=150)
        plt.close()
    
    return [name, linSlope, first_peak, max_ACF/baseline_ACF, periods[:10]]

def findMaxes(t_search, ACF_search, min_max_distance):
    '''
    Takes a list of time shifts, a list of their associated autocorrelation values, and the minimum distance between maxes in the ACF you want to allow.
    Returns a list of the list of time shifts and the list of associated autocorrelation values found to be at local maximums of the ACF data.
    '''
    grad = np.gradient(ACF_search)
    zeros = []
    for i in range(5, len(grad)-5): # orginally 1 not 5
        if (grad[i-5]>0) & (grad[i+5]<0):
            if (len(zeros)>0):
                if (t_search[i]-t_search[zeros[-1]]) >= min_max_distance: #only counting maxes at least one day away from each other (try going down to .2 days)
                    zeros.append(i)
            else:
                zeros.append(i)
    
    gradSnd = np.gradient(grad)
    maxes = []
    for zero in zeros:
        if gradSnd[zero]<0:
            maxes.append(zero)
    
    return [t_search[maxes], ACF_search[maxes]]

def findWidthAtHalfMax(t_search, ACF_search):
    max = ACF_search[0]
    index = 0
    for i in range(len(ACF_search)):
        if ACF_search[i] < 0.5*max:
            index = i
            break
    return t_search[index]

# get things onto GitHub
# fine-tune where statement
# get things within 10% of ACF/LS median
# commenting/README
# r_est do within 200-300 parsecs
# look at ACF plot, put vertical line down where period is predicted -- nothing looks weird
# look at actual v. regularized time elapsed -- by end, 8-10 more days have passed according to actual
# put in Guassian for peaks

# took 5,290.2 secs (about 1 hour and 30 min) to run bothFindP with all plots on 7,748 files (all of campaign 0)
# took 14,624.8 secs (about 4 hours) to run bothFindP with all plots on 21,407 files (all of campaign 13)

# grab plus 10 (5) minus 20 (25)
# assume Lomb-Scargle is right

# POSTER STUFF
# save as dpi = 300 for poster printing
# funding for poster: 'This work was supported by an NSF Astronomy and Astrophysics Postdoctoral Fellowship under award AST-1501418.'

# be able to switch to linear ACF period
# switchable directories root_dir = '/Volumes/Zoe Bell Backup/'

# make concatenate func that concatenates merged files

# took 9,241.7 secs (about 2 hours and 35 min) to run bothFindP with all plots on 13,483 files (all of campaign seven)
# took 15,598.1 secs (about 4 hours and 20 min) to run bothFindP with all plots on 23,074 files (all of campaign five)

# look at LS prob v. amplitude
# add campaign to plot file names
# write two sentences in README ```` python code ```` (how to run, how to interpret output)
# look at differences in 2014 paper

# add quality cuts
# make histogram of logP for bp_rp>=2 (like in McQ '13) with _ = plt.hist
# save period-color, period historgram, and color-magnitude diagrams
# also plot Mg (intrinsic brightness) v. BP-RP; color data by Prot (Mg = mg-5*log(1000/parallax)+5)
# do both for 10% and 20%

# use pd.merge() (or join) (pandas) to match up two files based on star ID
# to read in file, once table use myTable.to_pandas() (something like that) to convert to pandas dataframe
# compare 3 log periods as scatter plots; make transparent (alpha = 0.3)
# trust those within 10% of the 1:1 line
# of those plot Prot v. BP-RP with y axis log scale

# took 14,496.4 secs (about 4.0 hours) to run bothFindP with all plots on 21,387 files (all of campaign eight)
# took 429.1 secs (about 7 minutes) to run bothFindP with all plots on 622 files

# add license to GitHub—-license.md (MIT one recommended)
# documentation!
# took 58.3 secs to run with ACF with plots on 622 files
# make new wrapper func
# change where figs save
# see if can get min max distance down to 0.2 days (make something that can be fiddled with, also do so for smoothing things)

# took 52.5 secs to run on 622 files
# print out time (import time)

# average over spacing of peaks—-graph
# check outputs against lomb-scargle
# look at 2013 paper for how to use ACF (use same limits)

# scipy spline to smooth, pandas rolling/running stats package for media (like boxcar for mean)
# check boxcar (why artifact at beginning?)
# write func that calcs derivs and finds zeros and neg curvature


# boxcar smooth (running average) for gross ones
# use full width at half max for box size
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



