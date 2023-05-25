import numpy as np
from Import_data import import_data
import matplotlib.pyplot as plt
from ECG_heartbeat_detection import r_peak_detect, detect_and_save
import pandas as pd
from IPython.display import display

Import_data_flag = True
r_peak_detect_save_flag = False  # True False
Import_r_peaks_flag = True

# Import Data
if Import_data_flag:
    print("importing data...")
    print("original signals")
    records_df = import_data(noisy_flag=False)
    print("noisy signals (e24)")
    noisy_records_df = import_data(noisy_flag=False, noise_str="e24")
    print("data imported.")

# r peak detection and saving
if r_peak_detect_save_flag:
    # https://github.com/berndporr/py-ecg-detectors
    # ECG heartbeat detection algorithms: hamilton_detector, christov_detector, engzee_detector, pan_tompkins_detector,
    # swt_detector, two_average_detector, matched_filter_detector, wqrs_detector
    ECG_algorithms = ['hamilton_detector', 'christov_detector', 'engzee_detector', 'pan_tompkins_detector',
                      'swt_detector', 'two_average_detector', 'matched_filter_detector', 'wqrs_detector']

    # for original signals
    print("detecting r peaks for original signals...")
    detect_and_save(records_df, ECG_algorithms, filename_str='r_peak_dictionary_original_sig.pkl')
    print("completed.")
    # exit()

    # for noisy signals
    print("detecting r peaks for noisy signals...")
    detect_and_save(noisy_records_df, ECG_algorithms, filename_str='r_peak_dictionary_noisy_sig_e24.pkl')
    print("completed.")
    # exit()


# Import r peaks
if Import_r_peaks_flag:
    # for original signals
    original_df = pd.read_pickle("r_peak_dictionary_original_sig.pkl")
    # display(original_df)
    print(">>>", original_df.loc[7000])
    print(">>>", original_df.loc[11518])
    print(">>>", original_df.loc[11520])
    print(original_df.shape)

    # for noisy signals
    noisy_df = pd.read_pickle("r_peak_dictionary_noisy_sig_e24.pkl")
    print(noisy_df.shape)

    # # delete 'engzee_detector' records (it has issue)
    # # for original signals
    # indices = original_df[(original_df['algorithm'] == 'engzee_detector')].index
    # original_df = original_df.drop(indices).reset_index(drop=True)
    # print(original_df.shape)
    # print("saving the results...")
    # original_df.to_pickle("r_peak_dictionary_original_sig.pkl")
    #
    # # for noisy signals
    # indices = noisy_df[(noisy_df['algorithm'] == 'engzee_detector')].index
    # noisy_df = noisy_df.drop(indices).reset_index(drop=True)
    # print(noisy_df.shape)
    # print("saving the results...")
    # noisy_df.to_pickle("r_peak_dictionary_noisy_sig_e24.pkl")

    exit()


