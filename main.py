import numpy as np
from Import_data import import_data
import matplotlib.pyplot as plt
from ECG_heartbeat_detection import r_peak_detect, detect_and_save
import pandas as pd
from IPython.display import display
import Analysis

Import_data_flag = True
r_peak_detect_save_flag = False  # True False
Import_r_peaks_flag = True


# Import Data
if Import_data_flag:
    print("importing data...")
    print("signals including noise (e_4). After first 5 min, every 2 min is noisy")
    """If no protocol annotation file is specified, nst generates one using a standard protocol 
    (a five-minute noise-free ‘‘learning period’’, followed by two-minute periods of noisy 
    and noise-free signals alternately until the end of the clean record). The gains to be applied 
    during the noisy periods are determined in this case by measuring the signal and noise amplitudes 
    (see Signal-to-noise ratios, below).
    https://physionet.org/physiotools/wag/nst-1.htm"""
    records_df = import_data(noise_str="e_6")
    print("data imported.")


# r peak detection and saving
if r_peak_detect_save_flag:
    # https://github.com/berndporr/py-ecg-detectors
    # ECG heartbeat detection algorithms: hamilton_detector, christov_detector, engzee_detector, pan_tompkins_detector,
    # swt_detector, two_average_detector, matched_filter_detector, wqrs_detector
    # , 'engzee_detector' excluded
    ECG_r_peak_detection_algorithms = ['hamilton_detector', 'christov_detector', 'pan_tompkins_detector',
                                       'swt_detector', 'two_average_detector', 'matched_filter_detector',
                                       'wqrs_detector']


    # for noisy signals
    print("detecting r peaks for noisy signals...")
    detect_and_save(records_df, ECG_r_peak_detection_algorithms,
                    filename_str='signals_including_noise/records_df_e_6.pkl')
    print("completed.")
    # exit()


# Import r-peak detection results (r peaks, rr intervals, etc.)
if Import_r_peaks_flag:
    records_df = pd.read_pickle("signals_including_noise/records_df_e_6.pkl")
    print(records_df.shape)

    # length = records_df['r_peak'].apply(len)
    # min_length = length.min()
    # max_length = length.max()
    # mean_length = length.mean()
    # std_length = length.std()
    # print(min_length, max_length, mean_length, std_length)

    print("preparing X_train, X_test, y_train, y_test...")
    X_train, X_test, y_train, y_test = Analysis.data_preparation(records_df)

    print('building the model...')
    model = Analysis.build_ml_model(X_train, y_train)

    print('evaluation of the model...')
    Analysis.evaluate(model, X_test, y_test)
