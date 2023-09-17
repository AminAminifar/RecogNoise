import numpy as np
from Import_data import import_data
import matplotlib.pyplot as plt
from ECG_heartbeat_detection import r_peak_detect, detect_and_save
import pandas as pd
from IPython.display import display
import Analysis
from sklearn.utils import shuffle
from tqdm import tqdm

Import_data_flag = False
r_peak_detect_save_flag = False  # True False
Analysis_one_noise_snr = True
Analysis_mixed = True
Analysis_mixed_snr = False
Analysis_mixed_all = not Analysis_mixed_snr  # True

num_iterations = 100
num_metrics = 4
result_mat = np.zeros((num_iterations, num_metrics))

em_str_list = ['e_6', 'e00', 'e06', 'e12', 'e18', 'e24']
bw_str_list = ['b_6', 'b00', 'b06', 'b12', 'b18', 'b24']
ma_str_list = ['m_6', 'm00', 'm06', 'm12', 'm18', 'm24']

noise_str = bw_str_list[0]  # bw_str_list[5] ma_str_list[5]
print("noise_str: ", noise_str, "\n")

# Import Data
if Import_data_flag:
    print("importing data...")
    print("signals including noise (e...). After first 5 min, every 2 min is noisy")
    """If no protocol annotation file is specified, nst generates one using a standard protocol 
    (a five-minute noise-free ‘‘learning period’’, followed by two-minute periods of noisy 
    and noise-free signals alternately until the end of the clean record). The gains to be applied 
    during the noisy periods are determined in this case by measuring the signal and noise amplitudes 
    (see Signal-to-noise ratios, below).
    https://physionet.org/physiotools/wag/nst-1.htm"""
    # records_df = import_data(records_folder_add='signals_including_noise/e00/mitbih_e00/', noise_str="e00")  # noise_str="e_6"
    records_df = import_data(records_folder_add='signals_including_noise/' + noise_str + '/mitbih_' + noise_str + '/',
                             noise_str=noise_str)  # noise_str="e_6"
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
    # detect_and_save(records_df, ECG_r_peak_detection_algorithms,
    #                 filename_str='signals_including_noise/e00/records_df_e00.pkl')
    detect_and_save(records_df, ECG_r_peak_detection_algorithms,
                    filename_str='signals_including_noise/' + noise_str + '/records_df_' + noise_str + '.pkl')
    print("completed.")
    # exit()


# Import r-peak detection results (r peaks, rr intervals, etc.)
if Analysis_one_noise_snr:
    print("\nAnalysis_one_noise_snr: ")

    df = pd.DataFrame(columns=['noise_str', 'f1_performance', 'acc_performance',
                               'precision_performance', 'recall_performance'])

    noise_str_list = em_str_list + bw_str_list + ma_str_list
    for noise_str in noise_str_list:
        print("noise_str: ", noise_str, "\n")
        print("num iterations: ", num_iterations, "\n")

        for iteration in tqdm(range(num_iterations)):
            # records_df = pd.read_pickle("signals_including_noise/e_6/records_df_e_6.pkl")
            # records_df = pd.read_pickle("signals_including_noise/e00/records_df_e00.pkl")
            records_df = pd.read_pickle('signals_including_noise/' + noise_str + '/records_df_' + noise_str + '.pkl')
            # print(">>>>", records_df.query('out_flag == False').index)
            # records_df = pd.read_pickle("signals_including_noise/no_noise/records_df.pkl")
            # print(records_df.shape)

            # length = records_df['r_peak'].apply(len)
            # min_length = length.min()
            # max_length = length.max()
            # mean_length = length.mean()
            # std_length = length.std()
            # print(min_length, max_length, mean_length, std_length)

            # Analysis
            # print("\npreparing X_train, X_test, y_train, y_test...")
            X_train, X_test, y_train, y_test = Analysis.data_preparation(records_df)

            # print('\nbuilding the model...')
            model = Analysis.build_ml_model(X_train, y_train)

            # print('\nevaluation of the model...')
            result_mat[iteration, :] = Analysis.evaluate(model, X_test, y_test)

        mean_results = np.mean(result_mat, axis=0)
        print("\nmean values for f1_performance, acc_performance, "
              "precision_performance, recall_performance:", mean_results)
        dictionary = {'noise_str': noise_str, 'f1_performance': [mean_results[0]], 'acc_performance': [mean_results[1]],
                      'precision_performance': [mean_results[2]], 'recall_performance': [mean_results[3]]}

        temp_df = pd.DataFrame(dictionary)
        df = pd.concat([df, temp_df], ignore_index=True)
        df.reset_index()
    df.to_csv('results_for_Analysis_one_noise_snr.csv', index=False)

if Analysis_mixed:
    if Analysis_mixed_snr:
        noise_str_list = em_str_list  # ma_str_list  bw_str_list  em_str_list
        print("\nAnalysis_mixed_snr: ")
    elif Analysis_mixed_all:
        noise_str_list = em_str_list + bw_str_list + ma_str_list
        # print(noise_str_list)
        print("\nAnalysis_mixed_all: ")

    print("num iterations: ", num_iterations, "\n")
    for iteration in tqdm(range(num_iterations)):

        noise_str = noise_str_list[0]
        records_df = pd.read_pickle('signals_including_noise/' + noise_str + '/records_df_' + noise_str + '.pkl')

        indices_noisy_records = records_df.query('noise_label == 1').index
        num_noisy_records = len(indices_noisy_records)
        num_chunks = len(noise_str_list)
        chunk_indices = np.array_split(shuffle(indices_noisy_records), num_chunks)
        # print("num_chunks, num_noisy_records: ", num_chunks, num_noisy_records)

        for i in range(1, num_chunks):
            noise_str = noise_str_list[i]
            temp_df = pd.read_pickle('signals_including_noise/' + noise_str + '/records_df_' + noise_str + '.pkl')

            records_df.loc[chunk_indices[i], 'rr_intervals'] = temp_df.loc[chunk_indices[i], 'rr_intervals']
            # not necessary for our analysis
            records_df.loc[chunk_indices[i], 'r_peak'] = temp_df.loc[chunk_indices[i], 'r_peak']
            # not necessary for our analysis
            records_df.loc[chunk_indices[i], 'rr_intervals_vec_length'] = temp_df.loc[chunk_indices[i], 'rr_intervals_vec_length']

        # Analysis


        # print("\npreparing X_train, X_test, y_train, y_test...")
        X_train, X_test, y_train, y_test = Analysis.data_preparation(records_df)

        # print('\nbuilding the model...')
        model = Analysis.build_ml_model(X_train, y_train)

        # print('\nevaluation of the model...')
        result_mat[iteration, :] = Analysis.evaluate(model, X_test, y_test)
        # print(result_mat[iteration, :])

    mean_results = np.mean(result_mat, axis=0)
    print("\nmean values for f1_performance, acc_performance, precision_performance, recall_performance:", mean_results)
