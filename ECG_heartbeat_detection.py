import numpy as np
from ecgdetectors import Detectors
from tqdm import tqdm
import pandas as pd
from IPython.display import display
import math


# detect r peaks
fs = 360  # sample freq
detectors = Detectors(fs)


def r_peak_detect(in_array, algorithm):
    string_script = 'detectors.' + algorithm + '(in_array)'

    result = None
    try:
        result = eval(string_script)  # exec(string_script)
        out_flag = True
    except (IndexError, ValueError):
        out_flag = False

    return out_flag, result


def get_rr_intervals(r_peak_vector):
    rr_intervals_vec = r_peak_vector[1:]
    rr_intervals_vec -= r_peak_vector[:-1]
    return rr_intervals_vec


def detect_and_save(records_df, ECG_r_peak_detection_algorithms, filename_str='r_peak_dictionary.pkl'):

    noise_label_per_minute = [0, 0, 0, 0, 0, 1, 1, 0, 0,
                     1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
                     0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]

    df = pd.DataFrame(columns=['patient', 'channel', 'window_num', 'minute_index', 'noise_label',
                               'algorithm', 'out_flag', 'r_peak', 'rr_intervals', 'rr_intervals_vec_length'])
    for alg in ECG_r_peak_detection_algorithms:
        print("\nalgorithm: ", alg)
        sum_true = 0
        sum_records = 0
        for patient, patient_index in \
                zip(records_df['patient'].unique(), tqdm(range(len(records_df['patient'].unique())))):
            for channel in records_df['channel'].unique():
                signals_index = \
                    records_df[(records_df['patient'] == patient) &
                                    (records_df['channel'] == channel)].index
                signals = records_df.loc[signals_index[0]]['signals']
                for i, arr in enumerate(signals):
                    out_flag, result = r_peak_detect(arr, alg)

                    if not out_flag:  # might be changed for 'engzee_detector'
                        result = []
                        rr_intervals_vec_len = 0
                        rr_intervals_result = []
                    else:
                        rr_intervals_vec_len = len(result) - 1
                        if rr_intervals_vec_len > 0:
                            rr_intervals_result = [get_rr_intervals(np.array(result))]
                        else:
                            rr_intervals_result = []
                        result = [np.array(result)]

                    minute_index = int(math.floor(i/3))  # since every minute was divided by 3

                    dictionary = {'patient': patient, 'channel': channel, 'window_num': i, 'minute_index': minute_index,
                                  'noise_label': noise_label_per_minute[minute_index], 'algorithm': alg,
                                  'out_flag': out_flag, 'r_peak': result, 'rr_intervals': rr_intervals_result,
                                  'rr_intervals_vec_length': rr_intervals_vec_len}
                    temp_df = pd.DataFrame(dictionary)
                    # display(temp_df)

                    df = pd.concat([df, temp_df], ignore_index=True)
                    df.reset_index()

                    sum_records += 1
                    if out_flag:
                        sum_true += 1
        print("\npercentage of samples with no error: ", (sum_true/sum_records)*100, "%. ",
              "sum_true: ", sum_true, "sum_records: ", sum_records)
        print("next algorithm...")

    print("performed the r peak detection for all algorithms in the list.")
    print("saving the results...")
    df.to_pickle(filename_str)
    print("results saved.")
