from ecgdetectors import Detectors
from tqdm import tqdm
import pickle
import pandas as pd

from IPython.display import display, HTML

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


def detect_and_save(records_df, ECG_algorithms, filename_str='r_peak_dictionary.pkl'):

    df = pd.DataFrame(columns=['patient', 'channel', 'algorithm', 'out_flag', 'r_peak'])
    for alg in ECG_algorithms:
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
                for arr in signals:
                    out_flag, result = r_peak_detect(arr, alg)

                    if not out_flag:
                        result = []
                    dictionary = {'patient': patient, 'channel': channel, 'algorithm': alg,
                                  'out_flag': out_flag, 'r_peak': result}
                    temp_df = pd.DataFrame(dictionary)

                    df = pd.concat([df, temp_df], ignore_index=True)
                    df.reset_index()

                    sum_records += 1
                    if out_flag:
                        sum_true += 1
        print("\npercentage of samples with no error: ", (sum_true/sum_records)*100, "%")
        print("next algorithm...")

    print("performed the r peak detection for all algorithms in the list.")
    print("saving the results...")
    df.to_pickle(filename_str)
    print("results saved.")
