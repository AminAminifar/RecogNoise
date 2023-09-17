import numpy as np
from wfdb.io.record import rdsamp
import pandas as pd


def import_data(records_folder_add, noise_str="e_6"):  # noise_str="e24"
    # Import original signals
    records = np.loadtxt("signals_including_noise/RECORDS", dtype=int)
    # num_records = len(records)
    num_channels = 2
    records_df = pd.DataFrame(columns=['patient', 'channel', 'signals'])
    for record in records:
        for ch in range(num_channels):
            # sig, _ = rdsamp('signals_including_noise/mitbih_e_6/' + str(record) + noise_str, channels=[ch])
            sig, _ = rdsamp( records_folder_add + str(record) + noise_str, channels=[ch])
            # sig, _ = rdsamp('signals_including_noise/no_noise/mitbih/' + str(record), channels=[ch])  # no noise
            sig_f = sig.flatten()
            # if we need the whole signal as one records:
            # records_list.append(sig_f)
            # (else) if we need to split the signal:
            num_windows = 30*3  # 20 seconds
            splitted_signals = [np.split(sig_f[:648000], num_windows)]  # 648000 = 30 * 60 * 360

            dictionary = {'patient': record, 'channel': ch, 'signals': splitted_signals}
            temp_df = pd.DataFrame(dictionary)
            records_df = pd.concat([records_df, temp_df], ignore_index=True)
            records_df.reset_index()

    return records_df

# print(len(records_list), np.array(records_list)[:, :, 0].shape, type(records_list))
# np.savetxt("records.csv", np.array(records_list)[:, :, 0], delimiter=",")
# with open('test.npy', 'wb') as f:
#     np.save(f, np.array(records_list)[:, :, 0])
# with open('test.npy', 'rb') as f:
#     a = np.load(f)