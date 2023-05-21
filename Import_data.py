import numpy as np
from wfdb.io.record import rdsamp


def import_data():
    # Import original signals
    # print("records_list:")
    records_list = []
    records = np.loadtxt("mit-bih-arrhythmia-database-1.0.0/RECORDS", dtype=int)
    print(len(records))
    for record in records:
        for ch in range(2):
            sig, fields = rdsamp('mit-bih-arrhythmia-database-1.0.0/' + str(record), channels=[ch])
            records_list = records_list + np.split(sig[:649980], 30)  # 648000
    # print(len(records_list), np.array(records_list)[:, :, 0].shape, type(records_list))
    # np.savetxt("records.csv", np.array(records_list)[:, :, 0], delimiter=",")
    # with open('test.npy', 'wb') as f:
    #     np.save(f, np.array(records_list)[:, :, 0])
    # with open('test.npy', 'rb') as f:
    #     a = np.load(f)

    # Import noisy signals
    # print("noisy_records_list:")
    noisy_records_list = []
    for record in records:
        for ch in range(2):
            sig, fields = rdsamp('mit-bih-arrhythmia-database-1.0.0/' + str(record) + "e24", channels=[ch])
            noisy_records_list = noisy_records_list + np.split(sig[:649980], 30)
    # print(len(noisy_records_list))
    return records_list, noisy_records_list
