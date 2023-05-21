import numpy as np

records = np.loadtxt("non noisy data/mit-bih-arrhythmia-database-1.0.0/RECORDS", dtype=int)

script_list = []
for record in records:
    # print(record)
    # print(f"nst -i {record:03d} em -o {record:03d}e24 -s 24 -F 212")
    script_list.append(f"nst -i {record:03d} em -o {record:03d}e24 -s 24 -F 212")

# print(script_list)
with open('non noisy data/mit-bih-arrhythmia-database-1.0.0/script_all.txt', mode='wt', encoding='utf-8') as myfile:
    myfile.write('\n'.join(script_list))