import numpy as np

records = np.loadtxt("signals_including_noise/RECORDS", dtype=int)

script_list = []
for record in records:
    # print(record)
    # print(f"nst -i {record:03d} em -o {record:03d}e24 -s 24 -F 212")
    # script_list.append(f"nst -i {record:03d} em -o {record:03d}e24 -s 24 -F 212")
    script_list.append(f"nst -i mitbih/{record:03d} nstdb/em -o {record:03d}e_6 -s -6")

# print(script_list)
with open('signals including noise/script_all_e_6.txt', mode='wt', encoding='utf-8') as myfile:
    myfile.write('\n'.join(script_list))