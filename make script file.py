import numpy as np

records = np.loadtxt("signals_including_noise/RECORDS", dtype=int)
em_str_list = ['e_6', 'e00', 'e06', 'e12', 'e18', 'e24']
bw_str_list = ['b_6', 'b00', 'b06', 'b12', 'b18', 'b24']
ma_str_list = ['m_6', 'm00', 'm06', 'm12', 'm18', 'm24']
snr_str_list = [' -s -6', ' -s 0', ' -s 6 -F 212', ' -s 12 -F 212', ' -s 18 -F 212', ' -s 24 -F 212']
#
# noise_str = ''

script_list = []
for noise_str, snr_str in zip(ma_str_list, snr_str_list):
    script_list = []
    print(noise_str, snr_str)
    for record in records:
        # print(record)
        # print(f"nst -i {record:03d} em -o {record:03d}e24 -s 24 -F 212")
        # script_list.append(f"nst -i {record:03d} em -o {record:03d}e24 -s 24 -F 212")
        # script_list.append(f"nst -i mitbih/{record:03d} nstdb/em -o {record:03d}e00 -s 0")
        script_list.append(f"nst -i mitbih/{record:03d} nstdb/ma -o {record:03d}" + noise_str + snr_str)

    # print(script_list)
    # 'signals_including_noise/e06/script_all_e06.txt'
    address_str = 'signals_including_noise/' + noise_str + '/script_all_' + noise_str + '.txt'
    with open(address_str, mode='wt', encoding='utf-8') as myfile:  # script_all_e_6.txt
        myfile.write('\n'.join(script_list))