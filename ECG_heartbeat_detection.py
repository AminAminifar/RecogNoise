from ecgdetectors import Detectors


# detect r peaks
fs = 360  # sample freq
detectors = Detectors(fs)


def r_peak_detect(in_array, algorithm):
    string_script = 'detectors.' + algorithm + '(in_array)'

    result = None
    try:
        result = eval(string_script)  # exec(string_script)
        out_flag = True
    except IndexError:
        out_flag = False

    return out_flag, result
