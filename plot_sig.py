from wfdb.io.record import rdsamp
import matplotlib.pyplot as plt


# sig, fields = rdsamp('data/em', channels=[0])  # 118e24
# sig, fields = rdsamp('data/118e_6', channels=[0])  # 118e24
# sig, fields = rdsamp('mit-bih-arrhythmia-database-1.0.0/118e24', channels=[0])  # 118e24
sig, fields = rdsamp('signals_including_noise/mitbih_e_6/115e_6', channels=[0])
print(len(sig))
print(fields)
minute = 5
start = minute * 60 + 0
end = minute * 60 + 60
fs = 360
ax = plt.axes()
ax.set_facecolor('silver')
ax.set_xlabel("Time")
ax.set_ylabel("Amplitude")
plt.plot(sig[start*360:end*360])
plt.grid(True)  # , color='r', linestyle='-', linewidth=.2

plt.show()



