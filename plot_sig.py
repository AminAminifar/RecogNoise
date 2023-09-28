from wfdb.io.record import rdsamp
import matplotlib.pyplot as plt
import  numpy as np

# sig, fields = rdsamp('data/em', channels=[0])
# sig, fields = rdsamp('data/118e_6', channels=[0])
noise_str = 'e06'
sig, fields = rdsamp('signals_including_noise/'+noise_str+'/mitbih_'+noise_str+'/118'+noise_str, channels=[0])
# sig, fields = rdsamp('signals_including_noise/b_6/mitbih_b_6/115b_6', channels=[0])
print(len(sig))
print(fields)

minute = 6
start = minute * 60 + 0
end = minute * 60 + 5 # 60 #+ 60
fs = 360

ax = plt.axes()
# ax.set_facecolor('silver')
ax.set_xlabel("Time")
ax.set_ylabel("Amplitude")
plt.plot(sig[start*fs:end*fs])

# plt.grid(True, color='pink')  # , color='r', linestyle='-', linewidth=.2
# plt.show()



# major_ticks_x = np.arange(0, 3600, 72)
# minor_ticks_x = np.arange(0, 3600, 36)
#
# major_ticks = np.arange(-9, -3, .5)
# minor_ticks = np.arange(-9, -3, .25)

major_ticks_x = np.arange(-50, 1900, 40)
minor_ticks_x = np.arange(-50, 1900, 20)

# major_ticks = np.arange(-9, -3, .5)
# minor_ticks = np.arange(-9, -3, .25)
major_ticks = np.arange(-10, -2, .5)
minor_ticks = np.arange(-10, -2, .25)


ax.set_xticks(major_ticks_x)
ax.set_xticks(minor_ticks_x, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)


ax.grid(which='both', color='pink') #, sketch_params=.01
ax.grid(which='minor', alpha=0.3)
ax.grid(which='major', alpha=0.9)

# plt.xticks([])
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

plt.show()




