from wfdb.io.record import rdsamp
import matplotlib.pyplot as plt

sig, fields = rdsamp('data/118e24', channels=[0])
print(len(sig))
print(fields)
plt.plot(sig)
plt.show()



