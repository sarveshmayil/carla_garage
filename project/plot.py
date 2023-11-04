import numpy as np
import matplotlib.pyplot as plt


wp = np.load("waypoints.npy")
N = wp.shape[0]
c = ['r'] * N
s = [0.3] * N
s[0] = 5
s[-1] = 5
for i in range(len(s)):
    if i % 20 == 0:
        s[i] = 5
print(s)

targ30 = np.load("veh_pos_30.npy")
targ45 = np.load("veh_pos_45.npy")
targ60 = np.load("veh_pos_60.npy")
targ75 = np.load("veh_pos_75.npy")
targ90 = np.load("veh_pos_90.npy")
targ200 = np.load("veh_pos_200.npy")



plt.plot(targ30[:, 0], targ30[:, 1], 'black')
plt.plot(targ45[:, 0], targ45[:, 1], 'green')
plt.plot(targ60[:, 0], targ60[:, 1], 'cyan')
plt.plot(targ75[:, 0], targ75[:, 1], 'magenta')
plt.plot(targ90[:, 0], targ90[:, 1], 'orange')
# plt.plot(targ200[:, 0], targ200[:, 1], 'blueviolet')
plt.scatter(wp[:, 0], wp[:, 1], c = c, s = s, zorder=10)
plt.legend(["target speed = 30", "target speed = 45", "target speed = 60", "target speed = 75", "target speed = 90", "waypoints"])
# plt.legend(["target speed = 30", "target speed = 45", "target speed = 60", "target speed = 75", "target speed = 90", "target speed = 200", "waypoints"])
plt.show()