import numpy as np
import matplotlib.pyplot as plt


wp = np.load("waypoints.npy")
N = wp.shape[0]
c = ['r'] * N
small_size = 0.5
big_size = 15
s = [small_size] * N
s[0] = big_size
s[-1] = big_size
for i in range(len(s)):
    if i % 20 == 0:
        s[i] = big_size
print(s)

targ30 = np.load("veh_pos_30.npy")
targ45 = np.load("veh_pos_45.npy")
targ60 = np.load("veh_pos_60.npy")
targ75 = np.load("veh_pos_75.npy")
targ90 = np.load("veh_pos_90.npy")
targ200 = np.load("veh_pos_200.npy")

print(wp.shape)
print(targ30.shape)

def match_to_waypoints(wp, pos):
    start = wp[0, :]
    end = wp[-1, :]

    start_ind = np.argmin(np.linalg.norm(pos - start, axis=1))
    end_ind = np.argmin(np.linalg.norm(pos - end, axis=1))

    return pos[start_ind:end_ind, :]

# plt.plot(targ30[:, 0], targ30[:, 1], 'black', linewidth=6)
# plt.plot(targ45[:, 0], targ45[:, 1], 'lime', linewidth=4)
# plt.plot(targ60[:, 0], targ60[:, 1], 'cyan', linewidth=3)
# plt.plot(targ75[:, 0], targ75[:, 1], 'magenta', linewidth=2)
# plt.plot(targ90[:, 0], targ90[:, 1], 'orange', linewidth=1.5)
# # plt.plot(targ200[:, 0], targ200[:, 1], 'blueviolet')
# plt.scatter(wp[:, 0], wp[:, 1], c = c, s = s, zorder=10)
# plt.legend(["target speed = 30", "target speed = 45", "target speed = 60", "target speed = 75", "target speed = 90", "waypoints"])
# plt.xlabel("X (m)")
# plt.ylabel("Y (m)")
# plt.title("PID Waypoint Following")
# # plt.axis('equal')
# # plt.legend(["target speed = 30", "target speed = 45", "target speed = 60", "target speed = 75", "target speed = 90", "target speed = 200", "waypoints"])
# plt.show()

# plt.scatter(wp[:, 0], wp[:, 1], c=c, s=s)
# plt.title("Waypoints Generated in CARLA")
# plt.xlabel("X (m)")
# plt.ylabel("Y (m)")
# plt.savefig("./plots/waypoints.png")
# plt.show()

# plt.scatter(wp[:, 0], wp[:, 1], c=c, s=s, zorder=10)
# plt.plot(targ30[:, 0], targ30[:, 1], 'black', linewidth=4)
# plt.title("CARLA PID Waypoint Following")
# plt.xlabel("X (m)")
# plt.ylabel("Y (m)")
# plt.text(wp[0, 0], wp[0, 1]-10, 'Start', fontsize=8, ha='center', va='bottom', color='black')
# plt.text(wp[-1, 0], wp[-1, 1]-10, 'End', fontsize=8, ha='center', va='bottom', color='black')
# plt.legend(['waypoints', 'vehicle trajectory (target speed = 30 kph)'])
# plt.savefig("./plots/waypoint_following.png")
# plt.show()


#=================================
plt.subplot(1, 2, 1)
inds = (40, 81)
wp_new = wp[inds[0]:inds[1], :]
plt.scatter(wp_new[:, 0], wp_new[:, 1], c=c[inds[0]:inds[1]], s=s[inds[0]:inds[1]], zorder=10)
################
targ30_new = match_to_waypoints(wp_new, targ30)
plt.plot(targ30_new[:, 0], targ30_new[:, 1], 'black', linewidth=4)
################
targ45_new = match_to_waypoints(wp_new, targ45)
plt.plot(targ45_new[:, 0], targ45_new[:, 1], 'lime', linewidth=3)
################
targ60_new = match_to_waypoints(wp_new, targ60)
plt.plot(targ60_new[:, 0], targ60_new[:, 1], 'cyan', linewidth=2)
################
targ75_new = match_to_waypoints(wp_new, targ75)
plt.plot(targ75_new[:, 0], targ75_new[:, 1], 'magenta', linewidth=2)
################
targ90_new = match_to_waypoints(wp_new, targ90)
plt.plot(targ90_new[:, 0], targ90_new[:, 1], 'orange', linewidth=2)
################
plt.axis([wp_new[:, 0].min() - 20, wp_new[:, 0].max() + 20, 320, 330])
plt.title("Straight")
plt.legend(["waypoints", "target = 30 kph", "target = 45 kph", "target = 60 kph", "target = 75 kph", "target = 90 kph"])
plt.text(wp_new[0, 0], wp_new[0, 1]+0.1, 'Start', fontsize=8, ha='center', va='bottom', color='black')
plt.text(wp_new[-1, 0], wp_new[-1, 1]+0.1, 'End', fontsize=8, ha='center', va='bottom', color='black')
plt.xlabel("X (m)")
plt.ylabel("Y (m)")

#=================================
plt.subplot(1, 2, 2)
inds = (120, 181)
wp_new = wp[inds[0]:inds[1], :]
plt.scatter(wp_new[:, 0], wp_new[:, 1], c=c[inds[0]:inds[1]], s=s[inds[0]:inds[1]], zorder=10)
################
targ30_new = match_to_waypoints(wp_new, targ30)
plt.plot(targ30_new[:, 0], targ30_new[:, 1], 'black', linewidth=4)
################
targ45_new = match_to_waypoints(wp_new, targ45)
plt.plot(targ45_new[:, 0], targ45_new[:, 1], 'lime', linewidth=3)
################
targ60_new = match_to_waypoints(wp_new, targ60)
plt.plot(targ60_new[:, 0], targ60_new[:, 1], 'cyan', linewidth=2)
################
targ75_new = match_to_waypoints(wp_new, targ75)
plt.plot(targ75_new[:, 0], targ75_new[:, 1], 'magenta', linewidth=2)
################
targ90_new = match_to_waypoints(wp_new, targ90)
plt.plot(targ90_new[:, 0], targ90_new[:, 1], 'orange', linewidth=2)
################
plt.axis('equal')
plt.title("Turn")
plt.text(wp_new[0, 0]-0.6, wp_new[0, 1]+1, 'Start', fontsize=8, ha='center', va='bottom', color='black')
plt.text(wp_new[-1, 0], wp_new[-1, 1]-5, 'End', fontsize=8, ha='center', va='bottom', color='black')
plt.xlabel("X (m)")
plt.ylabel("Y (m)")

plt.suptitle("PID Controller with Varying Target Speeds")
plt.show()


