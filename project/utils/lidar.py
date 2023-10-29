import numpy as np
import matplotlib.pyplot as plt

def lidar_to_bev(pointcloud, ranges=[(0,20), (-10,10), (-2,10)], res=0.05, visualize=False):
    """
    Convert pointcloud to birds-eye-view image

    Args:
        poincloud: numpy array of shape (N,3)
        ranges: List of tuples of x,y,z ranges to consider
        res: point resolution [m]
        visualize: flag to display output BEV image

    Returns:
        BEV image
    """
    x_mask = np.logical_and((pointcloud[:,0] > ranges[0][0]), (pointcloud[:,0] < ranges[0][1]))
    y_mask = np.logical_and((pointcloud[:,1] > ranges[1][0]), (pointcloud[:,1] < ranges[1][1]))
    mask = np.logical_and(x_mask, y_mask)
    indices = np.argwhere(mask).flatten()

    points_in_frame = pointcloud[indices,:]

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-points_in_frame[:,1] / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (-points_in_frame[:,0] / res).astype(np.int32)  # y axis is -x in LIDAR
    x_img -= int(np.floor(ranges[1][0] / res))
    y_img += int(np.ceil(ranges[0][1] / res))
    
    # Get height values
    heights = np.clip(points_in_frame[:,2], ranges[2][0], ranges[2][1])
    heights = (((heights - ranges[2][0]) / float(ranges[2][1] - ranges[2][0])) * 255).astype(np.uint8)

    x_max = 1+int((ranges[1][1] - ranges[1][0])/res)
    y_max = 1+int((ranges[0][1] - ranges[0][0])/res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[y_img, x_img] = heights

    if visualize:
        plt.imshow(im, cmap='gray', vmin=0, vmax=255)
        plt.show()

    return im

def lidar_to_ego_coordinates(lidar, lidar_pos=np.zeros(3), lidar_rot=np.zeros(3), intensity=False):
    """
    Converts the LiDAR points given by the simulator into the ego agents
    coordinate system
    :param lidar: the LiDAR point cloud as provided in the input of run_step with shape (N,4)
    :return: lidar where the points are w.r.t. 0/0/0 of the car and the carla
    coordinate system with shape (N,4)
    """
    yaw = np.deg2rad(lidar_rot[2])
    rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                                [np.sin(yaw),  np.cos(yaw), 0.0],
                                [        0.0,          0.0, 1.0]])
    if not isinstance(lidar_pos, np.ndarray):
        lidar_pos = np.array(lidar_pos)

    # The double transpose is a trick to compute all the points together.
    ego_lidar = (rotation_matrix @ lidar[1][:, :3].T).T + lidar_pos
    ego_lidar[:, :1] = -ego_lidar[:, :1]

    if intensity:
        ego_lidar = np.hstack((ego_lidar, lidar[1][:,-1]))

    return ego_lidar