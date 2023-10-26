import carla
import numpy as np
from math import sqrt
import cv2
import open3d as o3d

from control.controller_base import BaseController
from control.pid_vehicle_control import PIDController
from utils.misc import draw_waypoints

from typing import Tuple, Union, Optional, Dict

class Vehicle():
    """
    Vehicle class for carla vehicle.

    Includes vehicle controller and planner.
    """
    def __init__(self, world:carla.World, vehicle:Optional[carla.Actor]=None) -> None:
        self._world:carla.World = world
        self._vehicle:carla.Actor = vehicle
        self.route = []

    @property
    def location(self) -> carla.Location:
        return self._vehicle.get_location()
    
    @property
    def controller(self) -> BaseController:
        return self._controller
    
    @controller.setter
    def controller(self, controller:BaseController):
        self._controller = controller
    
    @property
    def planner(self):
        return self._planner
    
    @planner.setter
    def planner(self, planner):
        self._planner = planner

    def get_world(self) -> carla.World:
        return self._world

    def spawn(
        self,
        location:Union[Tuple[float], carla.Location],
        rotation:Union[Tuple[float], carla.Rotation]=(0.0, 0.0, 0.0),
        vehicle_idx=0
    ):
        """
        Spawns vehicle at specified location and rotation.

        Args:
            location: tuple of (x,y,z) location in world or a carla.Location object
            rotation: tuple of (r,p,y) rotation in world or a carla.Rotation object
            Vehicle_idx: vehicle index in world blueprint library

        Returns:
            None
        """
        if not isinstance(location, carla.Location):
            location = carla.Location(*location)
        if not isinstance(rotation, carla.Rotation):
            rotation = carla.Rotation(*rotation)

        spawnPoint = carla.Transform(location, rotation)

        blueprint_library = self._world.get_blueprint_library()
        blueprint:carla.ActorBlueprint = blueprint_library.filter('vehicle.*')[vehicle_idx]
        self._vehicle = self._world.spawn_actor(blueprint, spawnPoint)

        camera_blueprint = blueprint_library.find('sensor.camera.rgb')
        camera_blueprint.set_attribute("image_size_x", "640")
        camera_blueprint.set_attribute("image_size_y", "480")
        camera_blueprint.set_attribute("fov", "110")
        camera_init_transform = carla.Transform(carla.Location(z=0.7, x=2.5))
        self._camera = self._world.spawn_actor(camera_blueprint,camera_init_transform ,attach_to=self._vehicle)

        lidar_blueprint = blueprint_library.find('sensor.lidar.ray_cast')
        lidar_init_transform = carla.Transform(carla.Location(z=2, x=0))
        self._lidar = self._world.spawn_actor(lidar_blueprint, lidar_init_transform ,attach_to=self._vehicle)

    def camera_callback(self, image, data_dict):
        i = np.reshape(np.array(image.raw_data), (480, 640, 4))
        data_dict["image"] = i


    def lidar_callback(self, scan, data_dict):
        point_cloud = np.array(scan.raw_data)
        xyzi = np.reshape(point_cloud, (-1,4))
        xyz = self.lidar_to_ego_coordinate(xyzi)
        data_dict["lidar"] = xyz


        


    def dist(self, target) -> float:
        """
        Determines distance between vehicle and target location.

        Args:
            target: target location

        Returns:
            distance to target [m]
        """
        vehicle_loc = self.location
        dist = sqrt((target.transform.location.x - vehicle_loc.x)**2 + (target.transform.location.y - vehicle_loc.y)**2)
        return dist

    def set_controller_pid(self, lat_args:Dict[str, float]=None, long_args:Dict[str, float]=None):
        if lat_args is None:
            args_lateral_dict = {
                'K_P': 1.95,
                'K_D': 0.2,
                'K_I': 0.07,
                'dt' : 0.1
            }

        if long_args is None:
            args_long_dict = {
                'K_P': 1.0,
                'K_D': 0.0,
                'K_I': 0.75,
                'dt' : 0.1
            }
        self._controller = PIDController(self._vehicle, lateral_args=args_lateral_dict, longitudinal_args=args_long_dict)

    def set_route(self, target:carla.Location):
        """
        Function to set a route for the vehicle to follow.
        
        !!! currently only works with carla'a GlobalRoutePlanner !!!

        Args:
            target: target location for the vehicle to reach.
        
        Returns:
            None
        """
        self.route = [wp[0] for wp in self._planner.trace_route(self.location, target)]

    def follow_route(self, target_speed=30.0, threshold=3.5, visualize=False):
        """
        Function to use controller to follow a route.

        Args:
            target_speed: vehicle's target speed [km/h]
            threshold: distance threshold to check if vehicle has reached waypoint [m]
            visualize: flag to visualize the route

        Returns:
            None
        """
        if self.route is None:
            raise Exception("No route was set. Use `set_route()` first.")
        
        n_wps = len(self.route)

        if visualize:
            draw_waypoints(self._world, self.route, life_time=n_wps*5.0)

        i = 0
        target_wp = self.route[0]

        pcl_vis = o3d.visualization.Visualizer()
        pcl_vis_control = o3d.visualization.ViewControl()
        pcl_vis.create_window(height=480, width=640)
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(np.random.rand(10,3))
        pcl_vis.add_geometry(pcl)

        sensor_data = {"image": np.zeros((480, 640, 4)), "lidar": pcl.points}
        self._camera.listen(lambda image: self.camera_callback(image, sensor_data))   
        self._lidar.listen(lambda scan: self.lidar_callback(scan, sensor_data))        

        while True:
            
            spectator_offset = -5 * self._vehicle.get_transform().rotation.get_forward_vector() + \
                            2 * self._vehicle.get_transform().rotation.get_up_vector()
            spectator_transform = self._vehicle.get_transform()
            spectator_transform.location += spectator_offset
            self.get_world().get_spectator().set_transform(spectator_transform)

            control = self._controller.get_control((target_speed, target_wp))
            self._vehicle.apply_control(control)
            veh_dist = self.dist(target_wp)
                
            cv2.imshow("", sensor_data['image'])
            cv2.waitKey(1)
            pcl.points = o3d.utility.Vector3dVector(sensor_data["lidar"])
            up_vector = np.array([self._vehicle.get_transform().rotation.get_up_vector().x, 
                                    self._vehicle.get_transform().rotation.get_up_vector().y,
                                    self._vehicle.get_transform().rotation.get_up_vector().z])
            pcl_vis_control.set_front(-up_vector)
            pcl_vis.update_geometry(pcl)
            pcl_vis.poll_events()
            pcl_vis.update_renderer()

            # If vehicle has reached waypoint, move to next waypoint
            if(veh_dist < threshold):
                control = self._controller.get_control((target_speed, target_wp))
                self._vehicle.apply_control(control)
                i += 1

                # Break once reaching last checkpoint
                if (i == n_wps):
                    break
                else:
                    target_wp = self.route[i]

        # Stop vehicle at final waypoint
        control = self._controller.get_control((0.0, self.route[-1]))
        self._vehicle.apply_control(control)


    def lidar_to_ego_coordinate(self, lidar):
        """
        Converts the LiDAR points given by the simulator into the ego agents
        coordinate system
        :param config: GlobalConfig, used to read out lidar orientation and location
        :param lidar: the LiDAR point cloud as provided in the input of run_step
        :return: lidar where the points are w.r.t. 0/0/0 of the car and the carla
        coordinate system.
        """
        vehicle_transform = self._vehicle.get_transform()
        yaw = vehicle_transform.rotation.yaw
        rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])

        translation = np.array([vehicle_transform.location.x, vehicle_transform.location.y, vehicle_transform.location.z]) + np.array([0, 0, 2])

        # The double transpose is a trick to compute all the points together.
        ego_lidar = np.matmul((lidar[:,:3] - translation), rotation_matrix[None,:,:]).squeeze(0)
        ego_lidar = ego_lidar/np.sqrt((np.sum(np.square(ego_lidar),axis=1)))[:,None]

        return ego_lidar

    def __del__(self):
        self._vehicle.destroy()
        self._camera.destroy()
        self._lidar.destroy()

            

