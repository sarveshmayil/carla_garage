import carla
import numpy as np
import torch
from math import sqrt
import cv2
import open3d as o3d
from matplotlib import cm

from project.config import Config
from project.control.controller_base import BaseController
from project.control.pid_vehicle_control import PIDController
from project.utils.misc import draw_waypoints
from project.utils.lidar import *
from leaderboard.leaderboard.envs.sensor_interface import CallBack, SensorInterface, SpeedometerReader

from typing import Tuple, Union, Optional, Dict

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

class Vehicle():
    """
    Vehicle class for carla vehicle.

    Includes vehicle controller and planner.
    """
    def __init__(self, world:carla.World, vehicle:Optional[carla.Actor]=None, device='cuda') -> None:
        self._world:carla.World = world
        self._vehicle:carla.Actor = vehicle
        self.device = device
        self.route = []

        self.vehicle_config = Config()

        self._sensor_interface = SensorInterface()
        self._sensors = {}

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
        vehicle_name="vehicle.tesla.model3"
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
        blueprint:carla.ActorBlueprint = blueprint_library.find(vehicle_name)
        self._vehicle = self._world.spawn_actor(blueprint, spawnPoint)

        sensor_list = self.get_sensor_config()
        self.setup_sensors(sensor_list)
        self._world.tick()

    def get_sensor_config(self):
        sensors = []
        for cam in self.vehicle_config.cameras:
            sensors.append(cam)
        if self.vehicle_config.lidar is not None:
            sensors.append(self.vehicle_config.lidar)
        if self.vehicle_config.other_sensors is not None:
            sensors += self.vehicle_config.other_sensors
        return sensors

    def setup_sensors(self, sensor_list):
        bp_library = self._world.get_blueprint_library()

        for sensor_spec in sensor_list:
            if sensor_spec['type'].startswith('sensor.speedometer'):
                delta_time = self._world.get_settings().fixed_delta_seconds
                frame_rate = 1 / delta_time
                sensor = SpeedometerReader(self._vehicle, frame_rate)
            else:
                bp = bp_library.find(str(sensor_spec['type']))
                if sensor_spec['type'].startswith('sensor.camera'):
                    bp.set_attribute('image_size_x', str(sensor_spec['size'][0]))
                    bp.set_attribute('image_size_y', str(sensor_spec['size'][1]))
                    bp.set_attribute('fov', str(sensor_spec['fov']))
                    bp.set_attribute('lens_circle_multiplier', str(3.0))
                    bp.set_attribute('lens_circle_falloff', str(3.0))
                    bp.set_attribute('chromatic_aberration_intensity', str(0.5))
                    bp.set_attribute('chromatic_aberration_offset', str(0))
                    sensor_location = carla.Location(*sensor_spec['position'])
                    sensor_rotation = carla.Rotation(*sensor_spec['rotation'])
                elif sensor_spec['type'].startswith('sensor.lidar'):
                    bp.set_attribute('range', str(85))
                    bp.set_attribute('rotation_frequency', str(sensor_spec['rot_freq']))
                    bp.set_attribute('channels', str(64))
                    bp.set_attribute('upper_fov', str(10))
                    bp.set_attribute('lower_fov', str(-30))
                    bp.set_attribute('points_per_second', str(sensor_spec['points_per_sec']))
                    bp.set_attribute('atmosphere_attenuation_rate', str(0.004))
                    bp.set_attribute('dropoff_general_rate', str(0.45))
                    bp.set_attribute('dropoff_intensity_limit', str(0.8))
                    bp.set_attribute('dropoff_zero_intensity', str(0.4))
                    sensor_location = carla.Location(*sensor_spec['position'])
                    sensor_rotation = carla.Rotation(*sensor_spec['rotation'])
                elif sensor_spec['type'].startswith('sensor.other.radar'):
                    bp.set_attribute('horizontal_fov', str(sensor_spec['fov']))  # degrees
                    bp.set_attribute('vertical_fov', str(sensor_spec['fov']))  # degrees
                    bp.set_attribute('points_per_second', '1500')
                    bp.set_attribute('range', '100')  # meters
                    sensor_location = carla.Location(*sensor_spec['position'])
                    sensor_rotation = carla.Rotation(*sensor_spec['rotation'])
                elif sensor_spec['type'].startswith('sensor.other.gnss'):
                    bp.set_attribute('noise_alt_stddev', str(0.000005))
                    bp.set_attribute('noise_lat_stddev', str(0.000005))
                    bp.set_attribute('noise_lon_stddev', str(0.000005))
                    bp.set_attribute('noise_alt_bias', str(0.0))
                    bp.set_attribute('noise_lat_bias', str(0.0))
                    bp.set_attribute('noise_lon_bias', str(0.0))
                    sensor_location = carla.Location(*sensor_spec['position'])
                    sensor_rotation = carla.Rotation()
                elif sensor_spec['type'].startswith('sensor.other.imu'):
                    bp.set_attribute('noise_accel_stddev_x', str(0.001))
                    bp.set_attribute('noise_accel_stddev_y', str(0.001))
                    bp.set_attribute('noise_accel_stddev_z', str(0.015))
                    bp.set_attribute('noise_gyro_stddev_x', str(0.001))
                    bp.set_attribute('noise_gyro_stddev_y', str(0.001))
                    bp.set_attribute('noise_gyro_stddev_z', str(0.001))
                    sensor_location = carla.Location(*sensor_spec['position'])
                    sensor_rotation = carla.Rotation(*sensor_spec['rotation'])

                # create sensor
                sensor_transform = carla.Transform(sensor_location, sensor_rotation)
                sensor = self._world.spawn_actor(bp, sensor_transform, attach_to=self._vehicle)
            sensor.listen(CallBack(sensor_spec['id'], sensor_spec['type'], sensor, self._sensor_interface))
            self._sensors[sensor_spec['id']] = sensor

            self._world.tick()

    def dist(self, target:Union[np.ndarray,carla.Waypoint,carla.Transform,carla.Location]) -> float:
        """
        Determines distance between vehicle and target location.

        Args:
            target: target location of type Union[np.ndarray,carla.Waypoint,carla.Transform,carla.Location]

        Returns:
            distance to target [m]
        """
        if isinstance(target, carla.Waypoint):
            target_loc = target.transform.location
            target_loc = np.array([target_loc.x, target_loc.y])
        elif isinstance(target, carla.Transform):
            target_loc = target.location
            target_loc = np.array([target_loc.x, target_loc.y])
        elif isinstance(target, carla.Location):
            target_loc = np.array([target.x, target.y])
        elif isinstance(target, np.ndarray):
            target_loc = target[:2]
        else:
            raise ValueError("target must be of type [np.ndarray,carla.Waypoint,carla.Transform,carla.Location]")

        vehicle_loc = self.location
        dist = sqrt((target_loc[0] - vehicle_loc.x)**2 + (target_loc[1] - vehicle_loc.y)**2)
        return dist

    def set_controller_pid(self, lat_args:Dict[str, float]=None, long_args:Dict[str, float]=None):
        if lat_args is None:
            args_lateral_dict = self.vehicle_config.pid["lateral"]

        if long_args is None:
            args_long_dict = self.vehicle_config.pid["longitudinal"]

        self._controller = PIDController(self._vehicle, lateral_args=args_lateral_dict, longitudinal_args=args_long_dict)

    def set_route(self, target:carla.Location, start:Optional[carla.Location]=None):
        """
        Function to set a route for the vehicle to follow.
        
        !!! currently only works with carla'a GlobalRoutePlanner !!!

        Args:
            target: target location for the vehicle to reach.
        
        Returns:
            None
        """
        if start is None:
            start = self.location
        self.route = [wp[0] for wp in self._planner.trace_route(start, target)]

    def waypoint_to_bev(self, waypoint:np.ndarray, vehicle_transform:carla.Transform, inverse:bool=False):
        '''
        Args:
            waypoint: numpy array with [x,y,...] coordinates in world/BEV frame
            vehicle_transform: vehicle's transform
            inverse: True=world->BEV ; False=BEV->world
        
        Returns:
            coordinates in world/BEV frame
                if inverse: return type np.ndarray
                otherwise: return type torch Tensor
        '''
        ndim = waypoint.ndim
        if ndim == 1:
            waypoint = np.expand_dims(waypoint, 0)
        out = waypoint[:,:2]
        yaw = -np.deg2rad(vehicle_transform.rotation.yaw)
        vehicle_pos = np.array([vehicle_transform.location.x, vehicle_transform.location.y])
        rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)],
                                    [np.sin(yaw),  np.cos(yaw)]])   
        if not inverse:
            out = (rotation_matrix @ (out - vehicle_pos).T).T
            out = torch.tensor(out).to(self.device).to(torch.float32)
        else:
            out = out.detach().cpu().numpy()
            out = (np.linalg.inv(rotation_matrix) @ out.T).T + vehicle_pos

        if ndim == 1:
            out = out[0]

        return out
    
    def follow_route(self, target_speed=30.0, threshold=3.5, visualize=False, debug=False):
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

        i = 0
        target_wp = self.route[0]

        while True:
            self._world.tick()

            if debug:
                draw_waypoints(self._world, [target_wp])

            # Offset spectator camera to follow car
            spectator_offset = -10 * self._vehicle.get_transform().rotation.get_forward_vector() + \
                                5 * self._vehicle.get_transform().rotation.get_up_vector()
            spectator_transform = self._vehicle.get_transform()
            spectator_transform.location += spectator_offset
            self._world.get_spectator().set_transform(spectator_transform)

            # Get and apply control initially
            control = self._controller.get_control((target_speed, target_wp))
            self._vehicle.apply_control(control)
            veh_dist = self.dist(target_wp)

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

    @torch.inference_mode()
    def data_tick(
        self,
        data_dict:Dict[str, Union[carla.libcarla.Image, carla.libcarla.LidarMeasurement]]
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Post-process image, lidar data.

        Adds jpg artifacts to image data, formats as torch tensor \\
        Converts lidar scan to ego vehicle coordinate frame
        """
        out_data = {}
        rgb = []
        for id in self._sensors.keys():
            if id.startswith('rgb'):
                image = data_dict[id][1][:,:,:3]
                # # Also add jpg artifacts at test time, because the training data was saved as jpg.
                # _, compressed_image = cv2.imencode('.jpg', image)
                # image = cv2.imdecode(compressed_image, cv2.IMREAD_UNCHANGED)
                # rgb_pos = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # # Switch to pytorch channel first order
                # rgb_pos = np.transpose(rgb_pos, (2, 0, 1))
                # rgb.append(rgb_pos)
                image = np.transpose(image, (2,0,1))
                rgb.append(image)
            elif id.startswith('lidar'):
                lidar_points = lidar_to_ego_coordinates(data_dict[id],
                                                        lidar_pos=self.vehicle_config.lidar['position'],
                                                        lidar_rot=self.vehicle_config.lidar['rotation'],
                                                        intensity=True)
                # intensity = lidar_points[:,-1]
                # intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 85))
                # int_color = np.c_[
                #     np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
                #     np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
                #     np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]
                # out_data['lidar'] = np.concatenate((lidar_points[:,:3], int_color), axis=1)
                
                
                out_data['lidar'] = lidar_points

        rgb = np.concatenate(rgb, axis=1)
        rgb = torch.from_numpy(rgb).to(self.device, dtype=torch.float32).unsqueeze(0)
        out_data['rgb'] = rgb

        return out_data

    def __del__(self):
        self._vehicle.destroy()
        for id in self._sensors.keys():
            if self._sensors[id] is not None:
                self._sensors[id].stop()
                self._sensors[id].destroy()
                self._sensors[id] = None
        self._sensors = {}
