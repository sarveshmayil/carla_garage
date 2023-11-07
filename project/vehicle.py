import carla
import numpy as np
import torch
from math import sqrt
import cv2
import open3d as o3d
from matplotlib import cm
from agents.tools.misc import get_speed

from config import Config, GlobalConfig
from control.controller_base import BaseController
from control.pid_vehicle_control import PIDController
from utils.misc import draw_waypoints
from utils.lidar import *
from model.model import LidarCenterNet
from leaderboard.envs.sensor_interface import CallBack, SensorInterface
from GlobalConfig import GlobalConfig

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
        self.model_config = GlobalConfig()

        self._sensor_interface = SensorInterface()
        self._sensors = {}

        model_config = ModelConfig().get()
        self._model = LidarCenterNet(model_config).to("cuda")
        self._model.eval()
        state_dict = torch.load(r"D:\Projects\av-project\carla_garage\pretrained_models\leaderboard\tfpp_wp_all_0\model_0030.pth")
        self._model.load_state_dict(state_dict)

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

        self.setup_sensors()
        self._world.tick()

    def get_sensor_config(self):
        sensors = []
        for cam in self.vehicle_config.cameras:
            sensors.append(cam)
        if self.vehicle_config.lidar is not None:
            sensors.append(self.vehicle_config.lidar)
        return sensors

    def setup_sensors(self):
        bp_library = self._world.get_blueprint_library()

        for sensor_spec in self.get_sensor_config():
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

            # create sensor
            sensor_transform = carla.Transform(sensor_location, sensor_rotation)
            sensor = self._world.spawn_actor(bp, sensor_transform, attach_to=self._vehicle)
            sensor.listen(CallBack(sensor_spec['id'], sensor_spec['type'], sensor, self._sensor_interface))
            self._sensors[sensor_spec['id']] = sensor

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

    def waypoint_to_bev(self, waypoint, inverse=False):
        '''
        in: numpy [3,] in world frame
        out: torch [2,] (x,y) in bev frame
        '''
        out = waypoint[:2]
        yaw = np.deg2rad(self._vehicle.get_transform().rotation.yaw)
        vehicle_pos = [self._vehicle.get_transform().location.x, self._vehicle.get_transform().location.y]
        rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)],
                                        [np.sin(yaw),  np.cos(yaw)]])   
        if not inverse:
            out = rotation_matrix @ (out - vehicle_pos).T
            out[0] = -out[0]
            out = torch.tensor(out).to("cuda").to(torch.float32)
        else:
            out = out.detach().cpu().numpy()
            out[0] = -out[0]
            out = np.linalg.inv(rotation_matrix) @ out + vehicle_pos

        return out
    

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

        # # PointCloud visualization
        # pcl_vis = o3d.visualization.Visualizer()
        # #pcl_vis.get_render_option().point_size = 1
        # # pcl_vis.get_render_option().show_coordinate_frame = True
        # pcl_vis_control = o3d.visualization.ViewControl()
        # pcl_vis.create_window(height=480, width=640)
        # pcl = o3d.geometry.PointCloud()
        # pcl.points = o3d.utility.Vector3dVector(np.random.rand(10,3))
        # pcl_vis.add_geometry(pcl)

        # # Visualize xyz axes
        # axis = o3d.geometry.LineSet()
        # axis.points = o3d.utility.Vector3dVector(np.array([
        #     [0.0, 0.0, 0.0],
        #     [1.0, 0.0, 0.0],
        #     [0.0, 1.0, 0.0],
        #     [0.0, 0.0, 1.0]]))
        # axis.lines = o3d.utility.Vector2iVector(np.array([
        #     [0, 1],
        #     [0, 2],
        #     [0, 3]]))
        # axis.colors = o3d.utility.Vector3dVector(np.array([
        #     [1.0, 0.0, 0.0],
        #     [0.0, 1.0, 0.0],
        #     [0.0, 0.0, 1.0]]))
        # pcl_vis.add_geometry(axis)

        lidar_buffer = np.empty((0,4))
        while True:
            self._world.tick()
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
            
            # Collect data and post-process
            data_dict = self._sensor_interface.get_data()
            out_data = self.data_tick(data_dict)
            cv2.imshow("camera", data_dict['rgb_front'][1])
            cv2.waitKey(1)
            
            # Handling of lidar buffer, display
            lidar_buffer = np.vstack((lidar_buffer, out_data['lidar']))
            if lidar_buffer.shape[0] >= self.vehicle_config.lidar['buffer_threshold']:
                # norm_lidar_buffer = lidar_buffer / np.max(abs(lidar_buffer))
                # pcl.points = o3d.utility.Vector3dVector(norm_lidar_buffer)

                # pcl_vis.update_geometry(pcl)
                # pcl_vis.poll_events()
                # pcl_vis.update_renderer()

                lidar_bev = lidar_to_bev(lidar_buffer, ranges=[(0,32), (-16,16), (-2,10)], res=0.125, visualize=True)
                lidar_histogram = lidar_to_histogram_features(lidar_buffer, self.model_config)[None,:]
                lidar_buffer = np.empty((0,4))

                control = torch.tensor([4,4])
                ego_vel = get_speed(self._vehicle)

                
                waypoint_fuck = np.array([target_wp.transform.location.x,target_wp.transform.location.y,target_wp.transform.location.z])

                #
                vehicle_transform = self._vehicle.get_transform()
                bev_waypoint = self.waypoint_to_bev(waypoint_fuck)
                preds = self._model(out_data['rgb'].permute(0,3,1,2).to("cuda"), torch.tensor(lidar_histogram).to("cuda"), target_point=bev_waypoint, 
                                        ego_vel=torch.tensor(ego_vel).reshape(1,1).to("cuda"), command = torch.tensor(control).to("cuda"))
                pred_wp = preds[0][0]
                for wp in pred_wp:
                    world_wp = self.waypoint_to_bev(wp, inverse=True)
                    begin = carla.Location(x=world_wp[0], y=world_wp[1])
                    end = begin + carla.Location(z=2)
                    self._world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=0)

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



class ModelConfig():
    def __init__(self) -> None:
        import json
        import pickle
        with open(r"D:\Projects\av-project\carla_garage\pretrained_models\leaderboard\tfpp_wp_all_0\args.txt") as file:
            stuff = json.load(file)
        with open(r"D:\Projects\av-project\carla_garage\pretrained_models\leaderboard\tfpp_wp_all_0\config.pickle",'rb') as file:
            loaded_config = pickle.load(file)

        self.model_config = GlobalConfig()
        self.model_config.__dict__.update(loaded_config.__dict__)

        for k, v in stuff.items():
            if not isinstance(v, str):
                exec(f'self.model_config.{k} = {v}')

        #self.model_config.use_discrete_command= 0

    def get(self):
        return self.model_config



            

