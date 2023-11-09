import carla
import os
import json
import pickle
import numpy as np
import torch
import cv2

from vehicle import Vehicle
from config import GlobalConfig
from model.tf_model import LidarCenterNet
from utils.misc import draw_waypoints
from utils.lidar import *
from agents.tools.misc import get_speed

from typing import Optional


class Agent(Vehicle):
    def __init__(self, world:carla.World,  vehicle:Optional[carla.Actor]=None, device='cuda') -> None:
        super().__init__(world, vehicle, device)

        self.model_config = GlobalConfig()
        with open(os.path.join(self.vehicle_config.model["dir"], self.vehicle_config.model["args"])) as file:
            args = json.load(file)
        with open(os.path.join(self.vehicle_config.model["dir"], self.vehicle_config.model["config"]),'rb') as file:
            loaded_config = pickle.load(file)
        self.model_config.__dict__.update(loaded_config.__dict__)

        for k, v in args.items():
            if not isinstance(v, str):
                exec(f'self.model_config.{k} = {v}')
        
        self.model_config.use_our_own_config()

        self._model = LidarCenterNet(self.model_config).to(self.device)
        self._model.eval()
        state_dict = torch.load(os.path.join(self.vehicle_config.model["dir"], self.vehicle_config.model["weights"]))
        self._model.load_state_dict(state_dict)

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

        tp_idx = 0
        small_wp_idx = 1
        target_wp = self.route[tp_idx]
        veh_dist = self.dist(target_wp)
        small_waypoints = np.repeat(np.array([self._vehicle.get_transform().location.x, self._vehicle.get_transform().location.y])[None,:], 8, axis=0)
        lidar_buffer = np.empty((0,4))

        while True:
            self._world.tick()

            if visualize:
                draw_waypoints(self._world, [target_wp])
            
            # Offset spectator camera to follow car
            spectator_offset = -10 * self._vehicle.get_transform().rotation.get_forward_vector() + \
                                5 * self._vehicle.get_transform().rotation.get_up_vector()
            spectator_transform = self._vehicle.get_transform()
            spectator_transform.location += spectator_offset
            self._world.get_spectator().set_transform(spectator_transform)
            
            # Collect data and post-process
            data_dict = self._sensor_interface.get_data()
            out_data = self.data_tick(data_dict)

            if visualize:
                cv2.imshow("camera", data_dict['rgb_front'][1])
                cv2.waitKey(1)
    
            # Handling of lidar buffer, display
            lidar_buffer = np.vstack((lidar_buffer, out_data['lidar']))
            if lidar_buffer.shape[0] >= self.vehicle_config.lidar['buffer_threshold']:
                lidar_histogram = lidar_to_histogram_features(lidar_buffer[:,:3], self.model_config)
                lidar_buffer = np.empty((0,4))

                if visualize:
                    cv2.imshow("histogram", lidar_histogram[0])
                    cv2.waitKey(1)
                
                if self.dist(small_waypoints[small_wp_idx]) < threshold: #replan                    
                    ego_vel = get_speed(self._vehicle)
                    freeze_vehicle_transform = self._vehicle.get_transform()
                    
                    waypoint = np.array([target_wp.transform.location.x, target_wp.transform.location.y, target_wp.transform.location.z])
                    bev_waypoint = self.waypoint_to_bev(waypoint, freeze_vehicle_transform)
                    preds = self._model(out_data['rgb'].to(self.device),
                                        torch.tensor(lidar_histogram).unsqueeze(0).to(self.device),
                                        target_point=bev_waypoint, 
                                        ego_vel=torch.tensor(ego_vel).reshape(1,1).to(self.device))
                    pred_wp = preds[0][0]

                    small_waypoints = self.waypoint_to_bev(pred_wp, freeze_vehicle_transform, inverse=True)

                    if visualize:
                        for ind in range(small_waypoints.shape[0]):
                            point = carla.Location(x=float(small_waypoints[ind][0]), y=float(small_waypoints[ind][1]), z=1)
                            self._world.debug.draw_point(point, size=0.2, life_time=1.0)

            # Get and apply control initially
            next_wp = small_waypoints[small_wp_idx]
            next_wp = carla.Transform(carla.Location(float(next_wp[0]), float(next_wp[1]), 0.0))
            control = self._controller.get_control((target_speed, next_wp))
            self._vehicle.apply_control(control)

            veh_dist = self.dist(target_wp)
            # If vehicle has reached waypoint, move to next waypoint
            if(veh_dist < threshold):
                tp_idx += 1

                # Break once reaching last checkpoint
                if (tp_idx == n_wps):
                    break
                else:
                    target_wp = self.route[tp_idx]

        # Stop vehicle at final waypoint
        control = self._controller.get_control((0.0, self.route[-1]))
        self._vehicle.apply_control(control)
