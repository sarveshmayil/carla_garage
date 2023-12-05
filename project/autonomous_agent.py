import carla
import os
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from vehicle import Vehicle
from config import GlobalConfig
from model.tf_model_minimal import LidarCenterNet
from utils.misc import draw_waypoints
from utils.lidar import *
from utils.image_noise import *
from agents.tools.misc import get_speed

from typing import Optional, Dict, Union


class Agent(Vehicle):
    def __init__(self, world:carla.World,  vehicle:Optional[carla.Actor]=None, data_listener=None, device='cuda') -> None:
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
        for param in self._model.parameters():
            param.requires_grad = False

        if self.model_config.sync_batch_norm:
          # Model was trained with Sync. Batch Norm.
          # Need to convert it otherwise parameters will load wrong.
          net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        self._model.eval()
        state_dict = torch.load(os.path.join(self.vehicle_config.model["dir"], self.vehicle_config.model["weights"]), map_location=self.device)
        self._model.load_state_dict(state_dict, strict=False)

        self.data_listener = data_listener

    @torch.inference_mode()
    def follow_route(self, tp_threshold=5.0, wp_threshold=7.0, visualize=False, debug=False):
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
        pred_target_speed = 0.0
        lidar_buffer = np.empty((0,3))

        finished = False

        while not finished:
            self._world.tick()

            if debug:
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
                cv2.imshow("camera front", data_dict['rgb_front'][1])
                # cv2.imshow("camera left", data_dict['rgb_left'][1])
                # cv2.imshow("camera right", data_dict['rgb_right'][1])
                cv2.waitKey(1)
    
            # Handling of lidar buffer, display
            lidar_buffer = np.vstack((lidar_buffer, out_data['lidar']))
            if lidar_buffer.shape[0] >= self.vehicle_config.lidar['buffer_threshold']:
                lidar_histogram = lidar_to_histogram_features(lidar_buffer[:,:3], self.model_config)
                lidar_buffer = np.empty((0,3))

                if visualize:
                    cv2.imshow("histogram", lidar_histogram[0])
                    cv2.waitKey(1)
                
                if self.dist(small_waypoints[small_wp_idx]) < wp_threshold: #replan                    
                    ego_vel = get_speed(self._vehicle)
                    freeze_vehicle_transform = self._vehicle.get_transform()
                    
                    waypoint = np.array([target_wp.transform.location.x, target_wp.transform.location.y, target_wp.transform.location.z])
                    bev_waypoint = self.waypoint_to_bev(waypoint, freeze_vehicle_transform)
                    preds = self._model(out_data['rgb'][:,:,:,:].to(self.device), #[B, 3, H, W]
                                        torch.tensor(lidar_histogram).unsqueeze(0).to(self.device),
                                        target_point=bev_waypoint, 
                                        ego_vel=torch.tensor(ego_vel).reshape(1,1).to(self.device))
                    
                    pred_wp = preds[2][0]
                    target_speed_uncertainty = F.softmax(preds[1][0], dim=0)
                    
                    if self.data_listener:       
                        data = {"preds":preds, 
                                "lidar":torch.tensor(lidar_histogram).unsqueeze(0).to(self.device),
                                "target_point":bev_waypoint, 
                                "ego_vel":torch.tensor(ego_vel).reshape(1,1).to(self.device),
                                "rgb":out_data['rgb'].to(self.device)}
                        self.data_listener.publish(data) 
                        while not self.data_listener.is_listening:
                            print("agent thread waiting", end='\r')

                    if self.model_config.use_target_speed_uncertainty:
                        uncertainty = target_speed_uncertainty.detach().cpu().numpy()
                        if uncertainty[0] > self.model_config.brake_uncertainty_threshold:
                            pred_target_speed = self.model_config.target_speeds[0]
                        else:
                            pred_target_speed = sum(uncertainty * self.model_config.target_speeds)
                    else:
                        pred_target_speed_index = torch.argmax(target_speed_uncertainty)
                        pred_target_speed = self.model_config.target_speeds[pred_target_speed_index]

                    small_waypoints = self.waypoint_to_bev(pred_wp, freeze_vehicle_transform, inverse=True)

                    if debug:
                        for ind in range(small_waypoints.shape[0]):
                            point = carla.Location(x=float(small_waypoints[ind][0]), y=float(small_waypoints[ind][1]), z=1)
                            self._world.debug.draw_point(point, size=0.2, life_time=1.0)

            # Get and apply control initially
            next_wp = small_waypoints[small_wp_idx]
            next_wp = carla.Transform(carla.Location(float(next_wp[0]), float(next_wp[1]), 0.0))
            control = self._controller.get_control((pred_target_speed*3.6, next_wp))  # multiply by 3.6 to go from m/s to km/h
            self._vehicle.apply_control(control)

            veh_dist = self.dist(target_wp)
            # If vehicle has reached waypoint, move to next waypoint
            if(veh_dist < tp_threshold):
                tp_idx += 1

                # Break once reaching last checkpoint
                if (tp_idx == n_wps):
                    finished = True
                else:
                    target_wp = self.route[tp_idx]

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
                _, compressed_image = cv2.imencode('.jpg', image)
                image = cv2.imdecode(compressed_image, cv2.IMREAD_UNCHANGED)

                # Add noise
                if self.vehicle_config.add_image_noise:
                    if np.random.rand() < self.vehicle_config.noise["add_noise_prob"]:
                        noise_type = np.random.choice(self.vehicle_config.noise["types"])
                        if noise_type in ['gaussian', 'speckle']:
                            image = add_random_noise(image, noise_type=noise_type, mean=self.vehicle_config.noise["mean"], var=self.vehicle_config.noise["var"])
                        else:
                            image = add_random_noise(image, noise_type=noise_type)
                    if np.random.rand() < self.vehicle_config.noise["add_patch_prob"]:
                        image = add_random_black_patches(image, max_n_patches=self.vehicle_config.noise["max_n_patches"], min_patch_size=self.vehicle_config.noise["min_patch_size"])
                    image = adjust_exposure(image, gamma_var=self.vehicle_config.noise["gamma_var"])

                rgb_pos = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Switch to pytorch channel first order
                rgb_pos = np.transpose(rgb_pos, (2, 0, 1))
                rgb.append(rgb_pos)
                # rgb.append(image)
            elif id.startswith('lidar'):
                out_data['lidar'] = lidar_to_ego_coordinates(data_dict[id][1],
                                                             lidar_pos=self.vehicle_config.lidar['position'],
                                                             lidar_rot=self.vehicle_config.lidar['rotation'],
                                                             intensity=False)
                
        rgb = np.concatenate(rgb, axis=1)
        # cv2.imshow("b", rgb.transpose(1,2,0))
        rgb = torch.from_numpy(rgb).to(self.device, dtype=torch.float32).unsqueeze(0)
        # cv2.waitKey(1)
        out_data['rgb'] = rgb

        return out_data
            
