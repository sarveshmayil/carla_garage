import carla
import os
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import math

from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF

from team_code.nav_planner import RoutePlanner
import team_code.transfuser_utils as t_u

from project.vehicle import Vehicle
from model.tf_model_minimal import LidarCenterNet
from project.config import GlobalConfig
from project.utils.misc import draw_waypoints
from project.utils.lidar import *
from project.utils.image_noise import *

from agents.tools.misc import get_speed

from typing import Optional, Dict, Union


# Configure pytorch for maximum performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True


class TFPPAgent(Vehicle):
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

        # Load model and set to eval
        self._model = LidarCenterNet(self.model_config).to(self.device)
        #self._model.load_state_dict(torch.load("model.pt"))

        for param in self._model.parameters():
            param.requires_grad = False

        if self.model_config.sync_batch_norm:
            # Model was trained with Sync. Batch Norm.
            # Need to convert it otherwise parameters will load wrong.
            self._model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self._model)
        self._model.eval()
        state_dict = torch.load(os.path.join(self.vehicle_config.model["dir"], self.vehicle_config.model["weights"]), map_location=self.device)
        self._model.load_state_dict(state_dict, strict=False)

        self.data_listener = data_listener

        self.step = -1
        self.initialized = False

        self.lidar_buffer = np.empty((0,3))

        # Initialize route planner
        self._route_planner = RoutePlanner(3.5, 6.0)

        # Filtering
        self.points = MerweScaledSigmaPoints(n=4, alpha=0.00001, beta=2, kappa=0, subtract=residual_state_x)
        self.ukf = UKF(dim_x=4,
                       dim_z=4,
                       fx=bicycle_model_forward,
                       hx=measurement_function_hx,
                       dt=self.model_config.carla_frame_rate,
                       points=self.points,
                       x_mean_fn=state_mean,
                       z_mean_fn=measurement_mean,
                       residual_x=residual_state_x,
                       residual_z=residual_measurement_h)

        # State noise, same as measurement because we
        # initialize with the first measurement later
        self.ukf.P = np.diag([0.5, 0.5, 0.000001, 0.000001])
        # Measurement noise
        self.ukf.R = np.diag([0.5, 0.5, 0.000000000000001, 0.000000000000001])
        self.ukf.Q = np.diag([0.0001, 0.0001, 0.001, 0.001])  # Model noise
        self.filter_initialized = False

    def set_route(self, target: carla.Location, start:Optional[carla.Location]=None):
        if start is None:
            start = self.location
        self.route = [(wp_tuple[0].transform, wp_tuple[1]) for wp_tuple in self._planner.trace_route(start, target)]
        self._route_planner.set_route(self.route, False)

    @torch.inference_mode()
    def follow_route(self, visualize=False, debug=False):
        """
        Function to use controller to follow a route.

        Args:
            target_speed: vehicle's target speed [km/h]
            threshold: distance threshold to check if vehicle has reached waypoint [m]
            visualize: flag to visualize the route

        Returns:
            None
        """
        finished = False

        while not finished:
            self._world.tick()

            if debug:
                draw_waypoints(self._world, [self._route_planner.route[0][0]], life_time=1.0)
            
            if self.data_listener:
                if self.data_listener.end_epoch:
                    break
            
            # Offset spectator camera to follow car
            if self._world.get_settings().no_rendering_mode == False:
                spectator_offset = -10 * self._vehicle.get_transform().rotation.get_forward_vector() + \
                                     5 * self._vehicle.get_transform().rotation.get_up_vector()
                spectator_transform = self._vehicle.get_transform()
                spectator_transform.location += spectator_offset
                self._world.get_spectator().set_transform(spectator_transform)
            
            # Collect data and post-process
            data_dict = self._sensor_interface.get_data()
            control = self.run_step(data_dict, debug=debug)

            if visualize:
                cv2.imshow("camera front", data_dict['rgb_front'][1])
                cv2.waitKey(1)

            self._vehicle.apply_control(control)

            # CARLA will not let the car drive in the initial frames.
            # We set the action to brake so that the filter does not get confused.
            if self.step < self.model_config.inital_frames_delay:
                self.control = carla.VehicleControl(0.0, 0.0, 1.0)
            else:
                self.control = control

            if len(self._route_planner.route) < 2:
                finished = True
            
        self._vehicle.apply_control(carla.VehicleControl(0.0, 0.0, 1.0))

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

            elif id.startswith('lidar'):
                out_data['lidar'] = lidar_to_ego_coordinates(data_dict[id][1],
                                                             lidar_pos=self.vehicle_config.lidar['position'],
                                                             lidar_rot=[0.0, 0.0, -90.0],
                                                             intensity=False)
                
        rgb = np.concatenate(rgb, axis=1)
        rgb = torch.from_numpy(rgb).to(self.device, dtype=torch.float32).unsqueeze(0)
        out_data['rgb'] = rgb

        gps_pos = self._route_planner.convert_gps_to_carla(data_dict['gps'][1][:2])
        out_data['vehicle_pos'] = gps_pos
        # speed = data_dict['speed'][1]['speed']
        speed = get_speed(self._vehicle)
        out_data['speed'] = torch.FloatTensor([speed]).to(self.device, dtype=torch.float32)
        compass = t_u.preprocess_compass(data_dict['imu'][1][-1])
        out_data['compass'] = compass

        if not self.filter_initialized:
            self.ukf.x = np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed])
            self.filter_initialized = True

        self.ukf.predict(steer=self.control.steer, throttle=self.control.throttle, brake=self.control.brake)
        self.ukf.update(np.array([gps_pos[0], gps_pos[1], t_u.normalize_angle(compass), speed]))
        filtered_state = self.ukf.x

        out_data['gps'] = filtered_state[0:2]
        
        waypoint_route = self._route_planner.run_step(filtered_state[0:2])

        # print(waypoint_route[0], waypoint_route[1])
        if len(waypoint_route) > 1:
            target_point, _ = waypoint_route[1]
        else:
            target_point, _ = waypoint_route[0]

        ego_target_point = t_u.inverse_conversion_2d(target_point, out_data['gps'], out_data['compass'])
        ego_target_point = torch.from_numpy(ego_target_point[np.newaxis]).to(self.device, dtype=torch.float32)

        out_data['target_point'] = ego_target_point

        return out_data

    @torch.inference_mode()
    def run_step(
        self,
        input_data:Dict[str, Union[np.ndarray, torch.Tensor]],
        debug:bool=False
    ) -> carla.VehicleControl:
        self.step += 1

        if not self.initialized:
            control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            self.control = control
            tick_data = self.data_tick(input_data)
            self.initialized = True
            return control
        
        tick_data = self.data_tick(input_data)

        self.lidar_buffer = np.vstack((self.lidar_buffer, tick_data['lidar']))

        # If enough lidar points are collected, run model prediction
        if self.lidar_buffer.shape[0] >= self.vehicle_config.lidar['buffer_threshold']:
            lidar_histogram = lidar_to_histogram_features(self.lidar_buffer[:,:3], self.model_config)
            self.lidar_buffer = np.empty((0,3))
        # Otherwise, continue using same control
        else:
            return self.control
        
        freeze_vehicle_transform = self._vehicle.get_transform()
        
        preds = self._model(rgb=tick_data['rgb'][:,:,:256,:], #[B, 3, H, W]
                            lidar_bev=torch.tensor(lidar_histogram).unsqueeze(0).to(self.device),
                            target_point=tick_data['target_point'].reshape(-1), 
                            ego_vel=tick_data['speed'].reshape(1,1))
        
        pred_wp = preds[2][0]
        target_speed_uncertainty = F.softmax(preds[1][0], dim=0)

        if debug:
            small_waypoints = self.waypoint_to_bev(pred_wp, freeze_vehicle_transform, inverse=True)
            for ind in range(small_waypoints.shape[0]):
                point = carla.Location(x=float(small_waypoints[ind][0]), y=float(small_waypoints[ind][1]), z=1)
                self._world.debug.draw_point(point, size=0.2, life_time=1.0)

        if self.data_listener:       
            data = {"preds": preds, 
                    "lidar": torch.tensor(lidar_histogram).unsqueeze(0).to(self.device),
                    "target_point": tick_data['target_point'].reshape(-1), 
                    "ego_vel": torch.tensor(tick_data['speed']).reshape(1,1).to(self.device),
                    "rgb": tick_data['rgb'].to(self.device)}
            self.data_listener.publish(data) 
            while not self.data_listener.is_listening:
                print("", end='\r')

        if self.model_config.use_target_speed_uncertainty:
            uncertainty = target_speed_uncertainty.detach().cpu().numpy()
            if uncertainty[0] > self.model_config.brake_uncertainty_threshold:
                pred_target_speed = self.model_config.target_speeds[0]
            else:
                pred_target_speed = sum(uncertainty * self.model_config.target_speeds)
        else:
            pred_target_speed_index = torch.argmax(target_speed_uncertainty)
            pred_target_speed = self.model_config.target_speeds[pred_target_speed_index]

        pred_wp = pred_wp.detach().cpu().numpy()
        pred_angle = -math.degrees(math.atan2(-pred_wp[1][1], pred_wp[1][0])) / 90.0
        steer, throttle, brake = self._model.control_pid_direct(pred_target_speed, pred_angle, tick_data['speed'])
        control = carla.VehicleControl(steer=float(steer), throttle=float(throttle), brake=float(brake))

        # print(tick_data['target_point'], pred_wp[1], pred_target_speed)

        # CARLA will not let the car drive in the initial frames.
        # We set the action to brake so that the filter does not get confused.
        if self.step < self.model_config.inital_frames_delay:
            self.control = carla.VehicleControl(0.0, 0.0, 1.0)
        else:
            self.control = control

        return control

# Filter Functions
def bicycle_model_forward(x, dt, steer, throttle, brake):
    # Kinematic bicycle model.
    # Numbers are the tuned parameters from World on Rails
    front_wb = -0.090769015
    rear_wb = 1.4178275

    steer_gain = 0.36848336
    brake_accel = -4.952399
    throt_accel = 0.5633837

    locs_0 = x[0]
    locs_1 = x[1]
    yaw = x[2]
    speed = x[3]

    if brake:
        accel = brake_accel
    else:
        accel = throt_accel * throttle

    wheel = steer_gain * steer

    beta = math.atan(rear_wb / (front_wb + rear_wb) * math.tan(wheel))
    next_locs_0 = locs_0.item() + speed * math.cos(yaw + beta) * dt
    next_locs_1 = locs_1.item() + speed * math.sin(yaw + beta) * dt
    next_yaws = yaw + speed / rear_wb * math.sin(beta) * dt
    next_speed = speed + accel * dt
    next_speed = next_speed * (next_speed > 0.0)  # Fast ReLU

    next_state_x = np.array([next_locs_0, next_locs_1, next_yaws, next_speed])

    return next_state_x

def measurement_function_hx(vehicle_state):
    '''
    For now we use the same internal state as the measurement state
    :param vehicle_state: VehicleState vehicle state variable containing
                          an internal state of the vehicle from the filter
    :return: np array: describes the vehicle state as numpy array.
                       0: pos_x, 1: pos_y, 2: rotatoion, 3: speed
    '''
    return vehicle_state

def state_mean(state, wm):
    '''
    We use the arctan of the average of sin and cos of the angle to calculate
    the average of orientations.
    :param state: array of states to be averaged. First index is the timestep.
    :param wm:
    :return:
    '''
    x = np.zeros(4)
    sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
    sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
    x[0] = np.sum(np.dot(state[:, 0], wm))
    x[1] = np.sum(np.dot(state[:, 1], wm))
    x[2] = math.atan2(sum_sin, sum_cos)
    x[3] = np.sum(np.dot(state[:, 3], wm))

    return x

def measurement_mean(state, wm):
    '''
    We use the arctan of the average of sin and cos of the angle to
    calculate the average of orientations.
    :param state: array of states to be averaged. First index is the
    timestep.
    '''
    x = np.zeros(4)
    sum_sin = np.sum(np.dot(np.sin(state[:, 2]), wm))
    sum_cos = np.sum(np.dot(np.cos(state[:, 2]), wm))
    x[0] = np.sum(np.dot(state[:, 0], wm))
    x[1] = np.sum(np.dot(state[:, 1], wm))
    x[2] = math.atan2(sum_sin, sum_cos)
    x[3] = np.sum(np.dot(state[:, 3], wm))

    return x

def residual_state_x(a, b):
    y = a - b
    y[2] = t_u.normalize_angle(y[2])
    return y

def residual_measurement_h(a, b):
    y = a - b
    y[2] = t_u.normalize_angle(y[2])
    return y