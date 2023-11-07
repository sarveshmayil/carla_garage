import carla
import numpy as np
from math import acos
from agents.tools.misc import get_speed

from .controller_base import BaseController
from .pid import PID

from typing import Dict, Tuple, Union


class PIDController(BaseController):
    def __init__(self, vehicle:carla.Actor, lateral_args:Dict[str,float], longitudinal_args:Dict[str,float], offset=0, **kwargs) -> None:
        super().__init__(vehicle=vehicle, **kwargs)
        self.long_controller = PIDLongitudinalController(**longitudinal_args)
        self.lat_controller = PIDLateralController(offset=offset, **lateral_args)

        self.last_steer_command = self._vehicle.get_control().steer

    def get_control(self, input:Tuple[float,Union[carla.Transform,carla.Waypoint]]) -> carla.VehicleControl:
        """
        Gets controls for vehicle based on input.

        Args:
            input: Tuple of (target speed:float, waypoint:carla.Transform/Waypoint)

        Returns:
            control: carla.VehicleControl instance
        """
        target_speed, waypoint = input
        if isinstance(waypoint, carla.Waypoint):
            waypoint = waypoint.transform

        control = carla.VehicleControl()
        current_speed = get_speed(self._vehicle)  # speed in kph
        accel_command = self.long_controller.run_step(target_speed, current_speed)
        steer_command = self.lat_controller.run_step(waypoint, self._vehicle.get_transform())

        # Cap throttle and brake commands based on specified max values
        if accel_command > 0.0:
            control.throttle = min(accel_command, self.max_throttle)
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = min(abs(accel_command), self.max_brake)

        steer_diff = steer_command - self.last_steer_command
        if steer_diff > self.max_steer_diff:
            steer_command = self.last_steer_command + self.max_steer_diff
        elif steer_diff < -self.max_steer_diff:
            steer_command = self.last_steer_command - self.max_steer_diff

        steer_command = np.clip(steer_command, -self.max_steer, self.max_steer)
        control.steer = steer_command
        control.reverse = False
        control.hand_brake = False
        control.manual_gear_shift = False

        self.last_steer_command = steer_command

        return control

class PIDLongitudinalController(PID):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def run_step(self, target_speed:float, current_speed:float) -> float:
        return super().run_step(target_speed, current_speed)
    
class PIDLateralController(PID):
    def __init__(self, offset=0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._offset = offset

    def run_step(self, waypoint:carla.Transform, veh_transform:carla.Transform) -> float:
        veh_loc = veh_transform.location
        veh_vec = veh_transform.get_forward_vector()
        veh_vec = np.array([veh_vec.x, veh_vec.y, 0.0])  # remove any z velocity

        if self._offset != 0:
            # Displace the wp to the side
            r_vec = waypoint.get_right_vector()
            wp_loc = waypoint.location + carla.Location(x=self._offset*r_vec.x,
                                                        y=self._offset*r_vec.y)
        else:
            wp_loc = waypoint.location

        # difference between vehicle loc and waypoint
        wp_vec = np.array([wp_loc.x - veh_loc.x,
                           wp_loc.y - veh_loc.y,
                           0.0])
        
        dot_prod_norm = np.linalg.norm(wp_vec) * np.linalg.norm(veh_vec)
        if dot_prod_norm == 0.0:
            dot_prod = 1.0  # can't divide by 0
        else:
            dot_prod = acos(np.clip(np.dot(wp_vec, veh_vec) / dot_prod_norm, -1.0, 1.0))
        
        # If cross product between veh_vec and wp_vec is
        #           > 0, then need to turn left
        #           < 0, then need to turn right
        cross_prod = np.cross(veh_vec, wp_vec)
        if cross_prod[2] < 0:
            dot_prod *= -1.0

        self._error_buffer.append(dot_prod)
        if len(self._error_buffer) >= 2:
            d_term = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            i_term = sum(self._error_buffer) * self._dt
        else:
            d_term = 0.0
            i_term = 0.0

        return np.clip((self._k_p * dot_prod) + (self._k_d * d_term) + (self._k_i * i_term), -1.0, 1.0)