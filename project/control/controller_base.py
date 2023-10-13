from abc import ABC, abstractmethod

from carla import VehicleControl

class BaseController(ABC):
    def __init__(self, vehicle, max_throttle=0.75, max_brake=0.3, max_steering=0.8, max_steer_diff=0.1) -> None:
        super().__init__()
        self.max_brake = max_brake
        self.max_throttle = max_throttle
        self.max_steer = max_steering
        self.max_steer_diff = max_steer_diff
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()

    @abstractmethod
    def get_control(self, input) -> VehicleControl:
        """
        Gets controls for vehicle based on input.
        Returns carla.VehicleControl object.
        """
        pass