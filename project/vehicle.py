import carla
from math import sqrt

from control.controller_base import BaseController
from control.pid_vehicle_control import PIDController
from agents.tools.misc import draw_waypoints

from typing import Tuple, Union, Optional, Dict

class Vehicle():
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
        if not isinstance(location, carla.Location):
            location = carla.Location(*location)
        if not isinstance(rotation, carla.Rotation):
            rotation = carla.Rotation(*rotation)

        spawnPoint = carla.Transform(location, rotation)

        blueprint_library = self._world.get_blueprint_library()
        blueprint:carla.ActorBlueprint = blueprint_library.filter('vehicle.*')[vehicle_idx]
        self._vehicle = self._world.spawn_actor(blueprint, spawnPoint)

    def dist(self, target) -> float:
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
        self.route = [wp[0] for wp in self._planner.trace_route(self.location, target)]

    def show_route(self):
        print("BEFORE")
        draw_waypoints(self._world, self.route)
        print("AFTER")

    def follow_route(self, target_speed=30.0, threshold=3.5, max_iters=1000):
        if self.route is None:
            raise Exception("No route was set. Use `set_route()` first.")
        
        i = 0
        target_wp = self.route[0]
        while True:
            if (i == len(self.route)-1):
                break

            veh_dist = self.dist(target_wp)
            control = self._controller.get_control((target_speed, target_wp))
            self._vehicle.apply_control(control)

            if(veh_dist < threshold):
                control = self._controller.get_control((target_speed, target_wp))
                self._vehicle.apply_control(control)
                i += 1
                target_wp = self.route[i]

        # Stop vehicle at final waypoint
        control = self._controller.get_control((0.0, self.route[-1]))
        self._vehicle.apply_control(control)

    def __del__(self):
        self._vehicle.destroy()

            

