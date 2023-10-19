import carla
from math import sqrt

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
        while True:
            
            camera_offset = -5 * self._vehicle.get_transform().rotation.get_forward_vector() + \
                            2 * self._vehicle.get_transform().rotation.get_up_vector()
            camera_transform = self._vehicle.get_transform()
            camera_transform.location += camera_offset
            self.get_world().get_spectator().set_transform(camera_transform)

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

    def __del__(self):
        self._vehicle.destroy()

            

