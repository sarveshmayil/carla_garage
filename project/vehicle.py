import carla
from math import sqrt

from control.controller_base import BaseController
from agents.tools.misc import draw_waypoints

from typing import Tuple, Union

class Vehicle():
    def __init__(self) -> None:
        self.vehicle:carla.Actor = None
        self.route = []

    @property
    def location(self) -> carla.Location:
        return self.vehicle.get_location()
    
    @property
    def controller(self) -> BaseController:
        return self.controller
    
    @controller.setter
    def controller(self, controller:BaseController):
        self.controller = controller
    
    @property
    def planner(self):
        return self.planner
    
    @planner.setter
    def planner(self, planner):
        self.planner = planner

    def spawn(
        self,
        world:carla.World,
        location:Union[Tuple[float], carla.Location],
        rotation:Union[Tuple[float], carla.Rotation]=(0.0, 0.0, 0.0),
        vehicle_idx=0
    ):
        if not isinstance(location, carla.Location):
            location = carla.Location(*location)
        if not isinstance(rotation, carla.Rotation):
            rotation = carla.Rotation(*rotation)

        spawnPoint = carla.Transform(location, rotation)

        blueprint_library = self.world.get_blueprint_library()
        blueprint:carla.ActorBlueprint = blueprint_library.filter('vehicle.*')[vehicle_idx]
        self.vehicle = world.spawn_actor(blueprint, spawnPoint)

    def dist(self, target) -> float:
        vehicle_loc = self.location
        dist = sqrt( (target.transform.location.x - vehicle_loc.x)**2 + (target.transform.location.y - vehicle_loc.y)**2 )
        return dist

    def set_route(self, target:carla.Location):
        self.route = [wp[0] for wp in self.planner.trace_route(self.location, target)]

    def show_route(self, world):
        draw_waypoints(world, self.route)

    def follow_route(self, target_speed=30.0, threshold=2.0, max_iters=10):
        if self.route is None:
            raise Exception("No route was set. Use `set_route()` first.")
        
        for wp in self.route:
            veh_dist = self.dist(wp)
            i = 0
            while (veh_dist > threshold and i < max_iters):
                control = self.controller.get_control((target_speed, wp))
                self.vehicle.apply_control(control)
                veh_dist = self.dist(wp)
                i += 1

        # Stop vehicle at final waypoint
        control = self.controller.get_control((0.0, self.route[-1]))
        self.vehicle.apply_control(control)

            

