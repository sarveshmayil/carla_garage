import carla

from vehicle import Vehicle
from control.pid_vehicle_control import PIDController

from agents.navigation.controller import VehiclePIDController
from agents.navigation.global_route_planner import GlobalRoutePlanner

from typing import Dict

def pid_controller(vehicle:carla.Actor, lat_args:Dict[str, float]=None, long_args:Dict[str, float]=None, ours=True):
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

    if ours:
        return PIDController(vehicle, lateral_args=args_lateral_dict, longitudinal_args=args_long_dict)
    else:
        return VehiclePIDController(vehicle, args_lateral=args_lateral_dict, args_longitudinal=args_long_dict)

if __name__ == "__main__":
    client = carla.Client("localhost", 2000)
    client.set_timeout(100)
    world:carla.World = client.load_world('Town01')

    wmap:carla.Map = world.get_map()
    spawn_points = wmap.get_spawn_points()
    a = carla.Location(spawn_points[0].location)
    b = carla.Location(spawn_points[100].location)

    vehicle = Vehicle(world=world)
    vehicle.spawn(location=a, vehicle_idx=7)
    vehicle.controller = pid_controller(vehicle._vehicle)
    # vehicle.set_controller_pid()
    vehicle.planner = GlobalRoutePlanner(wmap, sampling_resolution=2)
    vehicle.set_route(target=b)
    vehicle.follow_route(target_speed=30, visualize=True)
