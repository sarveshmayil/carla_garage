import math
import carla
from numpy import ndarray

from typing import List, Union

def draw_waypoints(world, waypoints:List[Union[carla.Waypoint, carla.Transform]], z=0.5, life_time=1.0):
    """
    Draw a list of waypoints at a certain height given in z.

        :param world: carla.world object
        :param waypoints: list or iterable container with the waypoints to draw
        :param z: height in meters
        :param life_time: life time of arrows (set to 0 for permanent)
    """
    for wpt in waypoints:
        if isinstance(wpt, carla.Waypoint):
            wpt = wpt.transform
        elif isinstance(wpt, ndarray):
            wpt = carla.Transform(carla.Location(x=wpt[0], y=wpt[1], z=0.0))
        begin = wpt.location + carla.Location(z=z)
        angle = math.radians(wpt.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=life_time)