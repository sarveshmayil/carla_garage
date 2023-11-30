import carla

from vehicle import Vehicle
from autonomous_agent import Agent

from agents.navigation.global_route_planner import GlobalRoutePlanner
from obstacles import StaticObstacles

if __name__ == "__main__":
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world:carla.World = client.load_world('Town01')

    settings = world.get_settings()
    settings.no_rendering_mode = False
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    wmap:carla.Map = world.get_map()
    spawn_points = wmap.get_spawn_points()
    a = carla.Location(spawn_points[3].location)
    b = carla.Location(spawn_points[100].location)

    static_obst = StaticObstacles(world, spawn_points)
    #spawn obstacles randomly
    static_obst.spawn_obstacles_random(num=100, filter="static.prop.streetbarrier")
    #spawn obstacle at specific location
    #loc = carla.Location(a.x+15, a.y, a.z)
    #static_obst.spawn_obstacle(loc, yaw=90, filter="static.prop.streetbarrier")
    
    # vehicle = Vehicle(world=world)
    vehicle = Agent(world=world)
    vehicle.spawn(location=carla.Location(spawn_points[3].location))
    vehicle.set_controller_pid()
    vehicle.planner = GlobalRoutePlanner(wmap, sampling_resolution=10)
    vehicle.set_route(start=a, target=b)
    # vehicle.follow_route(target_speed=30, visualize=True, debug=True)  # For Vehicle
    vehicle.follow_route(visualize=True, debug=False)  # For Agent
