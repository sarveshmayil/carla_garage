import carla

from vehicle import Vehicle
from autonomous_agent import Agent

from agents.navigation.global_route_planner import GlobalRoutePlanner
from traffic import TrafficUwU
import random
from leaderboard.utils.route_indexer import RouteIndexer



if __name__ == "__main__":
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    
    # world:carla.World = client.load_world("Town01")

    route_indexer = RouteIndexer("leaderboard/data/training/routes/s1/Town01_Scenario1.xml", "leaderboard/data/training/scenarios/s1/Town01_Scenario1.json", 1)
    route = route_indexer.next()
    world:carla.World = client.load_world(route.town)

    settings = world.get_settings()
    settings.no_rendering_mode = False
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    wmap:carla.Map = world.get_map()
    #spawn_points = wmap.get_spawn_points()
    #a = carla.Location(spawn_points[3].location)
    #b = carla.Location(spawn_points[100].location)
    weather = carla.WeatherParameters(
                                        cloudiness=random.random()*100,
                                        precipitation=random.random()*100,
                                        sun_altitude_angle=(random.random()-0.5)*180,
                                        fog_density = random.random()*50)
    world.set_weather(weather)

    #traffic_man = TrafficUwU(idxA = 0, idxB = 100)
    
    # vehicle = Vehicle(world=world)
    vehicle = Agent(world=world)
    # vehicle.spawn(location=carla.Location(spawn_points[3].location))
    start = carla.Location(route.trajectory[0].x, route.trajectory[0].y, route.trajectory[0].z+1) # z+1 to avoid collision with ground
    target = route.trajectory[-1]
    vehicle.spawn(location=start, rotation=route.rotations[0])
    vehicle.set_controller_pid()
    vehicle.planner = GlobalRoutePlanner(wmap, sampling_resolution=10)
    vehicle.set_route(start=start, target=target)
    # vehicle.follow_route(target_speed=30, visualize=True, debug=True)  # For Vehicle

    #traffic_man.spawn_traffic(numCars= 30)
    vehicle.follow_route(visualize=True, debug=False)  # For Agent

