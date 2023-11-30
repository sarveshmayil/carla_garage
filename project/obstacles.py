import carla
import random

class StaticObstacles:

    def __init__(self, world, spawn_points) -> None:
        self.world = world
        self.spawn_points = spawn_points[5:] #make sure it doesn't spawn close to starting
        self.obstacles = []

    def spawn_obstacles_random(self, num=30, filter='static.prop.*', rand_range=1.0):
        select_spawns = random.sample(self.spawn_points, k=num)
        for spawn in select_spawns:
            loc = spawn.location
            loc.x += random.uniform(-rand_range, rand_range)
            loc.y += random.uniform(-rand_range, rand_range)
            yaw = random.randint(-180, 180)
            self.spawn_obstacle(loc, yaw, filter=filter)

    def spawn_obstacle(self, location, yaw, filter='static.prop.*', verbose=False):
        blueprint_library = self.world.get_blueprint_library()
        obstacle_bp = random.choice(blueprint_library.filter(filter))
        
        obstacle_transform = carla.Transform(location, carla.Rotation(yaw=yaw))
        obstacle = self.world.spawn_actor(obstacle_bp, obstacle_transform)

        if verbose:
            print("Added obstacle ", obstacle, " at location", location)

        self.obstacles.append(obstacle)
    
    def destroy_obstacles(self):
        for obstacle in self.obstacles:
            obstacle.destroy()