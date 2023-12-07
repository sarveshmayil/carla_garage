import carla
# from carla.command import SpawnActor, SetAutopilot, FutureActor
import logging
from numpy import random
import time
random.seed(int(time.time()))

SpawnActor = carla.command.SpawnActor
SetAutopilot = carla.command.SetAutopilot
FutureActor = carla.command.FutureActor


class TrafficUwU:
    def __init__(self, idxA, idxB, host = "localhost", port = 2000, tm_port = 8000):
        
        # Start and end points indices for route generation
        self.idxA = idxA
        self.idxB = idxB

        self.client = carla.Client(host, port)
        self.world = self.client.get_world()
        
        self.traffic_manager = self.client.get_trafficmanager(tm_port)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)

        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []

        # self.blueprints = sorted(self._get_actor_blueprints(self.world), key=lambda bp: bp.id)
        self.blueprintsVehicles = self._get_actor_blueprints(self.world, filter = "vehicle.*", generation = "All")
        self.blueprintsWalkers = self._get_actor_blueprints(self.world, filter = "walker.pedestrian.*", generation = "2")

    def get_car_spawn_points(self, numCars):
    
        if self.idxA > self.idxB:
            spawn_points_list = self.world.get_map().get_spawn_points()[self.idxB:self.idxA+1]
        else:
            spawn_points_list = self.world.get_map().get_spawn_points()[self.idxA:self.idxB+1]

        return [spawn_points_list[idx] for idx in random.randint(0, abs(self.idxA - self.idxB), numCars)]

    def spawn_traffic(self, numCars= 30):
        batch = []
        spawn_points = self.get_car_spawn_points(numCars)
        for n, transform in enumerate(spawn_points):
            blueprint = random.choice(self.blueprintsVehicles)
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, self.traffic_manager.get_port())))
            
        for response in self.client.apply_batch_sync(batch, False):
            if response.error:
                logging.error(response.error)
            else:
                self.vehicles_list.append(response.actor_id)

    def _get_actor_blueprints(self, world, filter, generation):
        bps = world.get_blueprint_library().filter(filter)

        if generation.lower() == "all":
            return bps

        # If the filter returns only one bp, we assume that this one needed
        # and therefore, we ignore the generation
        if len(bps) == 1:
            return bps

        try:
            int_generation = int(generation)
            # Check if generation is in available generations
            if int_generation in [1, 2]:
                bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
                return bps
            else:
                print("   Warning! Actor Generation is not valid. No actor will be spawned.")
                return []
        except:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
